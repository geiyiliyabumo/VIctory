//! AMD FidelityFX SDK 2.x frame generation and interpolation swap chain.
//! Enable with VICTORY_FSR_FRAMEGEN=1. The signed runtime DLLs are loaded
//! from VICTORY_FIDELITYFX_DIR or the executable's fidelityfx directory.

use std::collections::VecDeque;
use std::ffi::{CStr, OsString, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::Foundation::{FreeLibrary, HANDLE, HMODULE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows_sys::core::GUID;

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

use std::time::{Duration, Instant};

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

/// Camera and dispatch parameters captured from the title's FidelityFX upscaler dispatch.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub render: [u32; 2],
    pub output: [u32; 2],
    pub jitter: [f32; 2],
    pub motion_scale: [f32; 2],
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
    pub reset: bool,
    pub inverted_depth: bool,
    pub infinite_depth: bool,
    pub display_res_mv: bool,
    pub mv_jitter_cancelled: bool,
    pub hdr: bool,
}

struct Inputs {
    depth: (ID3D12Resource, u32),
    motion: (ID3D12Resource, u32),
    frame_ms: f32,
    camera: Camera,
}

/// Bounded FIFO of recent upscaler dispatches so async submission pipelining
/// (recording frame N+1 before frame N presents) never drops frame N's inputs
/// and causes frame generation to toggle off between presents.
struct Latest<T>(VecDeque<(Instant, T)>);

impl<T> Latest<T> {
    const MAX_QUEUED: usize = 4;

    const fn new() -> Self {
        Self(VecDeque::new())
    }

    fn note(&mut self, now: Instant, inputs: T) {
        if self.0.len() >= Self::MAX_QUEUED {
            self.0.pop_front();
        }
        self.0.push_back((now, inputs));
    }

    fn take(&mut self, now: Instant) -> Option<T> {
        while let Some((noted, inputs)) = self.0.pop_front() {
            if now.duration_since(noted) < Duration::from_millis(2000) {
                return Some(inputs);
            }
        }
        None
    }
}

static INPUTS: Mutex<Latest<Inputs>> = Mutex::new(Latest::new());

/// Saves the most recent successful upscaler dispatch for the next present.
pub fn note_upscale(
    depth: (ID3D12Resource, u32),
    motion: (ID3D12Resource, u32),
    frame_ms: f32,
    camera: Camera,
) {
    let mut latest = INPUTS.lock().unwrap_or_else(|e| e.into_inner());
    latest.note(
        Instant::now(),
        Inputs {
            depth,
            motion,
            frame_ms,
            camera,
        },
    );
}

fn fresh_inputs() -> Option<Inputs> {
    let mut latest = INPUTS.lock().unwrap_or_else(|e| e.into_inner());
    latest.take(Instant::now())
}

const LOADER: &str = "amd_fidelityfx_loader_dx12.dll";
const FRAMEGEN: &str = "amd_fidelityfx_framegeneration_dx12.dll";

// Structure types (ffx_api.h, ffx_framegeneration.h, ffx_api_dx12.h,
// ffx_api_framegeneration_dx12.h of FidelityFX SDK 2.x).
const CREATE_BACKEND_DX12: u64 = 0x0000_0002;
const CREATE_FRAMEGENERATION: u64 = 0x0002_0001;
const CONFIGURE_FRAMEGENERATION: u64 = 0x0002_0002;
const DISPATCH_PREPARE_V2: u64 = 0x0002_000c;
const CREATE_FRAMEGENERATION_VERSION: u64 = 0x0002_000e;
/// `ffxCreateContextDescFrameGenerationHudless` (a header and one format, laid
/// out as [`Version`]).
const CREATE_FRAMEGENERATION_HUDLESS: u64 = 0x0002_0009;
const CREATE_SWAPCHAIN_FOR_HWND_DX12: u64 = 0x0003_0006;
const CREATE_SWAPCHAIN_VERSION_DX12: u64 = 0x0003_000b;
const DISPATCH_WAIT_FOR_PRESENTS_DX12: u64 = 0x0003_0007;
/// The versions our structures are those of.
const FRAMEGENERATION_VERSION: u32 = (4 << 22) | 2;
const SWAPCHAIN_VERSION: u32 = 4 << 22;

// Frame generation context creation flags.
const FG_DISPLAY_RESOLUTION_MV: u32 = 1 << 1;
const FG_MV_JITTER_CANCELLATION: u32 = 1 << 2;
const FG_DEPTH_INVERTED: u32 = 1 << 3;
const FG_DEPTH_INFINITE: u32 = 1 << 4;
const FG_HIGH_DYNAMIC_RANGE: u32 = 1 << 5;
/// Dispatch flag: draws the tear and pacing lines (`VICTORY_FSR_FRAMEGEN_DEBUG=1`).
const FG_DRAW_DEBUG_TEAR_LINES: u32 = 1 << 0;
const FG_DRAW_DEBUG_PACING_LINES: u32 = 1 << 4;

// Resource states and types.
const STATE_COMMON: u32 = 1 << 0;
const STATE_UNORDERED_ACCESS: u32 = 1 << 1;
const STATE_COMPUTE_READ: u32 = 1 << 2;
const STATE_PIXEL_READ: u32 = 1 << 3;
const STATE_COPY_SRC: u32 = 1 << 4;
const STATE_COPY_DEST: u32 = 1 << 5;
const STATE_RENDER_TARGET: u32 = 1 << 8;
const STATE_DEPTH_ATTACHMENT: u32 = 1 << 9;
const TYPE_TEXTURE2D: u32 = 2;
const USAGE_UAV: u32 = 1 << 1;
const USAGE_DEPTHTARGET: u32 = 1 << 2;
const USAGE_STENCILTARGET: u32 = 1 << 5;

const OK: u32 = 0;

#[repr(C)]
struct Header {
    kind: u64,
    next: *mut Header,
}

impl Header {
    fn new(kind: u64) -> Self {
        Header {
            kind,
            next: std::ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Dims {
    width: u32,
    height: u32,
}

impl From<[u32; 2]> for Dims {
    fn from([width, height]: [u32; 2]) -> Self {
        Dims { width, height }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Floats {
    x: f32,
    y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Rect {
    left: i32,
    top: i32,
    width: i32,
    height: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ResourceDescription {
    kind: u32,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    mip_count: u32,
    flags: u32,
    usage: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: ResourceDescription,
    state: u32,
}

impl Resource {
    const NONE: Resource = Resource {
        resource: std::ptr::null_mut(),
        description: ResourceDescription {
            kind: 0,
            format: 0,
            width: 0,
            height: 0,
            depth: 0,
            mip_count: 0,
            flags: 0,
            usage: 0,
        },
        state: STATE_COMMON,
    };

    /// `ffxApiGetResourceDX12` of a 2D texture in D3D12 state `state`.
    fn texture(texture: &ID3D12Resource, state: u32) -> Resource {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let usage = match desc.Format {
            DXGI_FORMAT_D16_UNORM | DXGI_FORMAT_D32_FLOAT => USAGE_DEPTHTARGET,
            DXGI_FORMAT_D24_UNORM_S8_UINT | DXGI_FORMAT_D32_FLOAT_S8X24_UINT => {
                USAGE_DEPTHTARGET | USAGE_STENCILTARGET
            }
            _ => 0,
        } | if desc
            .Flags
            .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        {
            USAGE_UAV
        } else {
            0
        };
        Resource {
            resource: texture.as_raw(),
            description: ResourceDescription {
                kind: TYPE_TEXTURE2D,
                format: surface_format(desc.Format),
                width: desc.Width as u32,
                height: desc.Height,
                depth: u32::from(desc.DepthOrArraySize),
                mip_count: u32::from(desc.MipLevels),
                flags: 0,
                usage,
            },
            state: ffx_state(state),
        }
    }
}

#[repr(C)]
struct CreateSwapChainForHwnd {
    header: Header,
    swap_chain: *mut *mut c_void,
    hwnd: HWND,
    desc: *const DXGI_SWAP_CHAIN_DESC1,
    fullscreen: *const c_void,
    factory: *mut c_void,
    queue: *mut c_void,
}

#[repr(C)]
struct Version {
    header: Header,
    version: u32,
}

#[repr(C)]
struct CreateFrameGeneration {
    header: Header,
    flags: u32,
    display_size: Dims,
    max_render_size: Dims,
    back_buffer_format: u32,
}

#[repr(C)]
struct CreateBackend {
    header: Header,
    device: *mut c_void,
}

type FrameGenerationCallback = unsafe extern "C" fn(params: *mut Header, user: *mut c_void) -> u32;

#[repr(C)]
struct ConfigureFrameGeneration {
    header: Header,
    swap_chain: *mut c_void,
    present_callback: *const c_void,
    present_callback_user: *mut c_void,
    frame_generation_callback: Option<FrameGenerationCallback>,
    frame_generation_callback_user: *mut c_void,
    frame_generation_enabled: bool,
    allow_async_workloads: bool,
    hudless_color: Resource,
    flags: u32,
    only_present_generated: bool,
    generation_rect: Rect,
    frame_id: u64,
}

#[repr(C)]
struct PrepareV2 {
    header: Header,
    frame_id: u64,
    flags: u32,
    command_list: *mut c_void,
    render_size: Dims,
    jitter_offset: Floats,
    motion_vector_scale: Floats,
    frame_time_delta: f32,
    reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_angle_vertical: f32,
    view_space_to_meters_factor: f32,
    depth: Resource,
    motion_vectors: Resource,
    camera_position: [f32; 3],
    camera_up: [f32; 3],
    camera_right: [f32; 3],
    camera_forward: [f32; 3],
}

// x64 layouts and field offsets from the FidelityFX SDK 2.x headers.
const _: () = {
    use std::mem::{align_of, offset_of};
    assert!(align_of::<Header>() == 8);
    assert!(align_of::<ResourceDescription>() == 4);
    assert!(align_of::<Resource>() == 8);
    assert!(align_of::<CreateFrameGeneration>() == 8);
    assert!(align_of::<CreateSwapChainForHwnd>() == 8);
    assert!(align_of::<ConfigureFrameGeneration>() == 8);
    assert!(align_of::<PrepareV2>() == 8);
    assert!(size_of::<CreateBackend>() == 24 && align_of::<CreateBackend>() == 8);
    assert!(size_of::<Version>() == 24 && align_of::<Version>() == 8);
    assert!(size_of::<Header>() == 16);
    assert!(size_of::<ResourceDescription>() == 32);
    assert!(size_of::<Resource>() == 48);
    assert!(size_of::<CreateFrameGeneration>() == 40);
    assert!(size_of::<CreateSwapChainForHwnd>() == 64);
    assert!(offset_of!(ConfigureFrameGeneration, hudless_color) == 64);
    assert!(offset_of!(ConfigureFrameGeneration, generation_rect) == 120);
    assert!(offset_of!(ConfigureFrameGeneration, frame_id) == 136);
    assert!(size_of::<ConfigureFrameGeneration>() == 144);
    assert!(offset_of!(PrepareV2, command_list) == 32);
    assert!(offset_of!(PrepareV2, frame_time_delta) == 64);
    assert!(offset_of!(PrepareV2, camera_near) == 72);
    assert!(offset_of!(PrepareV2, depth) == 88);
    assert!(offset_of!(PrepareV2, camera_position) == 184);
    assert!(size_of::<PrepareV2>() == 232);
};

type Context = *mut c_void;
type Create =
    unsafe extern "C" fn(context: *mut Context, desc: *mut Header, memory: *const c_void) -> u32;
type Destroy = unsafe extern "C" fn(context: *mut Context, memory: *const c_void) -> u32;
type Configure = unsafe extern "C" fn(context: *mut Context, desc: *const Header) -> u32;
type Dispatch = unsafe extern "C" fn(context: *mut Context, desc: *const Header) -> u32;

/// The loader's exports.
struct Ffx {
    _loader: Library,
    _framegen: Library,
    create: Create,
    destroy: Destroy,
    configure: Configure,
    dispatch: Dispatch,
}

static FFX: OnceLock<Ffx> = OnceLock::new();
/// Our swap chain is FidelityFX's frame interpolation swap chain.
static SWAP_CHAIN: AtomicBool = AtomicBool::new(false);
/// Frame generation ran on the last present.
static ON: AtomicBool = AtomicBool::new(false);
// Since the last frame-rate line: presents with frame generation on, off,
// and prepare dispatches that failed.
static PRESENTS_ON: AtomicU64 = AtomicU64::new(0);
static PRESENTS_OFF: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
/// Presents with frame generation on given the colour without the interface.
static HUDLESS_GIVEN: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, PartialEq, Eq)]
struct ContextConfig {
    display: [u32; 2],
    max_render: [u32; 2],
    format: DXGI_FORMAT,
    flags: u32,
    hudless: DXGI_FORMAT,
}

impl ContextConfig {
    fn covers(self, requested: Self) -> bool {
        self.display == requested.display
            && self.max_render[0] >= requested.max_render[0]
            && self.max_render[1] >= requested.max_render[1]
            && self.format == requested.format
            && self.flags == requested.flags
            && (requested.hudless == DXGI_FORMAT_UNKNOWN || self.hudless == requested.hudless)
    }
}

struct Live {
    swap_chain_context: Box<Context>,
    // The presenter owns the swap chain. Match its identity before using the context.
    swap_chain: *mut c_void,
    // The callback keeps the address of the frame generation context slot.
    framegen: Option<(Box<Context>, ContextConfig)>,
    frame_id: u64,
    refused: Option<ContextConfig>,
}

// SAFETY: FidelityFX contexts, used under the lock by one thread at a time.
unsafe impl Send for Live {}

static LIVE: Mutex<Option<Live>> = Mutex::new(None);

fn live() -> std::sync::MutexGuard<'static, Option<Live>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// `VICTORY_FSR_FRAMEGEN=1`.
pub fn wanted() -> bool {
    enabled(std::env::var("VICTORY_FSR_FRAMEGEN").ok().as_deref())
}

fn enabled(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// `VICTORY_FIDELITYFX_DIR`, or a `fidelityfx` folder beside our exe.
pub fn dir_from_env() -> PathBuf {
    fidelityfx_dir(std::env::var_os("VICTORY_FIDELITYFX_DIR"), || {
        std::env::current_exe().ok()
    })
}

fn fidelityfx_dir(dir: Option<OsString>, exe: impl FnOnce() -> Option<PathBuf>) -> PathBuf {
    match dir.filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => exe()
            .and_then(|exe| exe.parent().map(|d| d.join("fidelityfx")))
            .unwrap_or_else(|| PathBuf::from("fidelityfx")),
    }
}

/// Loads AMD's DLLs from `dir` (signatures checked first); whether they are in.
pub fn init(dir: &Path) -> bool {
    static INIT: Mutex<()> = Mutex::new(());
    let _guard = INIT.lock().unwrap_or_else(|e| e.into_inner());
    if FFX.get().is_some() {
        return true;
    }
    match load(dir) {
        Ok(ffx) => {
            let _ = FFX.set(ffx);
            tracing::info!(folder = %dir.display(), "FSR 3 frame generation: AMD's FidelityFX DLLs loaded (signatures valid)");
            true
        }
        Err(why) => {
            tracing::warn!(folder = %dir.display(), "FSR 3 frame generation: {why}; it stays off");
            false
        }
    }
}

fn wide(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain([0]).collect()
}

struct Library(HMODULE);

// SAFETY: Win32 module references and exports are process-wide.
unsafe impl Send for Library {}
// SAFETY: Lookups can share the handle. Drop requires exclusive ownership.
unsafe impl Sync for Library {}

impl Library {
    fn load(path: &Path, flags: u32) -> Result<Self, String> {
        let name = wide(path);
        // SAFETY: A NUL-terminated path and supported search flags.
        let module = unsafe { LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), flags) };
        if module.is_null() {
            return Err(format!(
                "{} did not load ({})",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self(module))
    }

    fn symbol(&self, name: &CStr) -> Result<*const c_void, String> {
        // SAFETY: A live module and a NUL-terminated export name.
        unsafe { GetProcAddress(self.0, name.as_ptr().cast()) }
            .map(|address| address as *const c_void)
            .ok_or_else(|| format!("missing export {}", name.to_string_lossy()))
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: This wrapper owns the reference returned by LoadLibraryExW.
        unsafe { FreeLibrary(self.0) };
    }
}

fn verify_signature(path: &Path) -> Result<(), String> {
    #[repr(C)]
    struct WinTrustFileInfo {
        size: u32,
        path: *const u16,
        file: HANDLE,
        known_subject: *const GUID,
    }

    #[repr(C)]
    struct WinTrustData {
        size: u32,
        policy_data: *mut c_void,
        sip_data: *mut c_void,
        ui_choice: u32,
        revocation_checks: u32,
        union_choice: u32,
        file_info: *mut WinTrustFileInfo,
        state_action: u32,
        state_data: HANDLE,
        url_reference: *const u16,
        provider_flags: u32,
        ui_context: u32,
        signature_settings: *mut c_void,
    }

    const _: () = assert!(size_of::<WinTrustFileInfo>() == 32 && size_of::<WinTrustData>() == 88);
    type WinVerifyTrust = unsafe extern "system" fn(HANDLE, *mut GUID, *mut c_void) -> i32;
    let library = Library::load(Path::new("wintrust.dll"), LOAD_LIBRARY_SEARCH_SYSTEM32)?;
    let symbol = library.symbol(c"WinVerifyTrust")?;
    // SAFETY: This is the documented WinVerifyTrust export from System32.
    let verify = unsafe { std::mem::transmute::<*const c_void, WinVerifyTrust>(symbol) };
    let mut policy = GUID {
        data1: 0x00aac56b,
        data2: 0xcd44,
        data3: 0x11d0,
        data4: [0x8c, 0xc2, 0, 0xc0, 0x4f, 0xc2, 0x95, 0xee],
    };
    let name = wide(path);
    let mut file_info = WinTrustFileInfo {
        size: size_of::<WinTrustFileInfo>() as u32,
        path: name.as_ptr(),
        file: std::ptr::null_mut(),
        known_subject: std::ptr::null(),
    };
    let mut trust = WinTrustData {
        size: size_of::<WinTrustData>() as u32,
        policy_data: std::ptr::null_mut(),
        sip_data: std::ptr::null_mut(),
        ui_choice: 2,         // WTD_UI_NONE
        revocation_checks: 0, // WTD_REVOKE_NONE
        union_choice: 1,      // WTD_CHOICE_FILE
        file_info: &raw mut file_info,
        state_action: 1, // WTD_STATEACTION_VERIFY
        state_data: std::ptr::null_mut(),
        url_reference: std::ptr::null(),
        provider_flags: 0x1000, // WTD_CACHE_ONLY_URL_RETRIEVAL
        ui_context: 0,          // WTD_UICONTEXT_EXECUTE
        signature_settings: std::ptr::null_mut(),
    };
    // SAFETY: These Authenticode descriptors and their pointers live through both calls.
    let status = unsafe {
        let status = verify(
            INVALID_HANDLE_VALUE,
            &raw mut policy,
            (&raw mut trust).cast(),
        );
        trust.state_action = 2; // WTD_STATEACTION_CLOSE
        verify(
            INVALID_HANDLE_VALUE,
            &raw mut policy,
            (&raw mut trust).cast(),
        );
        status
    };
    if status != 0 {
        return Err(format!(
            "{} failed Authenticode verification ({:#010x})",
            path.display(),
            status as u32
        ));
    }
    Ok(())
}

fn load(dir: &Path) -> Result<Ffx, String> {
    use std::os::windows::fs::OpenOptionsExt;

    let dir = std::path::absolute(dir).map_err(|e| e.to_string())?;
    let effect_path = dir.join(FRAMEGEN);
    let loader_path = dir.join(LOADER);
    // Keep both files read-only and in place during verification and loading.
    let open = |path: &Path| {
        const FILE_SHARE_READ: u32 = 1;
        std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(path)
            .map_err(|e| format!("{} could not be opened ({e})", path.display()))
    };
    let _effect_file = open(&effect_path)?;
    let _loader_file = open(&loader_path)?;
    verify_signature(&effect_path)?;
    verify_signature(&loader_path)?;

    let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    // The loader resolves the preloaded effect by its module name.
    let framegen = Library::load(&effect_path, search)?;
    let loader = Library::load(&loader_path, search)?;
    macro_rules! get {
        ($name:literal, $t:ty) => {
            // SAFETY: The export has the SDK header's signature.
            unsafe { std::mem::transmute::<*const c_void, $t>(loader.symbol($name)?) }
        };
    }
    Ok(Ffx {
        create: get!(c"ffxCreateContext", Create),
        destroy: get!(c"ffxDestroyContext", Destroy),
        configure: get!(c"ffxConfigure", Configure),
        dispatch: get!(c"ffxDispatch", Dispatch),
        _loader: loader,
        _framegen: framegen,
    })
}

/// Whether the initialized runtime can receive upscaling inputs.
pub fn noting() -> bool {
    FFX.get().is_some() && wanted()
}

/// Frames shown for each rendered frame.
pub(crate) fn shown_per_frame() -> u32 {
    if SWAP_CHAIN.load(Relaxed) && ON.load(Relaxed) {
        2
    } else {
        1
    }
}

/// Creates FidelityFX's interpolation swap chain for the host window.
pub(crate) fn swap_chain(
    factory: &IDXGIFactory2,
    queue: &ID3D12CommandQueue,
    hwnd: HWND,
    desc: &DXGI_SWAP_CHAIN_DESC1,
) -> Option<IDXGISwapChain3> {
    if !wanted() {
        return None;
    }
    let ffx = FFX.get()?;
    let mut active = live();
    if active.is_some() {
        return None;
    }
    let mut made: *mut c_void = std::ptr::null_mut();
    let mut version = Version {
        header: Header::new(CREATE_SWAPCHAIN_VERSION_DX12),
        version: SWAPCHAIN_VERSION,
    };
    let mut create = CreateSwapChainForHwnd {
        header: Header::new(CREATE_SWAPCHAIN_FOR_HWND_DX12),
        swap_chain: &mut made,
        hwnd,
        desc,
        fullscreen: std::ptr::null(),
        factory: factory.as_raw(),
        queue: queue.as_raw(),
    };
    create.header.next = (&raw mut version).cast();
    let mut context: Box<Context> = Box::new(std::ptr::null_mut());
    // SAFETY: filled descriptions that live through the call.
    let code = unsafe { (ffx.create)(&mut *context, (&raw mut create).cast(), std::ptr::null()) };
    if code != OK || made.is_null() || context.is_null() {
        if !context.is_null() {
            // SAFETY: The SDK returned a context to release.
            unsafe { (ffx.destroy)(&mut *context, std::ptr::null()) };
        }
        if !made.is_null() {
            // SAFETY: The SDK returned an owned swap-chain reference.
            drop(unsafe { IDXGISwapChain4::from_raw(made) });
        }
        tracing::warn!(
            code,
            "FSR 3 frame generation: FidelityFX did not make the swap chain; made as before, frame generation off"
        );
        return None;
    }
    // SAFETY: the swap chain FidelityFX made, with the reference it gave us.
    let swap_chain = unsafe { IDXGISwapChain4::from_raw(made) };
    let swap_chain = match swap_chain.cast::<IDXGISwapChain3>() {
        Ok(s) => s,
        Err(e) => {
            // SAFETY: The swap-chain context has no generation callback yet.
            unsafe { (ffx.destroy)(&mut *context, std::ptr::null()) };
            tracing::warn!(error = %e, "FSR 3 frame generation: the swap chain has no IDXGISwapChain3");
            return None;
        }
    };
    *active = Some(Live {
        swap_chain_context: context,
        swap_chain: swap_chain.as_raw(),
        framegen: None,
        frame_id: 0,
        refused: None,
    });
    SWAP_CHAIN.store(true, Release);
    tracing::info!(
        width = desc.Width,
        height = desc.Height,
        format = desc.Format.0,
        "FSR 3 frame generation: our swap chain is FidelityFX's frame interpolation swap chain"
    );
    Some(swap_chain)
}

/// The swap chain's buffers are about to be made again (everything sent has
/// run): the frame generation context goes, and is made again for the new
/// size at the next present with an upscaling.
pub(crate) fn resizing() {
    let Some(ffx) = FFX.get() else { return };
    let mut live = live();
    if let Some(live) = live.as_mut() {
        retire(ffx, live);
    }
}

/// Disables generation and drains the swap chain before releasing the callback target.
fn retire(ffx: &Ffx, live: &mut Live) -> bool {
    ON.store(false, Relaxed);
    live.refused = None;
    let mut wait = Header::new(DISPATCH_WAIT_FOR_PRESENTS_DX12);
    if let Some((context, _)) = live.framegen.as_mut() {
        live.frame_id += 1;
        let off = ConfigureFrameGeneration {
            header: Header::new(CONFIGURE_FRAMEGENERATION),
            swap_chain: live.swap_chain,
            present_callback: std::ptr::null(),
            present_callback_user: std::ptr::null_mut(),
            frame_generation_callback: None,
            frame_generation_callback_user: std::ptr::null_mut(),
            frame_generation_enabled: false,
            allow_async_workloads: false,
            hudless_color: Resource::NONE,
            flags: 0,
            only_present_generated: false,
            generation_rect: Rect::default(),
            frame_id: live.frame_id,
        };
        // SAFETY: The frame generation context is live.
        let code = unsafe { (ffx.configure)(&mut **context, &off.header) };
        if code != OK {
            tracing::warn!(
                code,
                "FSR 3 frame generation: could not disable generation during teardown"
            );
            return false;
        }
    }
    // SAFETY: The swap-chain context remains live throughout resize and recreation.
    let code = unsafe { (ffx.dispatch)(&mut *live.swap_chain_context, &raw mut wait) };
    if code != OK {
        tracing::warn!(
            code,
            "FSR 3 frame generation: could not wait for interpolated presents"
        );
        return false;
    }
    if let Some((context, _)) = live.framegen.as_mut() {
        // SAFETY: Generation is disabled and all calls using the context have drained.
        let code = unsafe { (ffx.destroy)(&mut **context, std::ptr::null()) };
        if code != OK && !context.is_null() {
            tracing::warn!(
                code,
                "FSR 3 frame generation: could not release the generation context"
            );
            return false;
        }
        live.framegen = None;
    }
    true
}

unsafe extern "C" fn generate(params: *mut Header, user: *mut c_void) -> u32 {
    let Some(ffx) = FFX.get() else { return 1 };
    // SAFETY: `user` is the boxed frame generation context the configuration
    // named; `params` the swap chain's dispatch description.
    unsafe { (ffx.dispatch)(user.cast(), params) }
}

/// Records preparation on the present list and configures the upcoming present.
/// A separate overlay plane allows the base plane to serve as HUD-less color.
pub(crate) fn before_present(
    device: &ID3D12Device,
    swap_chain: &IDXGISwapChain3,
    list: &ID3D12GraphicsCommandList,
    size: [u32; 2],
    format: DXGI_FORMAT,
    base: (&ID3D12Resource, u32),
    two_planes: bool,
) {
    if !SWAP_CHAIN.load(Acquire) {
        return;
    }
    let Some(ffx) = FFX.get() else { return };
    let hudless = two_planes.then_some(base);
    // Of the back buffer's size only (FidelityFX compares the two pixel by pixel).
    let hudless = hudless.filter(|(t, _)| {
        // SAFETY: a live texture.
        let desc = unsafe { t.GetDesc() };
        let same = [desc.Width as u32, desc.Height] == size;
        if !same {
            once!(warn, hudless = ?[desc.Width as u32, desc.Height], back_buffer = ?size, "FSR 3 frame generation: the colour without the interface is not the back buffer's size; the interface moves with the scene");
        }
        same
    });
    let hudless_format = hudless.as_ref().map_or(DXGI_FORMAT_UNKNOWN, |(t, _)| {
        // SAFETY: a live texture.
        unsafe { t.GetDesc() }.Format
    });
    let mut guard = live();
    let Some(live) = guard.as_mut() else { return };
    if live.swap_chain != swap_chain.as_raw() {
        return;
    }
    // The queue's submission line orders upscaling and presentation.
    let line = crate::queue::line_on();
    if !line {
        once!(warn, "FSR 3 frame generation requires the submission line");
    }
    let fresh = fresh_inputs().filter(|_| wanted() && line);
    if let Some(inputs) = &fresh {
        let config = ContextConfig {
            display: size,
            max_render: inputs.camera.render,
            format,
            flags: create_flags(&inputs.camera, format),
            hudless: hudless_format,
        };
        if live
            .framegen
            .as_ref()
            .is_some_and(|(_, current)| !current.covers(config))
        {
            crate::host::settle_for_present();
            if !retire(ffx, live) {
                PRESENTS_OFF.fetch_add(1, Relaxed);
                return;
            }
        }
        if live.framegen.is_none() && live.refused != Some(config) {
            match create_context(ffx, device, config) {
                Some(context) => {
                    live.framegen = Some((context, config));
                    live.refused = None;
                }
                None => live.refused = Some(config),
            }
        }
    }
    let Some((context, ..)) = live.framegen.as_mut() else {
        PRESENTS_OFF.fetch_add(1, Relaxed);
        return;
    };
    live.frame_id += 1;
    let frame_id = live.frame_id;
    let debug = debug_flags();
    // The presenter only reuses this list after its prior submission lands.
    // Attach both resources so they remain alive through GPU execution.
    let fresh = fresh.filter(|inputs| keep_inputs(list, inputs));
    let on = fresh.is_some();
    let mut configure = ConfigureFrameGeneration {
        header: Header::new(CONFIGURE_FRAMEGENERATION),
        swap_chain: swap_chain.as_raw(),
        present_callback: std::ptr::null(),
        present_callback_user: std::ptr::null_mut(),
        frame_generation_callback: on.then_some(generate),
        frame_generation_callback_user: if on {
            (&raw mut **context).cast()
        } else {
            std::ptr::null_mut()
        },
        frame_generation_enabled: on,
        allow_async_workloads: false,
        hudless_color: match (&hudless, on) {
            (Some((texture, state)), true) => Resource::texture(texture, *state),
            _ => Resource::NONE,
        },
        flags: debug,
        only_present_generated: false,
        generation_rect: Rect {
            left: 0,
            top: 0,
            width: size[0] as i32,
            height: size[1] as i32,
        },
        frame_id,
    };
    // SAFETY: The context and descriptors remain live through both calls.
    let code = unsafe { (ffx.configure)(&mut **context, &configure.header) };
    if code != OK {
        once!(
            warn,
            code,
            "FSR 3 frame generation: the configuration failed"
        );
    }
    let configured = code == OK && on;
    let mut prepared = false;
    if let Some(inputs) = fresh.as_ref().filter(|_| configured) {
        let camera = inputs.camera;
        let depth = Resource::texture(&inputs.depth.0, inputs.depth.1);
        let motion = Resource::texture(&inputs.motion.0, inputs.motion.1);
        let mut prepare = PrepareV2 {
            header: Header::new(DISPATCH_PREPARE_V2),
            frame_id,
            flags: debug,
            command_list: list.as_raw(),
            render_size: camera.render.into(),
            jitter_offset: Floats {
                x: camera.jitter[0],
                y: camera.jitter[1],
            },
            motion_vector_scale: Floats {
                x: camera.motion_scale[0],
                y: camera.motion_scale[1],
            },
            frame_time_delta: inputs.frame_ms,
            reset: camera.reset,
            // As the title gives them to its upscaler (FidelityFX's terms).
            camera_near: camera.near,
            camera_far: camera.far,
            camera_fov_angle_vertical: camera.fov_y,
            view_space_to_meters_factor: 1.0,
            depth,
            motion_vectors: motion,
            camera_position: [0.0; 3],
            camera_up: [0.0, 1.0, 0.0],
            camera_right: [1.0, 0.0, 0.0],
            camera_forward: [0.0, 0.0, 1.0],
        };
        // SAFETY: our context and a filled description of live textures in
        // the states they are in when `list` runs.
        let code = unsafe { (ffx.dispatch)(&mut **context, (&raw mut prepare).cast()) };
        if code == OK {
            prepared = true;
            once!(
                info,
                render = ?camera.render,
                output = ?camera.output,
                back_buffer = ?size,
                depth_state = format_args!("{:#x}", inputs.depth.1),
                motion_state = format_args!("{:#x}", inputs.motion.1),
                near = camera.near,
                far = camera.far,
                fov = camera.fov_y,
                jitter = ?camera.jitter,
                mvec_scale = ?camera.motion_scale,
                frame_ms = inputs.frame_ms,
                "FSR 3 frame generation: first frame prepared from depth and motion vectors"
            );
        } else {
            FAILED.fetch_add(1, Relaxed);
            once!(
                warn,
                code,
                "FSR 3 frame generation: the prepare dispatch failed"
            );
        }
    }
    if on && !prepared {
        configure.frame_generation_enabled = false;
        configure.frame_generation_callback = None;
        configure.frame_generation_callback_user = std::ptr::null_mut();
        configure.hudless_color = Resource::NONE;
        // SAFETY: The context is live. Disable interpolation for this failed frame.
        let code = unsafe { (ffx.configure)(&mut **context, &configure.header) };
        if code != OK {
            once!(
                warn,
                code,
                "FSR 3 frame generation: could not disable a failed frame"
            );
        }
    }
    let on = prepared;
    if ON.swap(on, Relaxed) != on {
        tracing::info!(on, frame_id, "FSR 3 frame generation switched");
    }
    if on {
        PRESENTS_ON.fetch_add(1, Relaxed);
        if hudless.is_some() {
            HUDLESS_GIVEN.fetch_add(1, Relaxed);
            once!(
                info,
                two_planes,
                "FSR 3 frame generation: first frame with the colour without the interface"
            );
        }
    } else {
        PRESENTS_OFF.fetch_add(1, Relaxed);
    }
}

fn keep_inputs(list: &ID3D12GraphicsCommandList, inputs: &Inputs) -> bool {
    use windows::core::GUID;
    const DEPTH: GUID = GUID::from_u128(0x978091ab_7ea8_4556_9785_178f3308a019);
    const MOTION: GUID = GUID::from_u128(0xd9b9609a_3755_4c4f_a673_193b55f0e2f9);
    // SAFETY: Both live resources implement IUnknown. The command list AddRefs them.
    let result = unsafe {
        list.SetPrivateDataInterface(&DEPTH, &inputs.depth.0)
            .and_then(|()| list.SetPrivateDataInterface(&MOTION, &inputs.motion.0))
    };
    if let Err(error) = result {
        once!(warn, %error, "FSR 3 frame generation: could not retain prepare resources");
        return false;
    }
    true
}

fn create_context(ffx: &Ffx, device: &ID3D12Device, config: ContextConfig) -> Option<Box<Context>> {
    let ContextConfig {
        display: size,
        max_render: render,
        format,
        flags,
        hudless: hudless_format,
    } = config;
    let mut backend = CreateBackend {
        header: Header::new(CREATE_BACKEND_DX12),
        device: device.as_raw(),
    };
    // The colour without the interface in a format of its own (else the back buffer's is assumed).
    let mut hudless = Version {
        header: Header::new(CREATE_FRAMEGENERATION_HUDLESS),
        version: surface_format(hudless_format),
    };
    let first_after_version = if hudless_format != DXGI_FORMAT_UNKNOWN && hudless_format != format {
        hudless.header.next = &raw mut backend.header;
        &raw mut hudless.header
    } else {
        &raw mut backend.header
    };
    let mut version = Version {
        header: Header::new(CREATE_FRAMEGENERATION_VERSION),
        version: FRAMEGENERATION_VERSION,
    };
    let mut create = CreateFrameGeneration {
        header: Header::new(CREATE_FRAMEGENERATION),
        flags,
        display_size: size.into(),
        max_render_size: render.into(),
        back_buffer_format: surface_format(format),
    };
    version.header.next = first_after_version;
    create.header.next = (&raw mut version).cast();
    let mut context: Box<Context> = Box::new(std::ptr::null_mut());
    // SAFETY: filled descriptions that live through the call.
    let code = unsafe { (ffx.create)(&mut *context, (&raw mut create).cast(), std::ptr::null()) };
    if code != OK || context.is_null() {
        tracing::warn!(code, display = ?size, max_render = ?render, format = format.0, "FSR 3 frame generation: its context was not made; presents stay plain");
        return None;
    }
    Some(context)
}

/// The frame generation context's flags from the title's upscaler's (the
/// same meanings, other bits) and our back buffer.
fn create_flags(camera: &Camera, format: DXGI_FORMAT) -> u32 {
    let mut flags = 0;
    if camera.display_res_mv {
        flags |= FG_DISPLAY_RESOLUTION_MV;
    }
    if camera.mv_jitter_cancelled {
        flags |= FG_MV_JITTER_CANCELLATION;
    }
    if camera.inverted_depth {
        flags |= FG_DEPTH_INVERTED;
    }
    if camera.infinite_depth {
        flags |= FG_DEPTH_INFINITE;
    }
    // (The title's upscaler takes the scene before its tone mapping; what we
    // present is after it: high dynamic range only for a float back buffer.)
    if format == DXGI_FORMAT_R16G16B16A16_FLOAT {
        flags |= FG_HIGH_DYNAMIC_RANGE;
    }
    flags
}

/// `VICTORY_FSR_FRAMEGEN_DEBUG=1`: tear and pacing lines drawn.
fn debug_flags() -> u32 {
    static FLAGS: OnceLock<u32> = OnceLock::new();
    *FLAGS.get_or_init(|| {
        if std::env::var("VICTORY_FSR_FRAMEGEN_DEBUG").is_ok_and(|v| v.trim() == "1") {
            FG_DRAW_DEBUG_TEAR_LINES | FG_DRAW_DEBUG_PACING_LINES
        } else {
            0
        }
    })
}

/// A D3D12 state as FidelityFX names it.
fn ffx_state(state: u32) -> u32 {
    let s = D3D12_RESOURCE_STATES(state as i32);
    if s == D3D12_RESOURCE_STATE_COMMON {
        return STATE_COMMON;
    }
    let mut out = 0;
    if s.contains(D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        out |= STATE_UNORDERED_ACCESS;
    }
    if s.contains(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) {
        out |= STATE_COMPUTE_READ;
    }
    if s.contains(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) {
        out |= STATE_PIXEL_READ;
    }
    if s.contains(D3D12_RESOURCE_STATE_COPY_SOURCE) {
        out |= STATE_COPY_SRC;
    }
    if s.contains(D3D12_RESOURCE_STATE_COPY_DEST) {
        out |= STATE_COPY_DEST;
    }
    if s.contains(D3D12_RESOURCE_STATE_RENDER_TARGET) {
        out |= STATE_RENDER_TARGET;
    }
    if s.contains(D3D12_RESOURCE_STATE_DEPTH_WRITE) || s.contains(D3D12_RESOURCE_STATE_DEPTH_READ) {
        out |= STATE_DEPTH_ATTACHMENT;
    }
    if out == 0 { STATE_COMMON } else { out }
}

/// `ffxApiGetSurfaceFormatDX12`.
fn surface_format(format: DXGI_FORMAT) -> u32 {
    match format {
        DXGI_FORMAT_R32G32B32A32_TYPELESS => 1,
        DXGI_FORMAT_R32G32B32A32_UINT => 2,
        DXGI_FORMAT_R32G32B32A32_FLOAT => 3,
        DXGI_FORMAT_R16G16B16A16_FLOAT => 4,
        DXGI_FORMAT_R32G32_FLOAT => 6,
        DXGI_FORMAT_R8_UINT
        | DXGI_FORMAT_X32_TYPELESS_G8X24_UINT
        | DXGI_FORMAT_X24_TYPELESS_G8_UINT => 7,
        DXGI_FORMAT_R32_UINT
        | DXGI_FORMAT_R24G8_TYPELESS
        | DXGI_FORMAT_D24_UNORM_S8_UINT
        | DXGI_FORMAT_R24_UNORM_X8_TYPELESS => 8,
        DXGI_FORMAT_R8G8B8A8_TYPELESS => 9,
        DXGI_FORMAT_R8G8B8A8_UNORM => 10,
        DXGI_FORMAT_R8G8B8A8_SNORM => 11,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => 12,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => 13,
        DXGI_FORMAT_B8G8R8A8_UNORM => 14,
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => 15,
        DXGI_FORMAT_R11G11B10_FLOAT => 16,
        DXGI_FORMAT_R10G10B10A2_UNORM => 17,
        DXGI_FORMAT_R16G16_FLOAT => 18,
        DXGI_FORMAT_R16G16_UINT => 19,
        DXGI_FORMAT_R16_FLOAT => 21,
        DXGI_FORMAT_R16_UINT => 22,
        DXGI_FORMAT_R16_UNORM | DXGI_FORMAT_D16_UNORM => 23,
        DXGI_FORMAT_R16_SNORM => 24,
        DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_A8_UNORM => 25,
        DXGI_FORMAT_R8G8_UNORM => 26,
        DXGI_FORMAT_R8G8_UINT => 27,
        DXGI_FORMAT_R32_FLOAT
        | DXGI_FORMAT_D32_FLOAT
        | DXGI_FORMAT_R32G8X24_TYPELESS
        | DXGI_FORMAT_D32_FLOAT_S8X24_UINT
        | DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS => 28,
        DXGI_FORMAT_R9G9B9E5_SHAREDEXP => 29,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => 30,
        DXGI_FORMAT_R32G32_TYPELESS => 31,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => 32,
        DXGI_FORMAT_R16G16_TYPELESS => 33,
        DXGI_FORMAT_R16_TYPELESS => 34,
        DXGI_FORMAT_R8_TYPELESS => 35,
        DXGI_FORMAT_R8G8_TYPELESS => 36,
        DXGI_FORMAT_R32_TYPELESS => 37,
        DXGI_FORMAT_R8_SNORM => 39,
        _ => 0,
    }
}

/// Logs frame-generation counts periodically alongside the frame-rate stats.
pub(crate) fn log_line(frames: f64) {
    static LINES: AtomicU64 = AtomicU64::new(0);
    if SWAP_CHAIN.load(Relaxed) && LINES.fetch_add(1, Relaxed) % 10 == 9 {
        let on = PRESENTS_ON.swap(0, Relaxed);
        let off = PRESENTS_OFF.swap(0, Relaxed);
        let failed = FAILED.swap(0, Relaxed);
        let hudless = HUDLESS_GIVEN.swap(0, Relaxed);
        tracing::info!(
            frames,
            on,
            off,
            failed,
            hudless,
            "FSR 3 frame generation presents"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_upscale_is_consumed_once_and_expires() {
        let mut latest = Latest::new();
        let now = Instant::now();
        assert_eq!(latest.take(now), None);
        // Pipelined recording (N then N+1 before N presents) preserves FIFO order.
        latest.note(now, 1);
        latest.note(now, 2);
        assert_eq!(latest.take(now + Duration::from_millis(1999)), Some(1));
        assert_eq!(latest.take(now + Duration::from_millis(1999)), Some(2));
        assert_eq!(latest.take(now), None);
        // Bounded to MAX_QUEUED (oldest dropped when exceeded).
        for v in 10..16 {
            latest.note(now, v);
        }
        assert_eq!(latest.take(now), Some(12));
        assert_eq!(latest.take(now), Some(13));
        assert_eq!(latest.take(now), Some(14));
        assert_eq!(latest.take(now), Some(15));
        assert_eq!(latest.take(now), None);
        // Expired entries are skipped cleanly.
        latest.note(now, 3);
        assert_eq!(latest.take(now + Duration::from_millis(2000)), None);
        assert_eq!(latest.take(now), None);
    }

    #[test]
    fn defaults_to_one_displayed_frame() {
        assert_eq!(shown_per_frame(), 1);
    }

    #[test]
    fn environment_options_are_opt_in_and_resolve_the_sdk_directory() {
        assert!(enabled(Some(" 1 ")));
        assert!(!enabled(Some("true")));
        assert!(!enabled(None));
        let exe = || Some(PathBuf::from("bin").join("victory.exe"));
        assert_eq!(
            fidelityfx_dir(Some("sdk".into()), exe),
            PathBuf::from("sdk")
        );
        assert_eq!(
            fidelityfx_dir(Some("".into()), exe),
            Path::new("bin").join("fidelityfx")
        );
        assert_eq!(fidelityfx_dir(None, || None), PathBuf::from("fidelityfx"));
    }

    #[test]
    fn context_flags_follow_temporal_inputs_and_swap_chain_format() {
        let camera = Camera {
            render: [1280, 720],
            output: [1920, 1080],
            jitter: [0.0; 2],
            motion_scale: [1.0; 2],
            near: 0.1,
            far: 1000.0,
            fov_y: 1.0,
            reset: false,
            inverted_depth: true,
            infinite_depth: false,
            display_res_mv: true,
            mv_jitter_cancelled: false,
            hdr: true,
        };
        assert_eq!(
            create_flags(&camera, DXGI_FORMAT_R8G8B8A8_UNORM),
            FG_DEPTH_INVERTED | FG_DISPLAY_RESOLUTION_MV
        );
        let camera = Camera {
            inverted_depth: false,
            infinite_depth: true,
            display_res_mv: false,
            mv_jitter_cancelled: true,
            hdr: false,
            ..camera
        };
        assert_eq!(
            create_flags(&camera, DXGI_FORMAT_R16G16B16A16_FLOAT),
            FG_DEPTH_INFINITE | FG_MV_JITTER_CANCELLATION | FG_HIGH_DYNAMIC_RANGE
        );
    }

    #[test]
    fn states_and_formats() {
        assert_eq!(ffx_state(0), STATE_COMMON);
        assert_eq!(ffx_state(0x8), STATE_UNORDERED_ACCESS);
        assert_eq!(ffx_state(0xC0), STATE_COMPUTE_READ | STATE_PIXEL_READ);
        assert_eq!(ffx_state(0x20), STATE_DEPTH_ATTACHMENT);
        assert_eq!(surface_format(DXGI_FORMAT_R8G8B8A8_UNORM), 10);
        assert_eq!(surface_format(DXGI_FORMAT_R10G10B10A2_UNORM), 17);
        assert_eq!(surface_format(DXGI_FORMAT_R32_TYPELESS), 37);
    }

    #[test]
    #[ignore = "needs VICTORY_FIDELITYFX_DIR and a host D3D12 GPU"]
    fn host_gpu_creates_and_destroys_frame_generation_context() {
        let dir = std::path::PathBuf::from(
            std::env::var_os("VICTORY_FIDELITYFX_DIR").expect("VICTORY_FIDELITYFX_DIR not set"),
        );
        assert!(init(&dir), "fsr_fg::init failed for {}", dir.display());
        let ffx = FFX.get().expect("FFX initialized");
        let device = crate::external::host_device().expect("host D3D12 GPU");
        let config = ContextConfig {
            display: [1920, 1080],
            max_render: [1280, 720],
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            flags: FG_DEPTH_INVERTED | FG_DEPTH_INFINITE,
            hudless: DXGI_FORMAT_UNKNOWN,
        };
        let mut ctx =
            create_context(ffx, &device, config).expect("create_context failed on host GPU");
        // SAFETY: Valid newly created context with no pending GPU work.
        let code = unsafe { (ffx.destroy)(&mut *ctx, std::ptr::null()) };
        assert_eq!(code, OK, "ffxDestroyContext returned {code}");
        eprintln!("LIVE AMD FSR 3 FG CONTEXT CREATE+DESTROY OK ON HOST GPU!");
    }
}
