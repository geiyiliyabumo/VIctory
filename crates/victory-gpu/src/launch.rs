//! Amplification shader for direct dispatches and single indirect mesh records.
//! The shader reads indirect arguments in place and consumes the list's mesh
//! budget. Calls with multiple records or a count buffer use the plain mesh
//! pipeline and the indirect clamp.

use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::validator::Validator;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::GUID;
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

use crate::host::HostList;

const CLSID_DXC_COMPILER: GUID = GUID::from_u128(0x73e22d93_e6ce_47f3_b5bf_f0664f39c1b0);
const IID_IDXC_COMPILER3: GUID = GUID::from_u128(0x228b4687_5a6a_4730_900c_9702b2203f54);
const IID_IDXC_RESULT: GUID = GUID::from_u128(0x58346cda_dde7_4497_9461_6f87af5e0659);
const IID_IDXC_BLOB: GUID = GUID::from_u128(0x8ba5fb08_5195_40e2_ac58_0d989c3a0102);
const DXC_OUT_OBJECT: u32 = 1;

#[repr(C)]
struct DxcBuffer {
    ptr: *const c_void,
    size: usize,
    encoding: u32,
}

struct Module(HMODULE);

impl Module {
    fn load(path: &Path) -> Option<Self> {
        let name: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        // SAFETY: a NUL-terminated path.
        let handle = unsafe { LoadLibraryW(name.as_ptr()) };
        if handle.is_null() {
            None
        } else {
            Some(Self(handle))
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: one reference from LoadLibraryW, held until all DXC objects drop.
        unsafe { FreeLibrary(self.0) };
    }
}

type Object = *mut *const usize;
type Hresult = i32;
type CreateInstance = unsafe extern "system" fn(*const GUID, *const GUID, *mut Object) -> Hresult;
type Compile = unsafe extern "system" fn(
    Object,
    *const DxcBuffer,
    *const *const u16,
    u32,
    Object,
    *const GUID,
    *mut Object,
) -> Hresult;

#[derive(Default)]
struct Com(Object);

impl Com {
    unsafe fn method<F: Copy>(&self, slot: usize) -> F {
        // SAFETY: the caller supplies the interface's slot and exact signature.
        unsafe { std::mem::transmute_copy(&*(*self.0).add(slot)) }
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: slot 2 of every COM interface is Release.
            unsafe { self.method::<unsafe extern "system" fn(Object) -> u32>(2)(self.0) };
        }
    }
}

fn signed(bytes: &[u8]) -> bool {
    bytes.starts_with(b"DXBC")
        && bytes
            .get(4..20)
            .is_some_and(|hash| hash.iter().any(|&b| b != 0))
}

pub(crate) fn compile_hlsl(source: &str, profile: &str, entry: &str) -> Option<Vec<u8>> {
    let explicit = std::env::var_os("VICTORY_DXIL").map(PathBuf::from);
    let default = Validator::default_path();
    let sibling = |path: &PathBuf| path.with_file_name("dxcompiler.dll");
    let candidates = [
        explicit.as_ref().map(sibling),
        default
            .as_ref()
            .filter(|p| Some(*p) != explicit.as_ref())
            .map(sibling),
        Some(PathBuf::from(
            r"C:\Windows\System32\Microsoft-Edge-WebView\dxcompiler.dll",
        )),
        Some(PathBuf::from("dxcompiler.dll")),
    ];
    let buffer = DxcBuffer {
        ptr: source.as_ptr().cast(),
        size: source.len(),
        encoding: 65001,
    };
    let options =
        ["-T", profile, "-E", entry].map(|arg| arg.encode_utf16().chain([0]).collect::<Vec<_>>());
    let args = options.each_ref().map(|arg| arg.as_ptr());
    for path in candidates.into_iter().flatten() {
        let Some(module) = Module::load(&path) else {
            continue;
        };
        let Some(mut bytes) = compile(&module, &buffer, &args) else {
            continue;
        };
        if !signed(&bytes) && bytes.starts_with(b"DXBC") {
            let sibling = path.with_file_name("dxil.dll");
            for path in [Some(sibling.as_path()), default.as_deref(), None] {
                if let Ok(mut validator) = Validator::load(path)
                    && validator.validate(&mut bytes).is_ok()
                    && signed(&bytes)
                {
                    break;
                }
            }
        }
        if signed(&bytes) {
            return Some(bytes);
        }
    }
    None
}

fn compile(module: &Module, buffer: &DxcBuffer, args: &[*const u16]) -> Option<Vec<u8>> {
    // SAFETY: DXC's interfaces use the Windows COM ABI. Each out pointer is
    // owned by a guard. The options and source stay live through Compile.
    unsafe {
        let create = GetProcAddress(module.0, c"DxcCreateInstance".as_ptr().cast())?;
        let create: CreateInstance = std::mem::transmute(create);
        let mut compiler = Com::default();
        if create(&CLSID_DXC_COMPILER, &IID_IDXC_COMPILER3, &mut compiler.0) < 0
            || compiler.0.is_null()
        {
            return None;
        }
        let mut result = Com::default();
        let compile: Compile = compiler.method(3);
        if compile(
            compiler.0,
            buffer,
            args.as_ptr(),
            args.len() as u32,
            std::ptr::null_mut(),
            &IID_IDXC_RESULT,
            &mut result.0,
        ) < 0
            || result.0.is_null()
        {
            return None;
        }

        // IDxcResult inherits IDxcOperationResult (slots 3 through 5);
        // HasOutput is slot 6 and GetOutput is slot 7.
        let status: unsafe extern "system" fn(Object, *mut Hresult) -> Hresult = result.method(3);
        let mut hr = 0;
        if status(result.0, &mut hr) < 0 || hr < 0 {
            return None;
        }
        let output: unsafe extern "system" fn(
            Object,
            u32,
            *const GUID,
            *mut Object,
            *mut Object,
        ) -> Hresult = result.method(7);
        let mut blob = Com::default();
        if output(
            result.0,
            DXC_OUT_OBJECT,
            &IID_IDXC_BLOB,
            &mut blob.0,
            std::ptr::null_mut(),
        ) < 0
            || blob.0.is_null()
        {
            return None;
        }
        let ptr: unsafe extern "system" fn(Object) -> *const u8 = blob.method(3);
        let size: unsafe extern "system" fn(Object) -> usize = blob.method(4);
        let (ptr, size) = (ptr(blob.0), size(blob.0));
        (!ptr.is_null() && size != 0).then(|| std::slice::from_raw_parts(ptr, size).to_vec())
    }
}

/// The root constant that has the card read the record.
pub(crate) const DIRECT_NONE: u32 = u32::MAX;

/// Where the amplification shader binds: `b0`, `t0`, `u0` in this space.
const SPACE: u32 = victory_shader::emit::LAUNCH_SPACE;

/// Whether mesh pipelines get the amplification stage and indirect mesh
/// records are launched by it (`VICTORY_MESH_LAUNCH=1`). The translated
/// mesh shaders declare the payload the stage needs under the same switch
/// (`victory_shader::emit::mesh_payload_bytes`).
pub(crate) fn on() -> bool {
    mode() != 0 && !bytes().is_empty()
}

/// `VICTORY_MESH_LAUNCH`: 0 off, 1 on (the default; run 361: card -5 ms), 2 in turn (`alt`, [`now`]).
fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_MESH_LAUNCH").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        Ok("0") => 0,
        _ => 1,
    })
}

/// Whether a mesh draw is launched by the stage now: with `=1` always; with
/// `alt` twenty seconds with, twenty without, in turn, each change said in
/// the log, so that one run measures the card both ways at equal draws (in
/// the phases without, the same pipelines run without their stage
/// (`Launch::plain`) behind the clamp, as with the switch off; the
/// translated mesh shaders declare the payload in both).
pub(crate) fn now() -> bool {
    match mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH
                .get_or_init(std::time::Instant::now)
                .elapsed()
                .as_secs()
                / 20)
                % 2
                == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "mesh draws from now (VICTORY_MESH_LAUNCH=alt): {}",
                    if on {
                        "ON, launched by the amplification stage"
                    } else {
                        "off, the clamp before each record"
                    }
                );
            }
            on
        }
    }
}

/// The amplification stage's code.
pub(crate) fn code() -> D3D12_SHADER_BYTECODE {
    let bytes = bytes();
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}

/// The bytes of [`code`], for the pipeline library's names.
pub(crate) fn bytes() -> &'static [u8] {
    static CODE: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    CODE.get_or_init(|| {
        let code = compile_hlsl(include_str!("launch.hlsl"), "as_6_6", "main");
        if code.is_none() {
            tracing::warn!("could not compile a signed mesh launch shader with dxcompiler.dll");
        }
        code
    })
    .as_deref()
    .unwrap_or_default()
}

/// The three root parameters the stage reads, in order: the direct group
/// count (one constant), the record (a root SRV), the list's budget (a
/// root UAV). Five root words.
pub(crate) fn params() -> [D3D12_ROOT_PARAMETER1; 3] {
    let descriptor = |kind| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: 0,
                RegisterSpace: SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
    };
    [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: SPACE,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        },
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV),
    ]
}

/// Mesh draws launched since the last line: records read by the card, and
/// dispatches whose count the CPU gave.
static LAUNCHED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Records the launch of one mesh draw, the mesh pipeline with its stage
/// bound and the stage's parameters at `roots`: `record` the record's
/// address on the card (the card reads it), or the group count the CPU
/// knows. False if the list has no budget for a record.
pub(crate) fn record(list: &HostList, roots: [u32; 3], what: Launch) -> bool {
    use windows::core::Interface;
    let l6 = match list.list.cast::<ID3D12GraphicsCommandList6>() {
        Ok(l6) => l6,
        Err(_) => return false,
    };
    let budget = list.budget();
    let (direct, record, budget) = match what {
        Launch::Record(address) => {
            let Some(budget) = budget else {
                return false;
            };
            (DIRECT_NONE, address, budget)
        }
        // The stage reads neither the record nor the budget; live
        // addresses all the same where the list has them.
        Launch::Direct(x) => (x.min(65_535), budget.unwrap_or(0), budget.unwrap_or(0)),
    };
    // SAFETY: an open list with the mesh pipeline and its root signature
    // set; the record and the budget are live buffers the list keeps.
    unsafe {
        list.list.SetGraphicsRoot32BitConstant(roots[0], direct, 0);
        list.list
            .SetGraphicsRootShaderResourceView(roots[1], record);
        list.list
            .SetGraphicsRootUnorderedAccessView(roots[2], budget);
        l6.DispatchMesh(1, 1, 1);
    }
    LAUNCHED[usize::from(matches!(what, Launch::Direct(_)))].fetch_add(1, Ordering::Relaxed);
    each_second();
    true
}

/// What one launch reads.
#[derive(Clone, Copy)]
pub(crate) enum Launch {
    /// The record at this address on the card.
    Record(u64),
    /// A group count known on the CPU.
    Direct(u32),
}

/// Says [`line`] once a second, from the launches themselves (no other
/// line of the log is touched), per frame begun meanwhile.
fn each_second() {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static SECOND: AtomicU64 = AtomicU64::new(0);
    static FRAME: AtomicU64 = AtomicU64::new(0);
    let now = START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs();
    let last = SECOND.load(Ordering::Relaxed);
    if now > last
        && SECOND
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        let frame = crate::frame::current();
        let frames = frame
            .saturating_sub(FRAME.swap(frame, Ordering::Relaxed))
            .max(1);
        line(frames as f64);
    }
}

/// The counts since the last line, per frame.
fn line(frames: f64) {
    let [records, direct] = [0, 1].map(|i| LAUNCHED[i].swap(0, Ordering::Relaxed));
    if records + direct > 0 {
        tracing::info!(
            "mesh draws launched by the card per frame: {:.0} records read where the title wrote them, {:.0} counts \
             known on the CPU (no pass of ours between them; without VICTORY_MESH_LAUNCH=1 each record is clamped by one)",
            records as f64 / frames,
            direct as f64 / frames
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn launch_hlsl_compiles_to_signed_as_6_6() {
        let bytes = super::bytes();
        assert!(
            super::signed(bytes),
            "expected signed amplification shader bytecode"
        );
        assert_eq!(&bytes[..4], b"DXBC");
    }
}
