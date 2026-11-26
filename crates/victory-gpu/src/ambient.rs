//! Depth-shaded indirect light and a reflection clamp for the non-RT path.
//! The title normally gets this light from traced screen probes. With their
//! inputs unwritten, replace the resolve's output before lighting samples it.
//! `VICTORY_AMBIENT=off` disables both passes. A finite nonnegative level
//! sets the constant level, or opts in when the RT builder is enabled.
//! With RT disabled, the light follows the sky dome when available.
//! `VICTORY_AMBIENT_SKY=off` uses the constant level throughout.

use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::descriptor::{View, host::placed_texture};
use crate::host::{HostList, host};
use crate::object::This;
use crate::resource::{HostTexture, host_copy, host_resource};

pub(crate) const PROGRAM: &[u8] = b"CS_ScreenProbes_TemporalResolve";
pub(crate) const REFLECTED: &[u8] = b"CS_UpdateHistory";
pub(crate) const SKY: &[u8] = b"ps_sky_paraboloid_reflection";

// Root parameter and dword in the descriptor table.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);
pub(crate) const REFLECTIONS: (u8, u32) = (1, 40);

pub(crate) fn level() -> Option<f32> {
    static LEVEL: OnceLock<Option<f32>> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let value = std::env::var("VICTORY_AMBIENT").unwrap_or_default();
        select_level(value.trim(), crate::pipeline::builder_allowed())
    })
}

fn select_level(value: &str, rt: bool) -> Option<f32> {
    if value == "off" {
        return None;
    }
    value
        .parse()
        .ok()
        .filter(|v: &f32| v.is_finite() && *v >= 0.0)
        .or((!rt).then_some(0.6))
}

#[derive(Clone, Copy)]
struct Settings {
    level: f32,
    tint: [f32; 3],
    shade: [f32; 4],
    sky: [f32; 2],
    held: f32,
    floor: f32,
}

fn values<const N: usize>(name: &str, mut defaults: [f32; N]) -> [f32; N] {
    if let Ok(value) = std::env::var(name) {
        for (to, field) in defaults.iter_mut().zip(value.split(',')) {
            if let Some(value) = field.trim().parse().ok().filter(|v: &f32| v.is_finite()) {
                *to = value;
            }
        }
    }
    defaults
}

fn settings(level: f32) -> Settings {
    static SKY_SETTING: OnceLock<Option<String>> = OnceLock::new();
    let sky_setting = SKY_SETTING.get_or_init(|| std::env::var("VICTORY_AMBIENT_SKY").ok());
    let sky_off = sky_setting.as_deref().is_some_and(|v| v.trim() == "off");
    static BASE: OnceLock<Settings> = OnceLock::new();
    let base = *BASE.get_or_init(|| Settings {
        level,
        tint: values("VICTORY_AMBIENT_TINT", [1.0, 0.98, 0.94]).map(|v| v.max(0.0)),
        shade: values("VICTORY_AMBIENT_SHADE", [1.0, 0.6, 0.1, 0.45]),
        sky: if !sky_off && (sky_setting.is_some() || !crate::pipeline::builder_allowed()) {
            values("VICTORY_AMBIENT_SKY", [0.5, 0.05])
        } else {
            [0.0; 2]
        },
        held: values("VICTORY_AMBIENT_REFLECT", [0.5])[0].max(0.0),
        floor: 0.0,
    });
    #[cfg(feature = "diagnostics")]
    {
        let mut live = tuned(base);
        if sky_off {
            live.sky = [0.0; 2];
        }
        live
    }
    #[cfg(not(feature = "diagnostics"))]
    base
}

/// Polls the reference port's tuning file twice a second in diagnostics builds.
/// Layouts are gain/least or level/R/G/B, optionally followed by strength,
/// radius, near plane and tan(half vertical FOV). The second layout also
/// accepts the reflection ceiling and floor as fractions of ambient light.
#[cfg(feature = "diagnostics")]
fn tuned(base: Settings) -> Settings {
    static LIVE: Mutex<Option<Settings>> = Mutex::new(None);
    static START: std::sync::Once = std::sync::Once::new();
    START.call_once(|| {
        let Some(file) = std::env::var_os("VICTORY_AMBIENT_FILE") else {
            return;
        };
        let _ = std::thread::Builder::new()
            .name("victory-ambient".into())
            .spawn(move || {
                loop {
                    if let Ok(text) = std::fs::read_to_string(&file)
                        && let Ok(fields) = text
                            .split_whitespace()
                            .map(str::parse::<f32>)
                            .collect::<Result<Vec<_>, _>>()
                        && fields.iter().all(|v| v.is_finite())
                    {
                        let mut next = base;
                        let shade = match fields.as_slice() {
                            [gain, least, rest @ ..] if matches!(rest.len(), 0 | 4) => {
                                next.sky = [*gain, *least];
                                Some(rest)
                            }
                            [level, r, g, b, rest @ ..] if matches!(rest.len(), 0 | 4..=6) => {
                                next.level = level.max(0.0);
                                next.tint = [*r, *g, *b].map(|v| v.max(0.0));
                                next.sky = [0.0; 2];
                                if let Some(held) = rest.get(4) {
                                    next.held = held.max(0.0);
                                    next.floor = rest
                                        .get(5)
                                        .copied()
                                        .unwrap_or_default()
                                        .clamp(0.0, next.held);
                                }
                                Some(rest)
                            }
                            _ => None,
                        };
                        if let Some(shade) = shade {
                            if let Some(values) = shade.get(..4) {
                                next.shade.copy_from_slice(values);
                            }
                            *LIVE.lock().unwrap_or_else(|e| e.into_inner()) = Some(next);
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
            });
    });
    LIVE.lock()
        .unwrap_or_else(|e| e.into_inner())
        .unwrap_or(base)
}

/// The sky dome's host copy can outlive the resource that supplied it.
static SKY_IMAGE: Mutex<Option<Arc<HostTexture>>> = Mutex::new(None);

pub(crate) fn sky_drawn(image: This) {
    let Some(level) = level() else { return };
    if settings(level).sky[0] <= 0.0 {
        return;
    }
    let Some(copy) = host_copy(image) else {
        return;
    };
    // SAFETY: the target is live on the recording list.
    let desc = unsafe { copy.texture.GetDesc() };
    // The same shader also draws into a packed float target. Following both
    // targets would make ambient depend on which list was recorded last.
    if texture_2d(&desc)
        && desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT
        && !desc
            .Flags
            .contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    {
        let mut sky = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner());
        if sky.as_ref().is_none_or(|held| !Arc::ptr_eq(held, &copy)) {
            let old = sky.replace(copy);
            drop(sky);
            drop(old);
        }
    }
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=12, b0), DescriptorTable(UAV(u0), SRV(t0), SRV(t1))"
cbuffer Floor : register(b0) {
    float level; float3 tint;
    float strength; float radius; float near; float tan_y;
    float gain; float least;
    float held; float floor_part;
};
Texture2D<float3> sky : register(t1);
// The light, from the sky's mean when it follows the sky (gain above 0).
float3 from_around() {
    if (gain <= 0.0) return tint * level;
    uint sw, sh;
    sky.GetDimensions(sw, sh);
    float3 sum = 0.0;
    const int N = 6;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++)
            // The sun's own texels are held down: it lights by itself.
            sum += min(sky.Load(int3((x * 2 + 1) * sw / (N * 2), (y * 2 + 1) * sh / (N * 2), 0)), 8.0);
    return sum / (N * N) * gain + float3(0.6, 0.75, 1.0) * least;
}
RWTexture2D<uint> light : register(u0);
Texture2D<float> depth : register(t0);
// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = near / max(d, 1e-7);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
// How much of the light from around reaches the texel: 1 in the open.
float open_to_light(int2 p, float2 size) {
    float3 c = seen(p, size);
    if (c.z > 400.0) return 1.0;
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, c) > 0.0) n = -n;
    float px = radius * size.y / (2.0 * tan_y * c.z);
    if (px < 1.5) return 1.0;
    px = min(px, 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    float sum = 0.0;
    const int N = 16;
    for (int i = 0; i < N; i++) {
        float a = i * 2.399963 + noise * 6.2831853;
        float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
        float3 v = seen(p + int2(round(o)), size) - c;
        float vv = dot(v, v);
        float over = dot(v, n) - 0.004 * c.z - 0.005;
        sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
    }
    return saturate(1.0 - strength * 2.0 * sum / N);
}
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(most, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (most * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    light.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    // Replace the resolve output because its traced inputs are unavailable.
    if (held > 0.0) {
        // Reflections: no brighter than a part of the light around.
        const float3 LUMA = float3(0.2126, 0.7152, 0.0722);
        float3 was = unpack(light[at.xy]);
        // Magenta (green far below red and blue) is what the probes' memory held, not light.
        if (was.g < 0.5 * min(was.r, was.b) && max(was.r, was.b) > 0.02) was = 0.0;
        float3 around = from_around();
        float most = dot(around, LUMA) * held;
        float has = dot(was, LUMA);
        float3 floor_light = around * floor_part;
        if (has > most) light[at.xy] = pack(was * (most / has));
        else if (has == 0.0 || has < dot(floor_light, LUMA)) light[at.xy] = pack(floor_light);
        return;
    }
    float open = strength > 0.0 ? open_to_light(int2(at.xy), float2(w, h)) : 1.0;
    light[at.xy] = pack(from_around() * open);
}
"#;

const VIEWS_PER_PASS: u32 = 3;
const PASSES_PER_HEAP: u32 = 16;
const HEAP_TYPE: D3D12_DESCRIPTOR_HEAP_TYPE = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

/// Each recorded pass gets a fresh table. A list can reuse its tables after
/// reset, when neither queued work nor a replay can still read them.
#[derive(Default)]
pub(crate) struct Views {
    heaps: Vec<ID3D12DescriptorHeap>,
    used: u32,
}

impl Views {
    pub(crate) fn reset(&mut self) {
        self.used = 0;
    }

    fn next(
        &mut self,
        device: &ID3D12Device,
    ) -> windows::core::Result<(&ID3D12DescriptorHeap, u32)> {
        let page = (self.used / PASSES_PER_HEAP) as usize;
        if page == self.heaps.len() {
            // SAFETY: a plain shader-visible heap on this list's device.
            let heap = unsafe {
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: HEAP_TYPE,
                    NumDescriptors: PASSES_PER_HEAP * VIEWS_PER_PASS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?
            };
            self.heaps.push(heap);
        }
        let slot = self.used % PASSES_PER_HEAP * VIEWS_PER_PASS;
        self.used += 1;
        Ok((&self.heaps[page], slot))
    }
}

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    step: u32,
}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let code = crate::host::compile(SOURCE, s!("main"), s!("cs_5_1"))?;
    // SAFETY: the compiled shader includes its root signature. The bytecode
    // and root signature live through pipeline creation.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        };
        let pso = device.CreateComputePipelineState(&desc);
        drop(ManuallyDrop::into_inner(desc.pRootSignature));
        Ok(Pass {
            root,
            pso: pso?,
            step: device.GetDescriptorHandleIncrementSize(HEAP_TYPE),
        })
    }
}

fn texture_2d(desc: &D3D12_RESOURCE_DESC) -> bool {
    desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.DepthOrArraySize == 1
        && desc.SampleDesc.Count == 1
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    // SAFETY: an open list; the barrier orders unordered accesses without
    // naming a resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Replaces the resolved light. Returns true if it changed the list's bindings.
pub(crate) fn raise(list: &HostList, light: &View, depth: Option<&View>) -> bool {
    record(list, light, depth, false)
}

/// Clamps the finished reflections. Returns true if it changed the bindings.
pub(crate) fn hold(list: &HostList, image: &View) -> bool {
    record(list, image, None, true)
}

fn record(list: &HostList, output: &View, depth: Option<&View>, reflection: bool) -> bool {
    let (Some(level), Some(host)) = (level(), host()) else {
        return false;
    };
    let settings = settings(level);
    if reflection && settings.held == 0.0 {
        return false;
    }
    let placed_output = placed_texture(output);
    let View::Unordered(output, 0, view) = placed_output.as_ref().map_or(output, |p| &p.view)
    else {
        return false;
    };
    if view.as_ref().is_some_and(|v| {
        !matches!(
            DXGI_FORMAT(v[0] as i32),
            DXGI_FORMAT_UNKNOWN | DXGI_FORMAT_R32_UINT
        ) || v[1] != D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32
            || v[2] != 0
            || v[3] != 0
    }) {
        return false;
    }
    let Some(light) = host_resource(*output as This) else {
        return false;
    };
    // SAFETY: a live bound resource.
    let d = unsafe { light.GetDesc() };
    if !texture_2d(&d)
        || d.Format != DXGI_FORMAT_R32_UINT
        || !d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
    {
        return false;
    }

    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS.get_or_init(|| {
        make(&host.device)
            .inspect_err(|error| tracing::warn!(%error, "no ambient light"))
            .ok()
    }) else {
        return false;
    };
    let mut views = list.ambient_views.lock().unwrap_or_else(|e| e.into_inner());
    let Ok((heap, slot)) = views.next(&host.device) else {
        return false;
    };

    let placed_depth = depth.and_then(placed_texture);
    let depth = placed_depth
        .as_ref()
        .map(|p| &p.view)
        .or(depth)
        .and_then(|view| {
            let View::Shader(resource, view) = view else {
                return None;
            };
            if view
                .as_ref()
                .is_some_and(|v| crate::resource::first_mip(v) != 0)
            {
                return None;
            }
            let texture = host_resource(*resource as This)?;
            // SAFETY: a live bound resource. The resolve's bind has already
            // made this depth plane readable and resolved its twin if needed.
            let desc = unsafe { texture.GetDesc() };
            (texture_2d(&desc)
                && desc.Format == DXGI_FORMAT_R32G8X24_TYPELESS
                && (desc.Width, desc.Height) == (d.Width, d.Height)
                && !desc
                    .Flags
                    .contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
            .then_some(texture)
        });
    let sky = if settings.sky[0] > 0.0 {
        SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()).clone()
    } else {
        None
    };
    let sky = sky.as_ref().map(|image| {
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        if let Some(before) = list.transition_copy(image, read.0 as u32) {
            let barrier =
                crate::host::transition(&image.texture, D3D12_RESOURCE_STATES(before as i32), read);
            // SAFETY: the list is open and its state tracker holds the copy.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
        &image.texture
    });

    let [r, g, b] = settings.tint;
    let [strength, radius, near, tan_y] = settings.shade;
    let strength = if depth.is_some() && radius > 0.0 && near > 0.0 && tan_y > 0.0 {
        strength
    } else {
        0.0
    };
    let [gain, least] = sky.map_or([0.0; 2], |_| settings.sky);
    let held = if reflection { settings.held } else { 0.0 };
    let constants = [
        settings.level,
        r,
        g,
        b,
        strength,
        radius,
        near,
        tan_y,
        gain,
        least,
        held,
        settings.floor,
    ];
    let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    let srv = |format| D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    list.keep(light);
    if let Some(depth) = depth {
        list.keep(depth);
    }
    let l = &list.list;
    uav_barrier(l);
    // SAFETY: the list holds the textures and heap through execution. This
    // table has not been recorded before, and the resolve bound light as a UAV.
    unsafe {
        let cpu = heap.GetCPUDescriptorHandleForHeapStart();
        let gpu = heap.GetGPUDescriptorHandleForHeapStart();
        let at = |offset| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + ((slot + offset) * pass.step) as usize,
        };
        host.device
            .CreateUnorderedAccessView(light, None::<&ID3D12Resource>, Some(&uav), at(0));
        host.device.CreateShaderResourceView(
            depth,
            Some(&srv(DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS)),
            at(1),
        );
        host.device
            .CreateShaderResourceView(sky, Some(&srv(DXGI_FORMAT_R16G16B16A16_FLOAT)), at(2));
        l.SetDescriptorHeaps(&[Some(heap.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from(slot * pass.step),
            },
        );
        l.Dispatch((d.Width as u32).div_ceil(8), d.Height.div_ceil(8), 1);
    }
    uav_barrier(l);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_defaults_and_rt_opt_in() {
        assert_eq!(select_level("", false), Some(0.6));
        assert_eq!(select_level("", true), None);
        assert_eq!(select_level("off", false), None);
        assert_eq!(select_level("off", true), None);
        assert_eq!(select_level("0", true), Some(0.0));
        assert_eq!(select_level("0.4", true), Some(0.4));
        assert_eq!(select_level("NaN", true), None);
        assert_eq!(select_level("-1", false), Some(0.6));
    }

    #[test]
    fn ambient_shader_compiles() {
        if let Some(host) = host() {
            make(&host.device).expect("ambient pipeline");
        } else {
            crate::host::compile(SOURCE, s!("main"), s!("cs_5_1")).expect("ambient shader");
        }
    }
}
