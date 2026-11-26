//! Opt-in output size, render scale and upscaler selection for the title.
//! Patch sites in the mapped executable are checked against the supported build.

use windows_sys::Win32::System::Diagnostics::Debug::{FlushInstructionCache, ReadProcessMemory};
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const MODES: [(u32, u32); 5] = [
    (1280, 720),
    (1664, 936),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
];

fn parse_output(spec: &str) -> Option<(u32, u32)> {
    let (width, height) = spec.trim().split_once(['x', 'X'])?;
    let size = (width.trim().parse().ok()?, height.trim().parse().ok()?);
    MODES.contains(&size).then_some(size)
}

fn parse_scale(spec: &str) -> Option<f32> {
    spec.trim()
        .parse::<f32>()
        .ok()
        .filter(|s| (0.5..=1.0).contains(s))
}

fn parse_upscaler(spec: &str) -> Option<u32> {
    spec.trim()
        .parse::<u32>()
        .ok()
        .filter(|kind| matches!(kind, 6 | 7))
}

fn requested<T>(name: &str, parse: fn(&str) -> Option<T>, expected: &str) -> Option<T> {
    let spec = std::env::var(name).ok()?;
    let value = parse(&spec);
    if value.is_none() {
        tracing::warn!(name, value = spec, expected, "invalid resolution setting");
    }
    value
}

/// A place to change: its offset in the image and the bytes there
/// (compared first; the replacement is at most as long).
struct Site {
    at: usize,
    what: &'static str,
    bytes: &'static [u8],
}

/// The screen size setter, its screen height, and what +0x22434a0 calls when
/// the height changed.
const SET_SCREEN_SIZE: usize = 0x2478130;
const SCREEN_HEIGHT: usize = 0xa9438d0;
const HEIGHT_CHANGED: usize = 0xd8c3a0;

/// +0x22434a0 from +0x224352b to +0x2243568: the settings' size handed to
/// the screen size.
const OUTPUT: Site = Site {
    at: 0x224352b,
    what: "settings' output size applied",
    bytes: &[
        0x8b, 0x0d, 0x9b, 0x03, 0x70, 0x08, // mov ecx, [+0xa9438cc]
        0x8b, 0x86, 0x04, 0x02, 0x00, 0x00, // mov eax, [rsi+204h]
        0x39, 0xc1, 0x75, 0x0d, // cmp ecx, eax ; jne
        0x8b, 0x96, 0x08, 0x02, 0x00, 0x00, // mov edx, [rsi+208h]
        0xe8, 0xea, 0x4b, 0x23, 0x00, 0xeb, 0x20, // call +0x2478130 ; jmp +0x2243568
        0x8b, 0xbe, 0x08, 0x02, 0x00, 0x00, // mov edi, [rsi+208h]
        0x8b, 0x2d, 0x7c, 0x03, 0x70, 0x08, // mov ebp, [+0xa9438d0]
        0x89, 0xc1, 0x89, 0xfa, // mov ecx, eax ; mov edx, edi
        0xe8, 0xd3, 0x4b, 0x23, 0x00, // call +0x2478130
        0x39, 0xfd, 0x74, 0x07, 0xb1, 0x01, // cmp ebp, edi ; je ; mov cl, 1
        0xe8, 0x38, 0x8e, 0xb4, 0xfe, // call +0xd8c3a0
    ],
};

/// +0x247ad10, after the back buffers are made in their pool (64 MB on
/// Series X, 82 MB on Series S, from +0x247ad45): `int3` unless less than
/// 2 MB of the pool is left over. Sized for 3840x2160; at 1920x1080 49 MB
/// are left. With a smaller output size the check
/// becomes a jump over the `int3`; the pool stays as large.
const POOL_CHECK: Site = Site {
    at: 0x247af25,
    what: "back buffer pool's left-over check",
    bytes: &[0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01, 0xcc], // cmp rax, 1FFFFFh ; jbe +1 ; int3
};

/// What goes at [`POOL_CHECK`]: `jmp +1` in place of `jbe +1`.
fn pool_check_code() -> Vec<u8> {
    let mut c = POOL_CHECK.bytes[..8].to_vec();
    c[6] = 0xeb;
    c
}
/// The title's own upscaler, its method +0x28f1e70: `mov dword [rcx+90h],
/// 2/3` (the scale +0x7e9440 renders at).
const OWN_SCALE: Site = Site {
    at: 0x28f1e78,
    what: "the title's own upscaler's render scale",
    bytes: &[0xc7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xab, 0xaa, 0x2a, 0x3f],
};

/// FidelityFX's render size method, +0x28ef680, and its padding: the scale
/// is 1.0 (+0x5fc6ed8) or, on Series S, 2/3 (+0x615d95c), loaded at +0x28
/// and +0x32.
const FFX_SCALE: Site = Site {
    at: 0x28ef680,
    what: "FidelityFX's render scale",
    bytes: &[
        0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x28, 0x44, 0x89, 0xcb, 0x4c, 0x89, 0xc6,
        0x48, 0x89, 0xd7, 0x49, 0x89, 0xce, 0xe8, 0xa6, 0x5a, 0x78, 0xfd, 0x0f, 0xb6, 0xc0, 0x41,
        0x89, 0x86, 0x90, 0x00, 0x00, 0x00, 0x84, 0xc0, 0x75, 0x0a, 0xc5, 0xfa, 0x10, 0x05, 0x28,
        0x78, 0x6d, 0x03, 0xeb, 0x08, 0xc5, 0xfa, 0x10, 0x05, 0xa2, 0xe2, 0x86, 0x03, 0xc5, 0xf2,
        0x2a, 0x54, 0x24, 0x70, 0xc5, 0xf2, 0x2a, 0xcb, 0xc5, 0xfa, 0x59, 0xc9, 0xc5, 0xfa, 0x2c,
        0xc1, 0x89, 0x07, 0xc5, 0xfa, 0x59, 0xc2, 0xc5, 0xfa, 0x2c, 0xc0, 0x89, 0x06, 0x48, 0x83,
        0xc4, 0x28, 0x5b, 0x5f, 0x5e, 0x41, 0x5e, 0xc3, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
    ],
};

/// The upscaler picker, +0x7e93f0, to its first `ret`.
const PICKER: Site = Site {
    at: 0x7e93f0,
    what: "upscaler picker",
    bytes: &[
        0x48, 0x83, 0xec, 0x28, 0x80, 0x3d, 0x6d, 0xf0, 0x3a, 0x07, 0x00, 0x75, 0x09, 0x83, 0x3d,
        0x68, 0xf0, 0x3a, 0x07, 0x00, 0x74, 0x28, 0xe8, 0x35, 0xbd, 0x88, 0xff, 0x8a, 0x0d, 0x5b,
        0xf0, 0x3a, 0x07, 0x80, 0xe1, 0x40, 0x80, 0xf9, 0x01, 0xb9, 0x07, 0x00, 0x00, 0x00, 0x83,
        0xd9, 0x00, 0x84, 0xc0, 0xb8, 0x06, 0x00, 0x00, 0x00, 0x0f, 0x45, 0xc1, 0x48, 0x83, 0xc4,
        0x28, 0xc3,
    ],
};

/// `rel32` from the end of an instruction at image offset `next` to `to`.
fn rel(to: usize, next: usize) -> [u8; 4] {
    ((to as i64 - next as i64) as i32).to_le_bytes()
}

/// What goes at [`OUTPUT`] for `width` x `height` (the same length).
fn output_code(width: u32, height: u32) -> Vec<u8> {
    let at = OUTPUT.at;
    let mut c = Vec::with_capacity(OUTPUT.bytes.len());
    c.push(0xb8); // mov eax, width
    c.extend(width.to_le_bytes());
    c.extend([0x89, 0x86, 0x04, 0x02, 0x00, 0x00]); // mov [rsi+204h], eax
    c.push(0xbf); // mov edi, height
    c.extend(height.to_le_bytes());
    c.extend([0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]); // mov [rsi+208h], edi
    c.extend([0x8b, 0x2d]); // mov ebp, [screen height]
    c.extend(rel(SCREEN_HEIGHT, at + c.len() + 4));
    c.extend([0x89, 0xc1, 0x89, 0xfa]); // mov ecx, eax ; mov edx, edi
    c.push(0xe8); // call the setter
    c.extend(rel(SET_SCREEN_SIZE, at + c.len() + 4));
    let end = OUTPUT.bytes.len();
    c.extend([0x39, 0xfd, 0x74, (end - (c.len() + 4)) as u8]); // cmp ebp, edi ; je end
    c.extend([0xb1, 0x01, 0xe8]); // mov cl, 1 ; call
    c.extend(rel(HEIGHT_CHANGED, at + c.len() + 4));
    c.extend([0xeb, (end - (c.len() + 2)) as u8]); // jmp end
    c.resize(end, 0xcc);
    c
}

/// What goes at [`OWN_SCALE`] for `scale`.
fn own_scale_code(scale: f32) -> Vec<u8> {
    let mut c = OWN_SCALE.bytes[..6].to_vec();
    c.extend(scale.to_le_bytes());
    c
}

/// What goes at [`FFX_SCALE`] for `scale`: both loads read `scale`, written
/// into the padding at +0x64.
fn ffx_scale_code(scale: f32) -> Vec<u8> {
    const LOADS: [usize; 2] = [0x28, 0x32];
    const SLOT: usize = 0x64;
    let mut c = FFX_SCALE.bytes.to_vec();
    c[SLOT..SLOT + 4].copy_from_slice(&scale.to_le_bytes());
    for l in LOADS {
        c[l + 4..l + 8].copy_from_slice(&((SLOT - (l + 8)) as i32).to_le_bytes());
    }
    c
}

/// `mov eax, kind ; ret`.
fn picker_code(kind: u32) -> Vec<u8> {
    let mut c = vec![0xb8];
    c.extend(kind.to_le_bytes());
    c.push(0xc3);
    c
}

// Read and protect every site in the group before writing the first one.
// In particular, a smaller output requires the matching pool adjustment.
unsafe fn apply_group<'a>(
    base: usize,
    size: usize,
    patches: &[(&'a Site, Vec<u8>)],
) -> Result<(), (&'a Site, &'static str)> {
    // SAFETY: the pseudo-handle refers to this process.
    let process = unsafe { GetCurrentProcess() };
    for (site, code) in patches {
        if code.len() > site.bytes.len()
            || site
                .at
                .checked_add(site.bytes.len())
                .is_none_or(|end| end > size)
        {
            return Err((site, "patch outside the image"));
        }
        let mut found = vec![0; site.bytes.len()];
        // SAFETY: the destination holds the signature. The OS checks the source.
        let read = unsafe {
            ReadProcessMemory(
                process,
                (base + site.at) as *const _,
                found.as_mut_ptr().cast(),
                found.len(),
                std::ptr::null_mut(),
            )
        };
        if read == 0 || found != site.bytes {
            return Err((site, "signature differs from the supported build"));
        }
    }

    let mut protections = Vec::with_capacity(patches.len());
    let mut result = Ok(());
    for (site, code) in patches {
        let at = (base + site.at) as *mut u8;
        let mut old = 0;
        // SAFETY: this range passed the signature check and stays mapped.
        if unsafe { VirtualProtect(at.cast(), code.len(), PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            result = Err((*site, "could not make the patch site writable"));
            break;
        }
        protections.push((*site, at, code.len(), old));
    }
    if result.is_ok() {
        for ((_, code), &(_, at, len, _)) in patches.iter().zip(&protections) {
            // SAFETY: the destination is writable; the caller excludes execution.
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), at, len) };
        }
    }
    let wrote = result.is_ok();
    for (site, at, len, old) in protections.into_iter().rev() {
        let mut replaced = 0;
        // SAFETY: restore the page permissions saved above.
        if unsafe { VirtualProtect(at.cast(), len, old, &mut replaced) } == 0 && result.is_ok() {
            result = Err((site, "could not restore the patch site's protection"));
        }
        // SAFETY: this range is part of the mapped executable.
        if wrote && unsafe { FlushInstructionCache(process, at.cast(), len) } == 0 && result.is_ok()
        {
            result = Err((site, "could not flush the instruction cache"));
        }
    }
    result
}

/// Apply the requested resolution overrides to the mapped executable.
///
/// # Safety
/// The image stays mapped and its code has yet to run.
pub unsafe fn apply_asked(base: usize, size: usize) {
    let apply = |name, patches: &[(&Site, Vec<u8>)]| {
        // SAFETY: the caller keeps the image mapped and excludes execution.
        match unsafe { apply_group(base, size, patches) } {
            Ok(()) => tracing::info!(name, "resolution override applied"),
            Err((site, reason)) => {
                tracing::warn!(
                    name,
                    site = site.what,
                    offset = site.at,
                    reason,
                    "resolution override failed"
                );
            }
        }
    };

    if let Some((width, height)) = requested(
        "VICTORY_OUTPUT",
        parse_output,
        "1280x720, 1664x936, 1920x1080, 2560x1440, or 3840x2160",
    ) {
        let mut patches = vec![(&OUTPUT, output_code(width, height))];
        if (width, height) != (3840, 2160) {
            patches.push((&POOL_CHECK, pool_check_code()));
        }
        apply("VICTORY_OUTPUT", &patches);
    }
    if let Some(scale) = requested("VICTORY_RENDER_SCALE", parse_scale, "0.5 through 1.0") {
        apply(
            "VICTORY_RENDER_SCALE",
            &[(&OWN_SCALE, own_scale_code(scale))],
        );
    }
    // The FidelityFX override is experimental. Some configurations cannot
    // calculate a jitter phase count at reduced scales.
    if let Some(scale) = requested("VICTORY_FFX_SCALE", parse_scale, "0.5 through 1.0") {
        apply("VICTORY_FFX_SCALE", &[(&FFX_SCALE, ffx_scale_code(scale))]);
    }
    if let Some(kind) = requested(
        "VICTORY_UPSCALER",
        parse_upscaler,
        "6 (native) or 7 (FidelityFX)",
    ) {
        apply("VICTORY_UPSCALER", &[(&PICKER, picker_code(kind))]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_switches_accept_the_supported_values() {
        assert_eq!(parse_output("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_output(" 2560X1440 "), Some((2560, 1440)));
        assert_eq!(parse_output("1600x900"), None);
        assert_eq!(parse_output("1080p"), None);
        assert_eq!(parse_scale("0.5"), Some(0.5));
        assert_eq!(parse_scale("1"), Some(1.0));
        assert_eq!(parse_scale("NaN"), None);
        assert_eq!(parse_scale("1.5"), None);
        assert_eq!(parse_upscaler("7"), Some(7));
        assert_eq!(parse_upscaler("8"), None);
    }

    #[test]
    fn replacement_output_block_sets_dimensions_and_keeps_the_height_callback() {
        for (width, height) in [(1920u32, 1080u32), (2560, 1440)] {
            let code = output_code(width, height);
            assert_eq!(code[1..5], width.to_le_bytes());
            assert_eq!(code[12..16], height.to_le_bytes());
            let target = |end: usize| {
                let displacement = i32::from_le_bytes(code[end - 4..end].try_into().unwrap());
                (OUTPUT.at + end)
                    .checked_add_signed(displacement as isize)
                    .unwrap()
            };
            assert_eq!(target(28), SCREEN_HEIGHT);
            assert_eq!(target(37), SET_SCREEN_SIZE);
            assert_eq!(target(48), HEIGHT_CHANGED);
            assert_eq!(41 + usize::from(code[40]), code.len());
            assert_eq!(50 + usize::from(code[49]), code.len());
        }
    }

    #[test]
    fn scale_replacements_point_to_the_requested_value() {
        assert_eq!(own_scale_code(0.75)[6..], 0.75f32.to_le_bytes());
        let code = ffx_scale_code(0.75);
        for end in [0x30, 0x3a] {
            let displacement = i32::from_le_bytes(code[end - 4..end].try_into().unwrap());
            let target = end.checked_add_signed(displacement as isize).unwrap();
            assert_eq!(code[target..target + 4], 0.75f32.to_le_bytes());
        }
        assert_eq!(picker_code(7), [0xb8, 7, 0, 0, 0, 0xc3]);
        assert_eq!(pool_check_code()[6..], [0xeb, 1]);
    }

    #[test]
    fn a_signature_mismatch_prevents_the_entire_group_from_being_written() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        let first = Site {
            at: 0x10,
            what: "test first",
            bytes: &[1, 2],
        };
        let second = Site {
            at: 0x20,
            what: "test second",
            bytes: &[3, 4],
        };
        let patches = [(&first, vec![10, 20]), (&second, vec![30, 40])];
        // SAFETY: these operations use a private page that is never executed.
        unsafe {
            let page = VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
            .cast::<u8>();
            assert!(!page.is_null());
            std::ptr::copy_nonoverlapping(first.bytes.as_ptr(), page.add(first.at), 2);
            let mut old = 0;
            assert_ne!(
                VirtualProtect(page.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old),
                0
            );
            assert!(apply_group(page as usize, 0x1000, &patches).is_err());
            assert_eq!(std::slice::from_raw_parts(page.add(first.at), 2), [1, 2]);

            assert_ne!(
                VirtualProtect(page.cast(), 0x1000, PAGE_READWRITE, &mut old),
                0
            );
            std::ptr::copy_nonoverlapping(second.bytes.as_ptr(), page.add(second.at), 2);
            assert!(apply_group(page as usize, 0x1000, &patches).is_ok());
            assert_eq!(std::slice::from_raw_parts(page.add(first.at), 2), [10, 20]);
            assert_eq!(std::slice::from_raw_parts(page.add(second.at), 2), [30, 40]);
            assert_ne!(VirtualFree(page.cast(), 0, MEM_RELEASE), 0);
        }
    }
}
