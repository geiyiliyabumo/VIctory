//! Renderer port #1: the title's 32-bit graphics heap as the card's own
//! memory (`VICTORY_PORT_CONSTANTS`, off by default).
//!
//! What the title does (game_scarlett_master.exe, the routine at image
//! +0x1e86440, reached at its start): one `XMemVirtualAlloc` of the five
//! sizes it is given together (kind 0x42000: a reservation of 64 KiB
//! pages, flags 7: graphics memory below 4 GiB, 185 MB at 0x34310000 in
//! every run so far), split into five placeholders (`VirtualFree` with
//! `MEM_PRESERVE_PLACEHOLDER`), each turned into graphics memory of its own
//! protection (`XMemVirtualAlloc`, kind 0x20406000, flags 0x27); then ONE
//! `XMemAllocatePhysicalPages` for all their pages and five
//! `XMemMapPhysicalPages`, in order, each from the next page of that one
//! array. `xmem` makes one section of all the pages, so the whole
//! reservation is one stretch of one section: views that follow one
//! another in memory and in the section alike. The title's 174 MB buffer at
//! 0x34e10000 (where its ring of constants lies, and what the skinning
//! passes write through one view of all of it) is in it; it is never
//! tile-mapped (`CopyPageMappings` maps the GPU heap's pages, not these).
//!
//! What the layer did with it: a host copy in the card's memory, the title's
//! writes found and sent up (17 MB a frame in ~600 pieces), what shaders
//! wrote fetched back and merged, and every constant buffer read out of it
//! (~6,000 fills a frame) either waited for flights or done on the card
//! from the copy.
//!
//! What is done here instead: the stretch is mapped once more, at an
//! address of ours (a view of the same section: the same physical pages),
//! and opened as a host heap there (`OpenExistingHeapFromAddress1`); title
//! buffers and windows of title memory in it are buffers placed in that
//! heap ([`crate::shadow::Shadow`]'s in-place form): the card reads and
//! writes the title's bytes where the title wrote them. Nothing is tracked,
//! sent up, fetched back or compared for it, and no compare or hash pass
//! of ours ever runs over it (they are for copies in the card's memory:
//! one such pass over memory shared with the PC removed the device on
//! another tree). No tile mapping is ever made into this heap (that removes
//! the device).
//!
//! Why our own view: a heap opened from an address holds the pages it was
//! opened on: the title could not unmap, decommit or protect them while it
//! stands. Opened on our view, the title's own views stay free to change.
//! When the title maps other pages at such an address, what is in place
//! there is let go (`crate::inplace::title_memory_freeing`) and the region
//! forgotten; the view of ours stays (shortcut: it is never unmapped, a
//! few hundred MB of address space at most).
//!
//! What it assumes: the CPU steps of ours that read this memory at
//! submission (fills of constant buffers, the title's buffer copies done on
//! the CPU) read what the title's CPU wrote, not what a submission still in
//! the air writes there: in place nothing tells the flights' writes, so
//! nothing waits for them. The title's constants are its CPU's; how many
//! such steps there are is on the frame-rate line ([`log`]).

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, OnceLock, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory as mem;

use crate::inplace::{Placed, cover, stretch};

const K64: u64 = 64 << 10;

/// The ported memory lies below this: the title's graphics memory for
/// 32-bit addresses, which only that heap is.
const LOW: u64 = 1 << 32;

/// The most bytes of one region: its pages are locked into the PC's memory
/// while the heap stands.
const MOST: u64 = 512 << 20;

/// `VICTORY_PORT_CONSTANTS`: `1` turns renderer port #1 on (title buffers
/// and windows of title memory in the title's 32-bit graphics memory, view
/// memory that is one stretch of a section, used in place); unset or
/// anything else: off, everything as before.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Read once, here: the switch of renderer port #1.
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_CONSTANTS").is_ok_and(|v| v.trim() == "1"))
}

/// `VICTORY_PORT_HIGH`: `1` enables in-place opening for >4 GiB `stretch=1`
/// (contiguous section run) title buffers >= `VICTORY_PORT_HIGH_MIN_KB`
/// (default 1024 KiB), opening only each buffer's exact 64 KiB pages through
/// our own `MapViewOfFile` view, up to `VICTORY_PORT_HIGH_MB` (default 512 MB).
fn high_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_HIGH").is_ok_and(|v| v.trim() == "1"))
}

/// `VICTORY_PORT_PRIVATE`: `1` enables in-place opening for >4 GiB
/// `MEM_PRIVATE | MEM_COMMIT | PAGE_READWRITE` title buffers >=
/// `VICTORY_PORT_PRIVATE_MIN_KB` (default 1024 KiB) via
/// `OpenExistingHeapFromAddress1`, up to `VICTORY_PORT_PRIVATE_MB` (default 512 MB),
/// optionally capped per-buffer by `VICTORY_PORT_PRIVATE_MAX_MB` (0 = unlimited).
fn private_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_PRIVATE").is_ok_and(|v| v.trim() == "1"))
}

struct PortLimits {
    budget: u64,
    min: u64,
    max: u64,
}

impl PortLimits {
    fn read(budget: &str, min: &str, max: &str) -> Self {
        let amount = |name, fallback, unit| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(fallback)
                .saturating_mul(unit)
        };
        Self {
            budget: amount(budget, 512, 1 << 20),
            min: amount(min, 1024, 1 << 10),
            max: amount(max, 0, 1 << 20),
        }
    }

    fn allows(&self, len: u64) -> bool {
        len >= self.min && (self.max == 0 || len <= self.max)
    }
}

fn high_limits() -> &'static PortLimits {
    static LIMITS: OnceLock<PortLimits> = OnceLock::new();
    LIMITS.get_or_init(|| PortLimits::read("VICTORY_PORT_HIGH_MB", "VICTORY_PORT_HIGH_MIN_KB", "VICTORY_PORT_HIGH_MAX_MB"))
}

fn private_limits() -> &'static PortLimits {
    static LIMITS: OnceLock<PortLimits> = OnceLock::new();
    LIMITS.get_or_init(|| PortLimits::read("VICTORY_PORT_PRIVATE_MB", "VICTORY_PORT_PRIVATE_MIN_KB", "VICTORY_PORT_PRIVATE_MAX_MB"))
}

/// Optional comma-separated hex start addresses (`VICTORY_PORT_EXCLUDE`)
/// skipped by `PORT_HIGH` / `PORT_PRIVATE` (for quick A/B without rebuild).
fn excluded(start: u64) -> bool {
    static LIST: OnceLock<Vec<u64>> = OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("VICTORY_PORT_EXCLUDE")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|part| {
                        let t = part.trim().trim_start_matches("0x").trim_start_matches("0X");
                        u64::from_str_radix(t, 16).ok()
                    })
                    .collect()
            })
            .unwrap_or_default()
    });
    list.contains(&start)
}

/// Allow learned high-address constants to use raw views when port 5 does.
pub(crate) fn late_cb_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    crate::port_cbv::in_place_on()
        && *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_LATE_CB").is_none_or(|v| v != "0"))
}

/// A stretch of title memory opened as a heap through a view of ours.
struct Region {
    start: u64,
    end: u64,
    heap: ID3D12Heap,
}

// SAFETY: D3D12 heaps are free-threaded.
unsafe impl Send for Region {}

static REGIONS: Mutex<Vec<Region>> = Mutex::new(Vec::new());

/// The lowest start and highest end of every low (<4 GiB) region (both 0: none),
/// for a quick look on the paths that run thousands of times a frame.
static BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Overall `[min(start), max(end)]` across ALL entries in `REGIONS` (any address),
/// so `forget()` never skips an active region regardless of low/high address.
static REGIONS_BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Exact `[start, end)` spans of high (>=4 GiB) in-place buffers (mapped + private)
/// plus fast-path `[lo, hi]` bounds so `filled`/`copied` stay zero-overhead.
static HIGH_BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static HIGH_SPANS: RwLock<Vec<(u64, u64)>> = RwLock::new(Vec::new());

/// High/private port stats:
/// 0: high mapped buffers placed, 1: high mapped bytes placed,
/// 2: high mapped refused (not stretch / budget / host),
/// 3: private buffers placed, 4: private bytes placed,
/// 5: private refused (not private / budget / host),
/// 6: fills touching high/private, 7: fill bytes,
/// 8: copies touching high/private, 9: copy bytes.
static HIGH_STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

const LATE_CB_MAX_BYTES: u64 = 4 << 20;
type LateCbPages = HashMap<(u64, u64), Option<Placed>>;
// Failed opens are cached until the title changes the memory at that address.
static LATE_CB_PAGES: LazyLock<RwLock<LateCbPages>> = LazyLock::new(|| RwLock::new(HashMap::new()));
static LATE_CB_BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static LATE_CB_BYTES: AtomicU64 = AtomicU64::new(0);
// Pages and bytes opened, cache hits, failed or refused opens.
static LATE_CB_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Since the start: regions opened, their bytes; title buffers and windows
/// placed; refused because the memory is not one stretch of a section,
/// not committed, refused by the host; regions forgotten because the title
/// mapped other pages there. Then, since the last line: fills read out of
/// ported memory on the CPU, their bytes, buffer copies on the CPU that
/// touch it, their bytes.
static STATS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

/// Whether `[at, at + len)` may be ported as renderer port #1: in the title's
/// 32-bit graphics memory.
pub(crate) fn wanted(at: u64, len: u64) -> bool {
    on() && at.checked_add(len).is_some_and(|end| end <= LOW && len > 0)
}

/// Title memory `[start, start + len)` in place, if eligible under
/// `VICTORY_PORT_CONSTANTS` (<4 GiB), `VICTORY_PORT_HIGH` (>4 GiB `stretch=1`
/// title buffers), or `VICTORY_PORT_PRIVATE` (>4 GiB `MEM_PRIVATE` title
/// buffers). `None` leaves the caller to make a copy as before.
pub(crate) fn place_for(start: u64, len: u64, window: bool) -> Option<Placed> {
    if wanted(start, len) {
        let host = crate::host::host()?;
        return match place_in(&host.device, start, len) {
            Ok(placed) => {
                crate::inplace::counted(window, len);
                let n = STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 32 || n.is_power_of_two() {
                    tracing::info!(
                        start = format_args!("{start:#x}"),
                        len,
                        window,
                        "renderer port 1: title memory in place, read and written by the card where the title has it"
                    );
                }
                Some(placed)
            }
            Err((at, why)) => {
                let n = STATS[at].fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 8 || n.is_power_of_two() {
                    tracing::warn!(start = format_args!("{start:#x}"), len, window, why, "renderer port 1: title memory not in place, a copy as before");
                }
                None
            }
        };
    }
    if window || start < LOW || len == 0 || !(high_on() || private_on()) || excluded(start) {
        return None;
    }
    if high_on() && high_limits().allows(len) && let Some(placed) = place_high(start, len) {
        return Some(placed);
    }
    if private_on() && private_limits().allows(len) {
        return place_private(start, len);
    }
    None
}

fn overlaps(bounds: &[AtomicU64; 2], at: u64, end: u64) -> bool {
    at < end && at < bounds[1].load(Ordering::Acquire) && end > bounds[0].load(Ordering::Relaxed)
}

fn span_bounds(bounds: &[AtomicU64; 2], spans: impl Iterator<Item = (u64, u64)>) {
    let (lo, hi) = spans.fold((u64::MAX, 0), |(lo, hi), (s, e)| (lo.min(s), hi.max(e)));
    bounds[0].store(if hi == 0 { 0 } else { lo }, Ordering::Relaxed);
    bounds[1].store(hi, Ordering::Release);
}

fn aligned_range(start: u64, len: u64) -> Option<(u64, u64)> {
    let end = start.checked_add(len)?;
    Some((start & !(K64 - 1), end.checked_add(end.wrapping_neg() & (K64 - 1))?))
}

fn note_high_span(spans: &mut Vec<(u64, u64)>, at: u64, end: u64) {
    spans.push((at, end));
    span_bounds(&HIGH_BOUNDS, spans.iter().copied());
}

fn place_high(start: u64, len: u64) -> Option<Placed> {
    let (at, end) = aligned_range(start, len)?;
    let host = crate::host::host()?;
    // Serialize admission so concurrent placements cannot overspend the budget.
    let mut spans = HIGH_SPANS.write().unwrap_or_else(|e| e.into_inner());
    let bytes = end - at;
    if bytes > MOST || HIGH_STATS[1].load(Ordering::Relaxed).saturating_add(bytes) > high_limits().budget {
        HIGH_STATS[2].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // Only open contiguous section runs (`Cover::Run`).
    match place_high_region(&host.device, start, at, end) {
        Ok(placed) => {
            crate::inplace::counted(false, len);
            let n = HIGH_STATS[0].fetch_add(1, Ordering::Relaxed) + 1;
            HIGH_STATS[1].fetch_add(bytes, Ordering::Relaxed);
            note_high_span(&mut spans, at, end);
            if n <= 64 || n.is_power_of_two() {
                tracing::info!(
                    start = format_args!("{start:#x}"),
                    mb = format_args!("{:.2}", len as f64 / 1048576.0),
                    "renderer port high (VICTORY_PORT_HIGH=1): >4GiB stretch buffer in place"
                );
            }
            Some(placed)
        }
        Err(None) => {
            // Not a contiguous stretch (e.g. MEM_PRIVATE or Cover::Scattered): normal, silent.
            None
        }
        Err(Some(why)) => {
            let n = HIGH_STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                tracing::warn!(start = format_args!("{start:#x}"), len, why, "renderer port high: not opened in place, a copy as before");
            }
            None
        }
    }
}

/// Opens the exact slice of a section run. Late CB pages use their own heap,
/// so retaining a broad high-region cache entry cannot extend their lifetime.
fn place_high_in(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, Option<String>> {
    let (at, end) = aligned_range(start, len).ok_or(None)?;
    let (region, view) = open_high_region(device, at, end)?;
    // As in port 1, a successful alias stays mapped for the process lifetime.
    place_high_buffer(device, &region, start, at, end).inspect_err(|_| {
        drop(region);
        unmap_view(view);
    })
}

fn place_high_region(device: &ID3D12Device, start: u64, at: u64, end: u64) -> Result<Placed, Option<String>> {
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(region) = regions.iter().find(|r| r.start <= at && end <= r.end) {
        return place_high_buffer(device, region, start, at, end);
    }
    let (region, view) = open_high_region(device, at, end)?;
    match place_high_buffer(device, &region, start, at, end) {
        Ok(placed) => {
            regions.push(region);
            bounds(&regions);
            Ok(placed)
        }
        Err(error) => {
            drop(region);
            unmap_view(view);
            Err(error)
        }
    }
}

fn place_high_buffer(device: &ID3D12Device, region: &Region, start: u64, at: u64, end: u64) -> Result<Placed, Option<String>> {
    let buffer = crate::inplace::place(device, &region.heap, at - region.start, end - at)
        .map_err(|e| Some(format!("CreatePlacedResource failed: {e}")))?;
    let bias = start - at;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + bias;
    Ok(Placed { buffer, heap: region.heap.clone(), gpu, bias })
}

fn open_high_region(device: &ID3D12Device, at: u64, end: u64) -> Result<(Region, usize), Option<String>> {
    let bytes = end - at;
    let Some((first, stretch_bytes, section, stretch_offset)) = stretch(at as usize, bytes as usize) else {
        return Err(None);
    };
    let first = first as u64;
    if first > at || first + (stretch_bytes as u64) < end {
        return Err(None);
    }
    let offset = stretch_offset + (at - first);
    if !offset.is_multiple_of(K64) {
        return Err(Some(format!("unaligned section offset {offset:#x}")));
    }
    // SAFETY: the runtime retains the section handle while the title has a
    // view, and the requested range was found in that section.
    let view = unsafe { mem::MapViewOfFile(section as _, mem::FILE_MAP_WRITE, (offset >> 32) as u32, offset as u32, bytes as usize) }.Value as usize;
    if view == 0 {
        return Err(Some(format!("MapViewOfFile failed: {}", std::io::Error::last_os_error())));
    }
    if !committed(view, bytes as usize) {
        unmap_view(view);
        return Err(Some("pages of the section run are not committed".to_owned()));
    }
    match crate::inplace::open(device, crate::inplace::From::Part(view, bytes as usize)) {
        Ok(heap) => Ok((Region { start: at, end, heap }, view)),
        Err(error) => {
            unmap_view(view);
            Err(Some(format!("OpenExistingHeapFromAddress1 refused: {error}")))
        }
    }
}

fn unmap_view(view: usize) {
    // SAFETY: this view belongs to us and no heap remains open on it.
    unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
}

fn place_private(start: u64, len: u64) -> Option<Placed> {
    let (at, end) = aligned_range(start, len)?;
    let host = crate::host::host()?;
    // Serialize admission so concurrent placements cannot overspend the budget.
    let mut spans = HIGH_SPANS.write().unwrap_or_else(|e| e.into_inner());
    let bytes = end - at;
    if bytes > MOST || HIGH_STATS[4].load(Ordering::Relaxed).saturating_add(bytes) > private_limits().budget {
        HIGH_STATS[5].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // Must not be view memory (already handled by `place_high` or `Scattered`).
    if cover(at as usize, bytes as usize) != crate::inplace::Cover::None {
        return None;
    }
    match crate::inplace::place_part(&host.device, start, len) {
        Ok(placed) => {
            crate::inplace::counted(false, len);
            let n = HIGH_STATS[3].fetch_add(1, Ordering::Relaxed) + 1;
            HIGH_STATS[4].fetch_add(bytes, Ordering::Relaxed);
            note_high_span(&mut spans, at, end);
            if n <= 64 || n.is_power_of_two() {
                tracing::info!(
                    start = format_args!("{start:#x}"),
                    mb = format_args!("{:.2}", len as f64 / 1048576.0),
                    "renderer port private (VICTORY_PORT_PRIVATE=1): >4GiB private buffer in place"
                );
            }
            Some(placed)
        }
        Err(None) => None,
        Err(Some(e)) => {
            let n = HIGH_STATS[5].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                tracing::warn!(start = format_args!("{start:#x}"), len, error = %e, "renderer port private: refused by host, a copy as before");
            }
            None
        }
    }
}

pub(crate) fn place_late_cb(address: u64, len: u64) -> Option<(ID3D12Resource, u64)> {
    if !late_cb_on() {
        return None;
    }
    place_late_cb_in(&crate::host::host()?.device, address, len)
}

fn late_cb_hit(start: u64, address: u64, placed: &Placed) -> (ID3D12Resource, u64) {
    LATE_CB_STATS[2].fetch_add(1, Ordering::Relaxed);
    (placed.buffer.clone(), placed.gpu + (address - start))
}

fn cached_late_cb(pages: &LateCbPages, address: u64, at: u64, bytes: u64) -> Option<Option<(ID3D12Resource, u64)>> {
    if let Some(entry) = pages.get(&(at, bytes)) {
        return Some(entry.as_ref().map(|placed| late_cb_hit(at, address, placed)));
    }
    // Only a two-page entry can cover a differently sized or aligned key.
    if bytes == K64 {
        for start in [at, at - K64] {
            if let Some(Some(placed)) = pages.get(&(start, 2 * K64)) {
                return Some(Some(late_cb_hit(start, address, placed)));
            }
        }
    }
    None
}

/// Direct device entry point, also used by the WARP tests.
fn place_late_cb_in(device: &ID3D12Device, address: u64, len: u64) -> Option<(ID3D12Resource, u64)> {
    if len == 0 || len > K64 || address < LOW || excluded(address) {
        return None;
    }
    let (at, end) = aligned_range(address, len)?;
    let bytes = end - at;
    if overlaps(&LATE_CB_BOUNDS, at, end) {
        let pages = LATE_CB_PAGES.read().unwrap_or_else(|e| e.into_inner());
        if let Some(hit) = cached_late_cb(&pages, address, at, bytes) {
            return hit;
        }
    }
    let mut pages = LATE_CB_PAGES.write().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = cached_late_cb(&pages, address, at, bytes) {
        return hit;
    }

    // Publish the pending entry before opening, so an overlapping forget
    // waits for the mapping or tombstone and removes it under the same lock.
    pages.insert((at, bytes), None);
    late_cb_bounds(&pages);
    let result = if LATE_CB_BYTES.load(Ordering::Relaxed) + bytes > LATE_CB_MAX_BYTES {
        Err(Some("late CB page budget exhausted".to_owned()))
    } else {
        match place_high_in(device, at, bytes) {
            Ok(placed) => Ok((placed, "section")),
            Err(None) if cover(at as usize, bytes as usize) == crate::inplace::Cover::None => {
                crate::inplace::place_part(device, at, bytes)
                    .map(|placed| (placed, "private"))
                    .map_err(|error| error.map(|e| e.to_string()))
            }
            Err(error) => Err(error),
        }
    };
    match result {
        Ok((placed, kind)) => {
            LATE_CB_BYTES.fetch_add(bytes, Ordering::Relaxed);
            LATE_CB_STATS[0].fetch_add(bytes / K64, Ordering::Relaxed);
            LATE_CB_STATS[1].fetch_add(bytes, Ordering::Relaxed);
            tracing::info!(at = format_args!("{at:#x}"), bytes = format_args!("{bytes:#x}"), kind, "renderer port late CB: pages opened in place");
            let result = (placed.buffer.clone(), placed.gpu + (address - at));
            pages.insert((at, bytes), Some(placed));
            Some(result)
        }
        Err(error) => {
            if LATE_CB_STATS[3].fetch_add(1, Ordering::Relaxed) < 8 {
                let why = error.as_deref().unwrap_or("range is neither a section run nor committed private memory");
                tracing::warn!(at = format_args!("{at:#x}"), bytes = format_args!("{bytes:#x}"), why, "renderer port late CB: pages not opened in place");
            }
            None
        }
    }
}

fn late_cb_bounds(pages: &LateCbPages) {
    span_bounds(&LATE_CB_BOUNDS, pages.keys().map(|&(start, bytes)| (start, start + bytes)));
}

/// [`place_for`]'s work: the region holding the pages of `[start, start +
/// len)`, opened if none does, and a buffer placed in it. `Err` with the
/// statistic to count and why.
pub(crate) fn place_in(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, (usize, String)> {
    let at = start & !(K64 - 1);
    let end = (start + len.max(1)).next_multiple_of(K64);
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match regions.iter().position(|r| r.start <= at && end <= r.end) {
        Some(i) => i,
        None => {
            let region = open_region(device, at, end)?;
            regions.push(region);
            let n = regions.len() - 1;
            let r = &regions[n];
            STATS[0].fetch_add(1, Ordering::Relaxed);
            STATS[1].fetch_add(r.end - r.start, Ordering::Relaxed);
            bounds(&regions);
            n
        }
    };
    let region = &regions[i];
    let buffer = crate::inplace::place(device, &region.heap, at - region.start, end - at).map_err(|e| (6, format!("no buffer placed in the region: {e}")))?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    Ok(Placed { buffer, heap: region.heap.clone(), gpu, bias: start - at })
}

/// Opens the stretch around `[at, end)` (64 KiB aligned) as a heap through
/// a view of ours of the same pages.
fn open_region(device: &ID3D12Device, at: u64, end: u64) -> Result<Region, (usize, String)> {
    let Some((first, bytes, section, offset)) = stretch(at as usize, (end - at) as usize) else {
        return Err((3, format!("not one stretch of a section: {:?}", cover(at as usize, (end - at) as usize))));
    };
    // The whole stretch, kept to what may be ported: the next buffer in it
    // finds the region made.
    let (mut first, mut bytes, mut offset) = (first as u64, bytes as u64, offset);
    if first < LOW && first + bytes > LOW {
        bytes = LOW.saturating_sub(first);
    }
    if bytes > MOST {
        // Only what was asked for.
        offset += at - first;
        (first, bytes) = (at, end - at);
    }
    if bytes > MOST {
        return Err((3, format!("{bytes:#x} bytes, over the most of a region")));
    }
    if first > at || first + bytes < end || offset % K64 != 0 {
        return Err((3, format!("the stretch {first:#x}+{bytes:#x} does not hold it")));
    }
    // SAFETY: the section of a view the title has; the runtime keeps its
    // handle open while a view of it is there, and the range is inside it.
    let view = unsafe { mem::MapViewOfFile(section as _, mem::FILE_MAP_WRITE, (offset >> 32) as u32, offset as u32, bytes as usize) }.Value as usize;
    if view == 0 {
        return Err((5, format!("no view of the stretch: {}", std::io::Error::last_os_error())));
    }
    let unmap = || {
        // SAFETY: our own view, nothing opened on it.
        unsafe { mem::UnmapViewOfFile(mem::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    };
    if !committed(view, bytes as usize) {
        unmap();
        return Err((4, "pages of the stretch are not committed".to_owned()));
    }
    match crate::inplace::open(device, crate::inplace::From::Part(view, bytes as usize)) {
        Ok(heap) => {
            tracing::info!(
                start = format_args!("{first:#x}"),
                bytes,
                section = format_args!("{section:#x}"),
                offset = format_args!("{offset:#x}"),
                view = format_args!("{view:#x}"),
                "renderer port 1: a stretch of the title's graphics memory opened as a host heap through a view of ours"
            );
            // shortcut: the view stays mapped for good (the heap holds its
            // pages while anything names it).
            Ok(Region { start: first, end: first + bytes, heap })
        }
        Err(e) => {
            unmap();
            Err((5, format!("refused by the host: {e}")))
        }
    }
}

/// Whether every page of `[at, at + len)` is committed.
fn committed(at: usize, len: usize) -> bool {
    let mut here = at;
    while here < at + len {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { mem::VirtualQuery(here as *const _, &mut info, size_of_val(&info)) } == 0 || info.State != mem::MEM_COMMIT {
            return false;
        }
        here = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    }
    true
}

fn bounds(regions: &[Region]) {
    span_bounds(&BOUNDS, regions.iter().filter(|r| r.end <= LOW).map(|r| (r.start, r.end)));
    span_bounds(&REGIONS_BOUNDS, regions.iter().map(|r| (r.start, r.end)));
}

/// Forget placements and failed opens when the title changes their backing memory.
pub(crate) fn forget(at: usize, len: usize) {
    let (at, end) = (at as u64, (at as u64).saturating_add(len as u64));
    let hit_regions = overlaps(&REGIONS_BOUNDS, at, end);
    let hit_high = overlaps(&HIGH_BOUNDS, at, end);
    // Keep the same lock order as high placement.
    let mut spans = hit_high.then(|| HIGH_SPANS.write().unwrap_or_else(|e| e.into_inner()));
    if hit_regions || hit_high {
        let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
        let low_before = regions.iter().filter(|r| r.end <= LOW).count();
        regions.retain(|r| r.end <= at || r.start >= end);
        let low_after = regions.iter().filter(|r| r.end <= LOW).count();
        STATS[7].fetch_add((low_before - low_after) as u64, Ordering::Relaxed);
        bounds(&regions);
    }
    if let Some(spans) = spans.as_mut() {
        spans.retain(|&(start, stop)| stop <= at || start >= end);
        span_bounds(&HIGH_BOUNDS, spans.iter().copied());
    }
    drop(spans);

    if overlaps(&LATE_CB_BOUNDS, at, end) {
        let mut pages = LATE_CB_PAGES.write().unwrap_or_else(|e| e.into_inner());
        let mut freed = 0;
        let evicted: Vec<_> = pages
            .extract_if(|&(start, bytes), _| start < end && start + bytes > at)
            .filter_map(|((_, bytes), entry)| {
                let placed = entry?;
                freed += bytes;
                Some(placed)
            })
            .collect();
        LATE_CB_BYTES.fetch_sub(freed, Ordering::Relaxed);
        late_cb_bounds(&pages);
        drop(pages);
        if !evicted.is_empty() {
            crate::host::retire(evicted);
        }
    }
}

/// Whether `[at, at + len)` touches low (<4 GiB) ported memory (a quick look).
fn touches(at: usize, len: usize) -> bool {
    let hi = BOUNDS[1].load(Ordering::Relaxed);
    hi != 0 && (at as u64) < hi && (at as u64).saturating_add(len as u64) > BOUNDS[0].load(Ordering::Relaxed)
}

/// Whether `[at, at + len)` touches any high (>=4 GiB) in-place buffer.
fn touches_high(at: usize, len: usize) -> bool {
    let a = at as u64;
    let e = a.saturating_add(len as u64);
    if !overlaps(&HIGH_BOUNDS, a, e) {
        return false;
    }
    HIGH_SPANS
        .read()
        .unwrap_or_else(|err| err.into_inner())
        .iter()
        .any(|&(s, stop)| a < stop && e > s)
}

/// A fill of upload memory read `len` bytes at `src` on the CPU.
pub(crate) fn filled(src: usize, len: usize) {
    if touches(src, len) {
        STATS[9].fetch_add(1, Ordering::Relaxed);
        STATS[10].fetch_add(len as u64, Ordering::Relaxed);
    } else if touches_high(src, len) {
        HIGH_STATS[6].fetch_add(1, Ordering::Relaxed);
        HIGH_STATS[7].fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// A title buffer copy was done on the CPU.
pub(crate) fn copied(dst: usize, src: usize, len: usize) {
    if touches(dst, len) || touches(src, len) {
        STATS[11].fetch_add(1, Ordering::Relaxed);
        STATS[12].fetch_add(len as u64, Ordering::Relaxed);
    } else if touches_high(dst, len) || touches_high(src, len) {
        HIGH_STATS[8].fetch_add(1, Ordering::Relaxed);
        HIGH_STATS[9].fetch_add(len as u64, Ordering::Relaxed);
    }
}

/// For the frame-rate line: what is in place, and what of ours read or
/// wrote it on the CPU per frame.
pub(crate) fn log(frames: f64) {
    if on() {
        let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
        let per: Vec<f64> = STATS[9..].iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "renderer port 1 (VICTORY_PORT_CONSTANTS=1): {} regions of the title's graphics memory in place ({} MB), {} title buffers and windows placed in them \
             (no host copy, nothing sent up or fetched back for them); per frame on the CPU: {:.0} fills read out of them ({:.1} KB), {:.1} buffer copies \
             touching them ({:.1} KB); since the start refused: {} not one stretch of a section, {} not committed, {} by the host, {} no buffer; {} regions \
             forgotten (other pages mapped there)",
            s[0],
            s[1] >> 20,
            s[2],
            per[0],
            per[1] / 1024.0,
            per[2],
            per[3] / 1024.0,
            s[3],
            s[4],
            s[5],
            s[6],
            s[7],
        );
    }
    if high_on() || private_on() {
        let h: Vec<u64> = HIGH_STATS[..6].iter().map(|s| s.load(Ordering::Relaxed)).collect();
        let per: Vec<f64> = HIGH_STATS[6..].iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "renderer port high/private (HIGH={}, PRIVATE={}): {} mapped stretches ({} MB, {} refused), {} private buffers ({} MB, {} refused); \
             per frame on the CPU: {:.1} fills ({:.1} KB), {:.1} buffer copies ({:.1} KB)",
            u8::from(high_on()),
            u8::from(private_on()),
            h[0],
            h[1] >> 20,
            h[2],
            h[3],
            h[4] >> 20,
            h[5],
            per[0],
            per[1] / 1024.0,
            per[2],
            per[3] / 1024.0,
        );
    }
    let [pages, bytes, hits, failed] = LATE_CB_STATS.each_ref().map(|count| count.load(Ordering::Relaxed));
    if pages != 0 || failed != 0 {
        let cached_bytes = LATE_CB_BYTES.load(Ordering::Relaxed);
        tracing::info!(pages, bytes, cached_bytes, hits, failed, "renderer port late CB");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the title's 32-bit memory is asked for.
    #[test]
    fn only_memory_below_4_gib_is_wanted_with_the_switch() {
        // The switch is read once from the environment; tests do not set
        // it, so nothing is wanted.
        if !on() {
            assert!(!wanted(0x3431_0000, 0x1000));
        }
        let low = |at: u64, len: u64| at.checked_add(len).is_some_and(|end| end <= LOW && len > 0);
        assert!(low(0x34e1_0000, 0xae0_0000));
        assert!(!low(0xffff_0000, 0x2_0000));
        assert!(!low(0xe_1c80_0000, 0x1000));
        assert!(!low(u64::MAX, 2));
    }

    /// A stretch of a reserved section, mapped as `xmem` maps physical
    /// pages (views of consecutive pages in turn, committed), is opened
    /// through a view of ours: the card reads the bytes the title's views
    /// show, at the buffer's offset, and its writes show in them; a second
    /// buffer in the stretch is placed in the same region.
    #[test]
    fn warp_a_stretch_of_a_reserved_section_is_used_in_place_through_a_view_of_ours() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY (the whole test): the section, placeholder and views made
        // here.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_RESERVE,
                0,
                8 * K as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let hole = mem::VirtualAlloc2(process, std::ptr::null(), 8 * K, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0)
                .cast::<u8>();
            assert!(!hole.is_null());
            // Two views, pages 0..3 and 3..8 of the section, in turn.
            mem::VirtualFree(hole.cast(), 3 * K, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
            let mut views = Vec::new();
            for (page, n) in [(0usize, 3usize), (3, 5)] {
                let view = mem::MapViewOfFile3(section, process, hole.add(page * K).cast(), (page * K) as u64, n * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
                assert!(!view.Value.is_null(), "{}", std::io::Error::last_os_error());
                assert!(!mem::VirtualAlloc(view.Value, n * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
                crate::inplace::title_memory_view(view.Value as usize, n * K, section as usize, (page * K) as u64);
                views.push(view);
            }
            let base = hole as u64;
            let (start, len) = (base + K as u64 + 0x40, 5 * K as u64);
            let placed = place_in(&device, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x40);
            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            for (i, b) in std::slice::from_raw_parts_mut(start as *mut u8, len as usize).iter_mut().enumerate() {
                *b = (i * 7 + 3) as u8;
            }
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert!(got == std::slice::from_raw_parts(start as *const u8, len as usize), "the card reads the title's bytes, across both views");
            rig.store(&placed.buffer, placed.bias + 3 * K as u64, &[0x5a; 256]).unwrap();
            assert!(std::slice::from_raw_parts((start + 3 * K as u64) as *const u8, 256).iter().all(|&b| b == 0x5a), "the card's write shows in the title's view");
            // Another buffer in the same stretch: the same region.
            let regions = REGIONS.lock().unwrap().len();
            let second = place_in(&device, base + 6 * K as u64, 100).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(REGIONS.lock().unwrap().len(), regions, "one region for the stretch");
            assert_eq!(second.bias, 0);
            // The title's views may change under the heap.
            drop((placed, second));
            forget(base as usize, 8 * K);
            for view in views {
                assert!(mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0, "the title's own view can be unmapped");
            }
            crate::inplace::views_gone(base as usize, 8 * K);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            mem::VirtualFree(hole.add(3 * K).cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    #[test]
    fn forgetting_high_spans_only_invalidates_overlapping_ranges() {
        let _turn = crate::indirect::tests::warp_turn();
        let at = LOW + K64;
        let next = at + 2 * K64;
        {
            let mut spans = HIGH_SPANS.write().unwrap();
            note_high_span(&mut spans, at, at + K64);
            note_high_span(&mut spans, next, next + K64);
        }
        assert!(touches_high(at as usize, 1));
        assert!(!touches_high((at + K64) as usize, K64 as usize));
        forget((at + 1) as usize, 1);
        assert!(!touches_high(at as usize, 1));
        assert!(touches_high(next as usize, 1));
        forget(next as usize, K64 as usize);
        assert!(!touches_high(next as usize, 1));
    }

    #[test]
    fn warp_late_cb_opens_caches_and_forgets_private_pages() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        // SAFETY: this test owns the allocation. Windows rejects release
        // while the retirement queue still holds the heap.
        unsafe {
            let page = mem::VirtualAlloc(std::ptr::null(), K64 as usize, mem::MEM_RESERVE, mem::PAGE_READWRITE);
            assert!(!page.is_null());
            let at = page as u64;
            assert!(at >= LOW && at.is_multiple_of(K64));
            let address = at + 0x120;
            let failed = LATE_CB_STATS[3].load(Ordering::Relaxed);
            assert!(place_late_cb_in(&device, address, 32).is_none());
            assert!(place_late_cb_in(&device, address, 32).is_none());
            assert_eq!(LATE_CB_STATS[3].load(Ordering::Relaxed), failed + 1);
            forget(address as usize, 32);
            assert!(!LATE_CB_PAGES.read().unwrap().contains_key(&(at, K64)));

            assert_eq!(mem::VirtualAlloc(page, K64 as usize, mem::MEM_COMMIT, mem::PAGE_READWRITE), page);
            let before = LATE_CB_BYTES.load(Ordering::Relaxed);
            let (buffer, gpu) = place_late_cb_in(&device, address, 32).expect("committed private page");
            let base = buffer.GetGPUVirtualAddress();
            assert!(base.is_multiple_of(K64));
            assert_eq!(gpu, base + 0x120);
            assert_eq!(LATE_CB_BYTES.load(Ordering::Relaxed), before + K64);

            let hits = LATE_CB_STATS[2].load(Ordering::Relaxed);
            let (cached, cached_gpu) = place_late_cb_in(&device, address + 16, 16).unwrap();
            assert_eq!(cached.GetGPUVirtualAddress(), base);
            assert_eq!(cached_gpu, gpu + 16);
            assert_eq!(LATE_CB_STATS[2].load(Ordering::Relaxed), hits + 1);

            drop((buffer, cached));
            forget(address as usize, 32);
            assert_eq!(LATE_CB_BYTES.load(Ordering::Relaxed), before);
            assert!(!LATE_CB_PAGES.read().unwrap().contains_key(&(at, K64)));
            // The host's default retirement lag may keep this page pinned
            // until the test process exits.
            mem::VirtualFree(page, 0, mem::MEM_RELEASE);
        }
    }

    /// `place_high_in` opens only the buffer's exact 64 KiB pages (not the whole
    /// stretch) through our own view of the section, preserving full read/write
    /// coherency and leaving the title's views free to unmap.
    #[test]
    fn warp_high_buffer_opens_exact_slice_of_stretch_in_place() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY: the section, placeholder, and view belong to this test.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_RESERVE,
                0,
                10 * K as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let hole = mem::VirtualAlloc2(process, std::ptr::null(), 10 * K, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0)
                .cast::<u8>();
            assert!(!hole.is_null());
            let view = mem::MapViewOfFile3(section, process, hole.cast(), 0, 10 * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
            assert!(!view.Value.is_null());
            assert!(!mem::VirtualAlloc(view.Value, 10 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            crate::inplace::title_memory_view(view.Value as usize, 10 * K, section as usize, 0);

            // Ask for only 3 pages in the middle (pages 4..7) — must open only 3 pages, not all 10.
            let base = hole as u64;
            assert!(base >= LOW);
            let (start, len) = (base + 4 * K as u64 + 0x80, 2 * K as u64);
            let placed = place_high_in(&device, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x80);
            assert_eq!(placed.heap.GetDesc().SizeInBytes, 3 * K as u64);
            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            for (i, b) in std::slice::from_raw_parts_mut(start as *mut u8, len as usize).iter_mut().enumerate() {
                *b = (i * 13 + 5) as u8;
            }
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert_eq!(got, std::slice::from_raw_parts(start as *const u8, len as usize));
            rig.store(&placed.buffer, placed.bias + K as u64, &[0xa5; 128]).unwrap();
            assert!(std::slice::from_raw_parts((start + K as u64) as *const u8, 128).iter().all(|&b| b == 0xa5));

            drop(placed);
            let (at, end) = aligned_range(start, len).unwrap();
            let cached = place_high_region(&device, start, at, end).unwrap();
            assert!(REGIONS.lock().unwrap().iter().any(|r| r.start == at && r.end == end));
            drop(cached);
            forget(base as usize, 10 * K);
            assert!(!REGIONS.lock().unwrap().iter().any(|r| r.start == at && r.end == end));
            assert!(mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0);
            crate::inplace::views_gone(base as usize, 10 * K);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }
}
