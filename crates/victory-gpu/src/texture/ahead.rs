//! First uploads of indexed textures on a dedicated D3D12 copy queue.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::Staging;
use crate::object::This;

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPLOAD_AHEAD").is_none_or(|v| v != "0"))
}

/// The largest initialized mip, and who may record GPU work on this copy.
pub(crate) struct Upload {
    first: AtomicU32,
    use_by: Mutex<Use>,
    landed: Condvar,
}

enum Use {
    Fresh,
    Copy(Arc<Job>),
    Graphics,
}

impl Default for Upload {
    fn default() -> Self {
        Self {
            first: AtomicU32::new(u32::MAX),
            use_by: Mutex::new(Use::Fresh),
            landed: Condvar::new(),
        }
    }
}

impl Upload {
    pub(crate) fn ready(&self, first: u32) -> bool {
        self.first.load(Ordering::Acquire) <= first
    }

    pub(crate) fn complete(&self, first: u32) {
        self.first.fetch_min(first, Ordering::Release);
    }

    pub(crate) fn copying(&self) -> bool {
        matches!(
            *self.use_by.lock().unwrap_or_else(|e| e.into_inner()),
            Use::Copy(_)
        )
    }

    /// Publishing a resource without a graphics fence rules out any later
    /// copy-queue claim. An existing copy job still owns publication.
    pub(crate) fn expose(&self) -> bool {
        let mut state = self.use_by.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*state, Use::Copy(_)) {
            return false;
        }
        *state = Use::Graphics;
        true
    }

    /// Claim graphics use before recording either a copy or a transition.
    /// An accepted copy job owns the resource until its fence completes.
    pub(crate) fn listed(&self) {
        if !on() {
            return;
        }
        let state = self.use_by.lock().unwrap_or_else(|e| e.into_inner());
        let mut state = self
            .landed
            .wait_while(state, |state| matches!(state, Use::Copy(_)))
            .unwrap_or_else(|e| e.into_inner());
        *state = Use::Graphics;
    }
}

type Callback = Box<dyn FnOnce(Option<u64>) + Send>;

enum Completion {
    Pending(Vec<Callback>),
    Done(Option<u64>),
}

pub(crate) struct Job {
    pub(crate) bytes: u64,
    pub(crate) memory: (u64, u64),
    first: u32,
    completion: Mutex<Completion>,
}

impl Job {
    /// Deliver the fingerprint of the uploaded bytes after the fence.
    /// A failed job delivers `None` so its slots can retry.
    pub(crate) fn then(&self, callback: impl FnOnce(Option<u64>) + Send + 'static) {
        let mut completion = self.completion.lock().unwrap_or_else(|e| e.into_inner());
        match &mut *completion {
            Completion::Pending(waiting) => waiting.push(Box::new(callback)),
            Completion::Done(result) => {
                let result = *result;
                drop(completion);
                callback(result);
            }
        }
    }

    fn finish(&self, result: Option<u64>) {
        let old = {
            let mut completion = self.completion.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::replace(&mut *completion, Completion::Done(result))
        };
        if let Completion::Pending(waiting) = old {
            for callback in waiting {
                callback(result);
            }
        }
    }
}

static USED: AtomicU64 = AtomicU64::new(0);

pub(super) struct Reservation {
    used: &'static AtomicU64,
    bytes: u64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

fn reserve_from(used: &'static AtomicU64, bytes: u64, limit: u64) -> Option<Reservation> {
    used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
        used.checked_add(bytes).filter(|&total| total <= limit)
    })
    .ok()?;
    Some(Reservation { used, bytes })
}

pub(super) fn reserve(bytes: u64) -> Option<Reservation> {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        let dedicated = crate::host::host()
            .and_then(crate::host::Host::dedicated_video_memory)
            .unwrap_or(0);
        let mib = std::env::var("VICTORY_UPLOAD_AHEAD_MB")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(if dedicated <= 9 << 30 { 128 } else { 256 });
        mib.saturating_mul(1 << 20)
    });
    reserve_from(&USED, bytes, limit)
}

struct Work {
    job: Arc<Job>,
    staging: Arc<Staging>,
    resident: crate::resource::ResidentTexture,
}

/// Take a fresh copy, or join its existing job if that job covers the requested mips.
pub(crate) fn take(resource: This, first: u32) -> Option<(Arc<Job>, bool)> {
    if !on()
        || !crate::resource::title_fills(resource)
        || super::host_wrote(resource as usize)
        || crate::resource::alias_of(resource).is_some()
    {
        return None;
    }
    let resident = crate::resource::resident_copy(resource)?;
    let copy = resident._gpu.clone();
    let mut state = copy.upload.use_by.lock().unwrap_or_else(|e| e.into_inner());
    match &*state {
        Use::Copy(job) => return (job.first <= first).then(|| (job.clone(), false)),
        Use::Graphics => return None,
        Use::Fresh => {}
    }
    if copy.state.load(Ordering::Relaxed) != D3D12_RESOURCE_STATE_COMMON.0 as u32
        || crate::resource::ahead_throttled()
    {
        return None;
    }
    let queue = worker()?;
    let staging = super::make_staging(resource, &copy.texture, first, true)?;
    let job = Arc::new(Job {
        bytes: staging.bytes,
        memory: (staging.memory as u64, staging.size as u64),
        first: staging.first,
        completion: Mutex::new(Completion::Pending(Vec::new())),
    });
    *state = Use::Copy(job.clone());
    // The state and submission are published together, so a graphics user
    // can never wait on a job that a caller has yet to enqueue.
    let work = Work {
        job: job.clone(),
        staging,
        resident,
    };
    if queue.send(work).is_err() {
        *state = Use::Graphics;
        return None;
    }
    Some((job, true))
}

fn worker() -> Option<&'static mpsc::Sender<Work>> {
    static WORKER: OnceLock<Option<mpsc::Sender<Work>>> = OnceLock::new();
    WORKER
        .get_or_init(|| {
            let host = crate::host::host()?;
            let copy = CopyList::new(&host.device).ok()?;
            let (send, receive) = mpsc::channel();
            std::thread::Builder::new()
                .name("indexed texture uploads".into())
                .spawn(move || work(copy, &receive))
                .ok()?;
            Some(send)
        })
        .as_ref()
}

fn work(mut copy: CopyList, receive: &mpsc::Receiver<Work>) {
    while let Ok(first) = receive.recv() {
        let mut bytes = first.job.bytes;
        let mut batch = vec![first];
        while bytes < 16 << 20 && batch.len() < 64 {
            let Ok(next) = receive.try_recv() else { break };
            bytes = bytes.saturating_add(next.job.bytes);
            batch.push(next);
        }
        // Reading through the kernel avoids waiting for a guest remap
        // while a graphics recorder is itself waiting on this job.
        let batch: Vec<_> = batch
            .into_iter()
            .map(|work| {
                let mut tiled = vec![0; work.staging.size];
                let print = crate::host::read_into(work.staging.memory, &mut tiled).then(|| {
                    work.staging.detile(&tiled);
                    super::fingerprint_from(tiled.len(), |hasher, from, bytes| {
                        hasher.update(&tiled[from..from + bytes]);
                    })
                });
                (work, print)
            })
            .collect();
        let ready = || batch.iter().filter(|(_, print)| print.is_some());
        let mut ok = false;
        if ready().next().is_some() {
            let result = copy.run(|list| {
                for (work, _) in ready() {
                    let staging = &work.staging;
                    for (i, footprint) in staging.footprints.iter().enumerate() {
                        record(
                            list,
                            &work.resident._gpu.texture,
                            staging.subresource(i),
                            &staging.buffer,
                            *footprint,
                        );
                    }
                }
            });
            if let Err(e) = &result {
                tracing::warn!(error = %e, "indexed texture uploads did not complete");
            }
            ok = result.is_ok();
        }
        finish(batch, ok);
    }
}

fn finish(batch: Vec<(Work, Option<u64>)>, ok: bool) {
    let mut jobs = Vec::with_capacity(batch.len());
    for (
        Work {
            job,
            staging,
            resident,
        },
        print,
    ) in batch
    {
        let result = if ok { print } else { None };
        let upload = &resident._gpu.upload;
        {
            let mut state = upload.use_by.lock().unwrap_or_else(|e| e.into_inner());
            if result.is_some() {
                upload.complete(job.first);
            }
            drop(staging);
            *state = Use::Graphics;
            upload.landed.notify_all();
        }
        jobs.push((job, resident, result));
    }
    // Release all graphics waiters before callbacks take any descriptor locks.
    for (job, _resident, result) in jobs {
        job.finish(result);
    }
}

pub(crate) struct CopyList {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
}

// SAFETY: the recording objects are used by a single worker. The device,
// queue and fence are free-threaded.
unsafe impl Send for CopyList {}

impl CopyList {
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let kind = D3D12_COMMAND_LIST_TYPE_COPY;
        // SAFETY: plain creation on a live device.
        unsafe {
            let queue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: kind,
                ..Default::default()
            })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(kind)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, kind, &allocator, None)?;
            list.Close()?;
            let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            Ok(Self {
                device: device.clone(),
                queue,
                allocator,
                list,
                fence,
                value: 0,
            })
        }
    }

    pub(crate) fn run(
        &mut self,
        fill: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> windows::core::Result<()> {
        // SAFETY: this list and allocator belong to the current recorder.
        // The caller holds all copy resources until the fence wait returns.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            fill(&self.list);
            self.list.Close()?;
            let list = self.list.cast()?;
            self.value += 1;
            self.queue.ExecuteCommandLists(&[Some(list)]);
            // After ExecuteCommandLists the resources and allocator have
            // to stay alive until a fence or device removal settles it.
            while let Err(error) = self.queue.Signal(&self.fence, self.value) {
                self.device.GetDeviceRemovedReason()?;
                tracing::warn!(%error, "retrying copy queue fence signal");
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            if self
                .fence
                .SetEventOnCompletion(self.value, HANDLE::default())
                .is_err()
            {
                while self.fence.GetCompletedValue() < self.value {
                    self.device.GetDeviceRemovedReason()?;
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            self.device.GetDeviceRemovedReason()
        }
    }
}

/// Record a buffer upload to a COMMON texture. A copy queue implicitly
/// promotes the destination and decays it back to COMMON after execution.
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    subresource: u32,
    buffer: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let to = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(Some(texture.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: subresource,
        },
    };
    let from = D3D12_TEXTURE_COPY_LOCATION {
        pResource: std::mem::ManuallyDrop::new(Some(buffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    };
    // SAFETY: both resources and the open copy list are live. The
    // locations borrow the resources for recording only.
    unsafe {
        let (mut to, mut from) = (to, from);
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        std::mem::ManuallyDrop::drop(&mut to.pResource);
        std::mem::ManuallyDrop::drop(&mut from.pResource);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_upload_ahead_respects_the_staging_budget() {
        static USED: AtomicU64 = AtomicU64::new(0);
        let older = reserve_from(&USED, 60, 128).unwrap();
        assert!(reserve_from(&USED, 69, 128).is_none());
        assert_eq!(USED.load(Ordering::Relaxed), 60);
        let newer = reserve_from(&USED, 68, 128).unwrap();
        assert!(reserve_from(&USED, 1, 128).is_none());
        drop(older);
        let fallback_can_retry = reserve_from(&USED, 60, 128).unwrap();
        drop((newer, fallback_can_retry));
        assert_eq!(USED.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn texture_upload_ahead_fences_first_upload_and_runs_then() {
        let Some(host) = crate::host::host() else {
            eprintln!("copy queue test skipped because no D3D12 device was available");
            return;
        };
        if !on() {
            return;
        }
        let image = super::super::TestTexture::new(0x37);
        let copy = crate::resource::host_copy(image.resource).unwrap();
        assert!(!copy.upload.ready(0));
        let (job, new) = take(image.resource, 0).expect("fresh upload accepted");
        assert!(new);
        let (send, receive) = mpsc::channel();
        let completed = copy.clone();
        job.then(move |result| {
            send.send(result.is_some() && completed.upload.ready(0))
                .unwrap()
        });
        assert!(
            receive
                .recv_timeout(std::time::Duration::from_secs(20))
                .unwrap()
        );
        let (send, receive) = mpsc::channel();
        let expected = super::super::fingerprint(job.memory.0, job.memory.1);
        job.then(move |result| send.send(result).unwrap());
        assert_eq!(receive.try_recv(), Ok(Some(expected)));
        assert!(take(image.resource, 0).is_none());
        assert_eq!(copy.state.load(Ordering::Relaxed), 0);

        // Read actual texels through a graphics list after the copy fence.
        const PITCH: usize = 256;
        const ROWS: usize = 4;
        let list = host.new_list().unwrap();
        let readback = host.read_back_buffer((PITCH * ROWS) as u64).unwrap();
        crate::list::ensure_state(
            &list,
            image.resource,
            D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32,
        );
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: the location borrows this live resource.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: the location borrows this live resource.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format:
                            windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16G16B16A16_FLOAT,
                        Width: 16,
                        Height: ROWS as u32,
                        Depth: 1,
                        RowPitch: PITCH as u32,
                    },
                },
            },
        };
        // SAFETY: the copy targets the live readback buffer. The mapping
        // spans that buffer and is read only after the submission finishes.
        unsafe {
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.close();
            host.execute(&[&list]);
            let mut data = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut data)).unwrap();
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), PITCH * ROWS);
            let matches = bytes
                .chunks_exact(PITCH)
                .all(|row| row[..128] == [0x37; 128]);
            readback.Unmap(0, None);
            assert!(matches);
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }
}
