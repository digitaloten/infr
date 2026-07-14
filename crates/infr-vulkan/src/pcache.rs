//! On-disk `vkPipelineCache` persistence. Pipeline creation (`vkCreateComputePipelines`) is the
//! driver's GPU-specific codegen and was measured at ~5s across a cold DG forward's kernel set —
//! all one-time work that previously re-ran every process launch (we passed
//! `vk::PipelineCache::null()` everywhere and relied incidentally on Mesa's own shader cache,
//! which softens but does not eliminate it). This module seeds ONE `vk::PipelineCache` from a
//! per-device file at backend init and writes it back (debounced, and finally on drop), so every
//! launch after the first creates pipelines from cached binaries.
//!
//! Invalidation is three-layer:
//! - The DRIVER validates its own header inside the blob (vendor/device/driverVersion/cacheUUID)
//!   and silently ignores data it can't use — so a driver upgrade never corrupts, at worst it
//!   recompiles once and the next save replaces the file.
//! - OUR envelope carries the build-time SHADER_SET_FINGERPRINT (FNV-1a over every compiled
//!   SPIR-V blob — see build.rs): any shader change discards the old file WHOLESALE instead of
//!   letting entries for retired pipeline variants accumulate in the blob forever.
//! - OUR envelope also carries an FNV-1a CHECKSUM of the payload, verified on load. What lands in
//!   this file is driver-authored machine code that we hand straight back to the driver, and
//!   `vkCreatePipelineCache`'s contract is explicit that invalid data is allowed to produce
//!   UNDEFINED BEHAVIOR — on a GPU that means a hung ring, not a clean error. A truncated or
//!   bit-rotted file must therefore die HERE, at a cheap one-time recompile, rather than reach
//!   the driver. (Mesa/RADV happens to hash its own cache objects and drop the ones that don't
//!   validate — measured: a blob with 25% of its payload bytes scrambled still ran correctly —
//!   so this layer is defense-in-depth against a less careful driver, not a load-bearing fix for
//!   any failure observed on RADV.)
//!
//! Writes are atomic AND durable: the payload is written to a per-pid temp file, `fsync`ed, then
//! `rename`d over the target, and the directory entry is `fsync`ed too. `rename` alone is only
//! atomic with respect to a concurrent READER — on a crash/power-loss it can leave the new name
//! pointing at an inode whose data blocks were never flushed (ext4 delayed allocation), i.e. a
//! valid-looking file over garbage. The checksum above would catch that; the fsync keeps it from
//! happening at all.
//!
//! `INFR_NO_PIPELINE_CACHE=1` disables persistence (the in-process cache handle still works).

use ash::vk;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

include!(concat!(env!("OUT_DIR"), "/shader_fingerprint.rs"));

/// Envelope version. Bumped from `INFRVPC1` when the payload checksum was added — an old file has
/// no checksum field, and its `1` magic makes `load` reject it outright (one free recompile).
const MAGIC: &[u8; 8] = b"INFRVPC2";
/// MAGIC(8) + fingerprint(8) + driver_version(4) + pipelineCacheUUID(16) + payload_len(8) +
/// payload_hash(8).
const HEADER_LEN: usize = 52;
/// Debounce for mid-run saves: pipeline creation comes in bursts (warmup, a new arch's first
/// forward) — one save per burst-second is plenty, and the final Drop save catches the tail.
const SAVE_DEBOUNCE_SECS: u64 = 1;

/// FNV-1a over the blob — the same hash build.rs uses for `SHADER_SET_FINGERPRINT`. Not a
/// cryptographic checksum and not meant to be one: it guards against truncation/bit-rot on a file
/// only this process family writes, not against an adversary who can already write to `$HOME`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Handle for one device's persisted pipeline cache: where it lives on disk and when it was last
/// written. `None`-able at the call sites (env-disabled or no writable cache dir).
pub(crate) struct PcachePersist {
    path: PathBuf,
    /// Driver version folded into the envelope alongside the shader-set fingerprint: the driver
    /// already ignores stale blobs itself, but a version flip also means retired entries would
    /// sit in the file forever — treat it like a shader-set change and start fresh.
    driver_version: u32,
    /// `VkPhysicalDeviceProperties::pipelineCacheUUID` — the driver's OWN identity for "binaries
    /// I can consume". The file is already keyed per (vendor_id, device_id) by NAME, so one GPU
    /// never reads another's blob on a multi-GPU box, and the driver re-checks this same UUID in
    /// the header it embeds inside the payload. Carrying it in OUR envelope too closes the one
    /// gap those leave: a driver REBUILD (or a distro rebuild) can keep `driver_version` while
    /// changing the cache UUID, and the reward for guessing wrong is a driver handed binaries it
    /// considers valid-ish — undefined behavior, i.e. a hung ring, not an error. Cheap to check,
    /// so check it.
    cache_uuid: [u8; 16],
    last_save: Mutex<Instant>,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl PcachePersist {
    /// `~/.cache/infr/vk-pipeline-cache-{vendor:08x}-{device:08x}.bin` (XDG-aware) — keyed per
    /// device so a multi-GPU box never clobbers one GPU's cache with another's.
    pub(crate) fn new(props: &vk::PhysicalDeviceProperties) -> Option<Self> {
        if std::env::var_os("INFR_NO_PIPELINE_CACHE").is_some() {
            return None;
        }
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        let dir = base.join("infr");
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self {
            path: dir.join(format!(
                "vk-pipeline-cache-{:08x}-{:08x}.bin",
                props.vendor_id, props.device_id
            )),
            driver_version: props.driver_version,
            cache_uuid: props.pipeline_cache_uuid,
            last_save: Mutex::new(Instant::now()),
        })
    }

    /// Read + validate the persisted blob. Any mismatch (magic, fingerprint, driver version,
    /// truncation, or a payload that fails its checksum) returns `None` — the stale/damaged file
    /// is simply replaced by the next save, at the cost of one cold pipeline build.
    pub(crate) fn load(&self) -> Option<Vec<u8>> {
        let data = std::fs::read(&self.path).ok()?;
        if data.len() < HEADER_LEN || &data[..8] != MAGIC {
            return None;
        }
        let fp = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let drv = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let uuid: [u8; 16] = data[20..36].try_into().unwrap();
        let len = u64::from_le_bytes(data[36..44].try_into().unwrap()) as usize;
        let sum = u64::from_le_bytes(data[44..52].try_into().unwrap());
        if fp != SHADER_SET_FINGERPRINT
            || drv != self.driver_version
            || uuid != self.cache_uuid
            || data.len() != HEADER_LEN + len
        {
            return None;
        }
        let payload = &data[HEADER_LEN..];
        if fnv1a(payload) != sum {
            // Damaged file: never hand it to `vkCreatePipelineCache` (invalid cache data is
            // explicitly undefined behavior, and on a GPU that reads as a hung ring rather than
            // an error). Drop it and let this launch rebuild.
            eprintln!(
                "[infr] pipeline cache {} failed its checksum — discarding and rebuilding",
                self.path.display()
            );
            let _ = std::fs::remove_file(&self.path);
            return None;
        }
        Some(payload.to_vec())
    }

    /// Serialize the live cache to disk atomically AND durably: write the temp file, `fsync` it,
    /// `rename` it over the target, then `fsync` the directory entry. See the module doc for why
    /// the plain `write` + `rename` this replaces was not enough (rename is atomic for a reader,
    /// but on an unclean shutdown it can publish a name over unflushed data blocks).
    pub(crate) fn save(&self, device: &ash::Device, cache: vk::PipelineCache) {
        if cache == vk::PipelineCache::null() {
            return;
        }
        let Ok(blob) = (unsafe { device.get_pipeline_cache_data(cache) }) else {
            return;
        };
        if blob.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(HEADER_LEN + blob.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&SHADER_SET_FINGERPRINT.to_le_bytes());
        out.extend_from_slice(&self.driver_version.to_le_bytes());
        out.extend_from_slice(&self.cache_uuid);
        out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        out.extend_from_slice(&fnv1a(&blob).to_le_bytes());
        out.extend_from_slice(&blob);
        let tmp = self
            .path
            .with_extension(format!("tmp.{}", std::process::id()));
        if write_durable(&tmp, &out).is_ok() && std::fs::rename(&tmp, &self.path).is_ok() {
            // The rename itself is a directory metadata change: sync the directory so the new
            // entry survives a crash too (the payload it points at is already on disk).
            if let Some(dir) = self.path.parent() {
                if let Ok(d) = std::fs::File::open(dir) {
                    let _ = d.sync_all();
                }
            }
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
        *self.last_save.lock().unwrap() = Instant::now();
    }

    /// Debounced save for mid-run persistence (called after each NEW pipeline lands) — covers
    /// long-lived processes that never Drop cleanly (serve under SIGKILL).
    pub(crate) fn maybe_save(&self, device: &ash::Device, cache: vk::PipelineCache) {
        {
            let last = self.last_save.lock().unwrap();
            if last.elapsed().as_secs() < SAVE_DEBOUNCE_SECS {
                return;
            }
        }
        self.save(device, cache);
    }
}

/// `fs::write` + `fsync`: the bytes are on the platter before the caller renames over the target.
fn write_durable(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The envelope round-trips, and every way a file can be damaged (bad magic, a shader-set /
    /// driver flip, truncation, a flipped payload byte) is REJECTED rather than handed to
    /// `vkCreatePipelineCache` — where invalid data is undefined behavior, i.e. a hung GPU.
    #[test]
    fn envelope_rejects_damage() {
        let dir = std::env::temp_dir().join(format!("infr-pcache-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        const UUID: [u8; 16] = [9u8; 16];
        let p = PcachePersist {
            path: path.clone(),
            driver_version: 7,
            cache_uuid: UUID,
            last_save: Mutex::new(Instant::now()),
        };
        let payload: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let envelope = |fp: u64, drv: u32, uuid: [u8; 16], sum: u64, body: &[u8]| {
            let mut out = Vec::new();
            out.extend_from_slice(MAGIC);
            out.extend_from_slice(&fp.to_le_bytes());
            out.extend_from_slice(&drv.to_le_bytes());
            out.extend_from_slice(&uuid);
            out.extend_from_slice(&(body.len() as u64).to_le_bytes());
            out.extend_from_slice(&sum.to_le_bytes());
            out.extend_from_slice(body);
            out
        };
        let good = envelope(SHADER_SET_FINGERPRINT, 7, UUID, fnv1a(&payload), &payload);

        std::fs::write(&path, &good).unwrap();
        assert_eq!(p.load().as_deref(), Some(&payload[..]), "good blob loads");

        // A single flipped payload byte must fail the checksum (and the file is removed).
        let mut rot = good.clone();
        rot[HEADER_LEN + 100] ^= 0x01;
        std::fs::write(&path, &rot).unwrap();
        assert!(p.load().is_none(), "bit-rotted payload must be rejected");
        assert!(!path.exists(), "a damaged cache file is deleted, not kept");

        // Truncation (the tail never reached disk).
        std::fs::write(&path, &good[..good.len() - 64]).unwrap();
        assert!(p.load().is_none(), "truncated blob must be rejected");

        // Wrong shader set / wrong driver.
        std::fs::write(
            &path,
            envelope(
                SHADER_SET_FINGERPRINT ^ 1,
                7,
                UUID,
                fnv1a(&payload),
                &payload,
            ),
        )
        .unwrap();
        assert!(p.load().is_none(), "stale shader set must be rejected");
        std::fs::write(
            &path,
            envelope(SHADER_SET_FINGERPRINT, 8, UUID, fnv1a(&payload), &payload),
        )
        .unwrap();
        assert!(p.load().is_none(), "driver-version flip must be rejected");

        // A blob from a driver that reports the SAME version but a different cache UUID (a driver
        // rebuild) — and, by the same check, any blob whose binaries this driver did not author.
        std::fs::write(
            &path,
            envelope(
                SHADER_SET_FINGERPRINT,
                7,
                [1u8; 16],
                fnv1a(&payload),
                &payload,
            ),
        )
        .unwrap();
        assert!(
            p.load().is_none(),
            "foreign pipelineCacheUUID must be rejected"
        );

        // A v1 (checksum-less) file from an older build.
        let mut v1 = good.clone();
        v1[..8].copy_from_slice(b"INFRVPC1");
        std::fs::write(&path, &v1).unwrap();
        assert!(p.load().is_none(), "old envelope version must be rejected");

        std::fs::remove_dir_all(&dir).ok();
    }
}
