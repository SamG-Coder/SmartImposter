//! Impostor atlas residency: the baked, BC-compressed atlases live on disk (assets/cache/impostors,
//! written the first time a variant is baked), every variant keeps a small *base* atlas on the GPU
//! (its levels from 1/4 resolution down: ~1/16 of the memory), and only the variants growing near
//! the camera get their *full* atlas, read on a loader thread and swapped in (renderer.rs
//! update_atlases). Far impostors sample the low mips anyway, so the base atlas looks the same
//! there; the full levels are only seen within a few hundred metres.
//!
//! A cache file is the bake's BC block stream for one variant: per level 0..mips the albedo
//! (BC3), normal (BC5) and depth (BC5) atlases, then the half-resolution visibility (BC3) levels.

use anyhow::{Context, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const DIR: &str = "assets/cache/impostors";
const MAGIC: &[u8; 8] = b"SIIMP\0\0\x02";
const HEADER: u64 = 8 + 8 + 4 * 4;

/// Levels the base atlas drops (1/4 resolution: 1/16 of the texels).
pub const BASE_SKIP: u32 = 2;

/// Base atlas levels skipped for an atlas with `mips` levels (keeps at least 2 levels).
pub fn skip_for(mips: u32) -> u32 {
    BASE_SKIP.min(mips.saturating_sub(2))
}

/// One BC block region of the stream: image (0 albedo, 1 normal, 2 depth, 3 visibility), its
/// level in the full atlas, texels per side, byte offset and length in the stream.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub img: usize,
    pub level: u32,
    pub size: u32,
    pub offset: u64,
    pub bytes: u64,
}

/// The stream layout of an atlas of `grid` x `grid` frames of `res` texels with `mips` levels
/// (the order renderer.rs bake_impostors encodes in).
pub fn layout(grid: u32, res: u32, mips: u32) -> Vec<Region> {
    let size = grid * res;
    let mut out = Vec::new();
    let mut offset = 0u64;
    let mut push = |img: usize, src_level: u32, level: u32| {
        let s = size >> src_level;
        let bytes = (s as u64 / 4) * (s as u64 / 4) * 16;
        out.push(Region { img, level, size: s, offset, bytes });
        offset += bytes;
    };
    for level in 0..mips {
        for img in 0..3 {
            push(img, level, level);
        }
    }
    // visibility: half resolution, so its level L comes from scratch level L + 1
    for level in 1..mips {
        push(3, level, level - 1);
    }
    out
}

pub fn stream_bytes(grid: u32, res: u32, mips: u32) -> u64 {
    layout(grid, res, mips).iter().map(|r| r.bytes).sum()
}

/// The cache file of a bake: its name and key (variants may share a name; a changed bake gets a
/// new file, see `prune`).
pub fn path_for(name: &str, key: u64) -> PathBuf {
    let clean: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    Path::new(DIR).join(format!("{clean}-{key:016x}.bc"))
}

/// Remove cache files no current bake uses (older keys, the old name-only files).
pub fn prune(keep: &[PathBuf]) {
    let keep: std::collections::HashSet<&Path> = keep.iter().map(|p| p.as_path()).collect();
    let Ok(dir) = std::fs::read_dir(DIR) else { return };
    for e in dir.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "bc") && !keep.contains(p.as_path()) {
            let _ = std::fs::remove_file(&p);
        }
    }
}

/// FNV-1a over everything that changes a bake.
pub struct Key(u64);

impl Key {
    pub fn new() -> Self {
        Key(0xcbf2_9ce4_8422_2325)
    }
    pub fn bytes(mut self, b: &[u8]) -> Self {
        for x in b {
            self.0 = (self.0 ^ *x as u64).wrapping_mul(0x100_0000_01b3);
        }
        self
    }
    pub fn u64(self, v: u64) -> Self {
        self.bytes(&v.to_le_bytes())
    }
    pub fn get(&self) -> u64 {
        self.0
    }
}

/// Does `path` hold a bake with this key and shape?
pub fn valid(path: &Path, key: u64, grid: u32, res: u32, mips: u32) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let mut h = [0u8; HEADER as usize];
    if f.read_exact(&mut h).is_err() || &h[..8] != MAGIC {
        return false;
    }
    let u = |i: usize| u32::from_le_bytes(h[i..i + 4].try_into().unwrap());
    let len = u(28) as u64;
    u64::from_le_bytes(h[8..16].try_into().unwrap()) == key
        && (u(16), u(20), u(24)) == (grid, res, mips)
        && len == stream_bytes(grid, res, mips)
        && f.metadata().map(|m| m.len() == HEADER + len).unwrap_or(false)
}

pub fn write(path: &Path, key: u64, grid: u32, res: u32, mips: u32, stream: &[u8]) -> Result<()> {
    std::fs::create_dir_all(DIR)?;
    let tmp = path.with_extension("tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(MAGIC)?;
    f.write_all(&key.to_le_bytes())?;
    for v in [grid, res, mips, stream.len() as u32] {
        f.write_all(&v.to_le_bytes())?;
    }
    f.write_all(stream)?;
    f.into_inner()?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// The given regions of a cache file, concatenated in order.
pub fn read_regions(path: &Path, regions: &[Region]) -> Result<Vec<u8>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut out = vec![0u8; regions.iter().map(|r| r.bytes as usize).sum()];
    let mut at = 0usize;
    for r in regions {
        f.seek(SeekFrom::Start(HEADER + r.offset))?;
        f.read_exact(&mut out[at..at + r.bytes as usize])?;
        at += r.bytes as usize;
    }
    Ok(out)
}

/// Where the loader writes a stream: an address in mapped upload memory and its length.
/// (The renderer keeps the region reserved until the GPU has copied it out.)
#[derive(Clone, Copy)]
pub struct Dest {
    pub addr: usize,
    pub len: usize,
}

/// Full-atlas reads off the render thread, straight into the renderer's upload buffer:
/// requests (variant, cache file, destination) in, (variant, result) out.
pub struct Loader {
    tx: std::sync::mpsc::Sender<(usize, PathBuf, Dest)>,
    pub rx: std::sync::mpsc::Receiver<(usize, Result<()>)>,
}

impl Loader {
    pub fn spawn() -> Self {
        let (tx, req) = std::sync::mpsc::channel::<(usize, PathBuf, Dest)>();
        let (done, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("atlas loader".into())
            .spawn(move || {
                for (v, path, dest) in req {
                    // SAFETY: the renderer reserved [addr, addr + len) of its persistently mapped
                    // upload buffer for this request and touches it again only after our reply.
                    let out = unsafe { std::slice::from_raw_parts_mut(dest.addr as *mut u8, dest.len) };
                    let r = (|| -> Result<()> {
                        let mut f = std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
                        f.seek(SeekFrom::Start(HEADER))?;
                        f.read_exact(out).with_context(|| format!("{} is truncated", path.display()))?;
                        Ok(())
                    })();
                    if done.send((v, r)).is_err() {
                        break;
                    }
                }
            })
            .expect("atlas loader thread");
        Loader { tx, rx }
    }

    pub fn request(&self, v: usize, path: PathBuf, dest: Dest) {
        let _ = self.tx.send((v, path, dest));
    }
}

/// First-fit allocator of byte ranges in the upload buffer (free ranges sorted, merged on release).
pub struct Ranges {
    free: Vec<(u64, u64)>,
}

impl Ranges {
    pub fn new(size: u64) -> Self {
        Ranges { free: vec![(0, size)] }
    }
    /// A range of `n` bytes starting at a multiple of 16 (buffer-to-image copy offsets must be
    /// multiples of the 16-byte BC block).
    pub fn alloc(&mut self, n: u64) -> Option<u64> {
        let n = n.div_ceil(16) * 16;
        let i = self.free.iter().position(|r| r.1 >= n)?;
        let start = self.free[i].0;
        self.free[i].0 += n;
        self.free[i].1 -= n;
        if self.free[i].1 == 0 {
            self.free.remove(i);
        }
        Some(start)
    }
    pub fn release(&mut self, start: u64, n: u64) {
        let n = n.div_ceil(16) * 16;
        let i = self.free.partition_point(|r| r.0 < start);
        self.free.insert(i, (start, n));
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_the_bake_order() {
        // 12 frames of 32 texels, 3 levels: 3 x 3 colour regions then 2 visibility levels
        let l = layout(12, 32, 3);
        assert_eq!(l.len(), 3 * 3 + 2);
        assert_eq!((l[0].img, l[0].level, l[0].size, l[0].offset), (0, 0, 384, 0));
        assert_eq!(l[3].size, 192);
        let vis: Vec<_> = l.iter().filter(|r| r.img == 3).map(|r| (r.level, r.size)).collect();
        assert_eq!(vis, vec![(0, 192), (1, 96)]);
        assert_eq!(stream_bytes(12, 32, 3), l.iter().map(|r| r.bytes).sum::<u64>());
        assert_eq!(skip_for(3), 1);
        assert_eq!(skip_for(7), 2);
    }

    #[test]
    fn ranges_reuse_and_merge() {
        let mut r = Ranges::new(1000);
        let a = r.alloc(100).unwrap();
        let b = r.alloc(250).unwrap();
        assert_eq!((a, b), (0, 112)); // 16-byte aligned
        assert!(r.alloc(1000).is_none());
        r.release(a, 100);
        r.release(b, 250);
        assert_eq!(r.alloc(990), Some(0)); // merged back into one range
    }

    #[test]
    fn cache_round_trip() {
        let p = std::env::temp_dir().join("si_atlas_test.bc");
        let n = stream_bytes(12, 32, 3) as usize;
        let data: Vec<u8> = (0..n).map(|i| (i * 7 % 251) as u8).collect();
        let tmpdir_write = |path: &Path| -> Result<()> {
            let mut f = std::fs::File::create(path)?;
            f.write_all(MAGIC)?;
            f.write_all(&42u64.to_le_bytes())?;
            for v in [12u32, 32, 3, n as u32] {
                f.write_all(&v.to_le_bytes())?;
            }
            f.write_all(&data)?;
            Ok(())
        };
        tmpdir_write(&p).unwrap();
        assert!(valid(&p, 42, 12, 32, 3));
        assert!(!valid(&p, 43, 12, 32, 3));
        let l = layout(12, 32, 3);
        let got = read_regions(&p, &l[3..5]).unwrap();
        let (a, b) = (l[3].offset as usize, (l[4].offset + l[4].bytes) as usize);
        assert_eq!(got, data[a..b]);
    }
}
