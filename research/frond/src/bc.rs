//! The data being compressed: SmartImposter's impostor atlas cache files (`src/atlas.rs` of the
//! renderer) and the BC blocks in them.
//!
//! A cache file is a 36-byte header and one stream of 16-byte BC blocks:
//!
//! ```text
//! "SIIMP\0\0\x03"  key: u64  grid: u32  rx: u32  ry: u32  mips: u32  len: u32
//! for level in 0..mips:   albedo (BC3)   normal (BC5)   depth (BC5)     -- (grid rx >> level) x (grid ry >> level) texels
//! for level in 1..mips:   visibility (BC3), half resolution: the size of the others' level
//! ```
//!
//! An atlas is `grid` x `grid` frames (views of the plant) of `rx` x `ry` texels. The blocks of an
//! image are in raster order, 16 bytes each:
//!
//! | image      | bytes 0..8 (BC4)             | bytes 8..16                          |
//! |------------|------------------------------|--------------------------------------|
//! | albedo     | ambient occlusion            | BC1 colour: sqrt(albedo)             |
//! | normal     | octahedral normal x          | BC4: octahedral normal y             |
//! | depth      | depth along the view         | BC4: coverage (0 = no surface)       |
//! | visibility | sky visibility               | BC1 colour: normal spread, light fit |
//!
//! A BC4 block is two endpoint bytes and sixteen 3-bit indices; a BC1 block two RGB565 endpoints
//! and sixteen 2-bit indices.

use anyhow::{bail, Result};

pub const SIIMP_MAGIC: &[u8; 8] = b"SIIMP\0\0\x03";
pub const SIIMP_HEADER: usize = 36;

/// The shape of an atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub grid: u32,
    pub rx: u32,
    pub ry: u32,
    pub mips: u32,
}

/// One image level in the stream: image (0 albedo, 1 normal, 2 depth, 3 visibility), its level,
/// its size in blocks, and its byte offset in the stream.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub bw: usize,
    pub bh: usize,
    pub offset: usize,
}

impl Shape {
    /// Blocks across and down of the colour images at `level` (the visibility image of level
    /// `level - 1` has the same size).
    pub fn blocks(&self, level: u32) -> (usize, usize) {
        ((((self.grid * self.rx) >> level) / 4) as usize, (((self.grid * self.ry) >> level) / 4) as usize)
    }

    /// The region of image `img` (0..3) at `level`; for the visibility image (3), its own level.
    pub fn region(&self, img: usize, level: u32) -> Region {
        let bytes = |l: u32| {
            let (bw, bh) = self.blocks(l);
            bw * bh * 16
        };
        if img < 3 {
            let before: usize = (0..level).map(|l| 3 * bytes(l)).sum();
            let (bw, bh) = self.blocks(level);
            Region { bw, bh, offset: before + img * bytes(level) }
        } else {
            let colour: usize = (0..self.mips).map(|l| 3 * bytes(l)).sum();
            let before: usize = (0..level).map(|l| bytes(l + 1)).sum();
            let (bw, bh) = self.blocks(level + 1);
            Region { bw, bh, offset: colour + before }
        }
    }

    pub fn stream_bytes(&self) -> usize {
        let bytes = |l: u32| {
            let (bw, bh) = self.blocks(l);
            bw * bh * 16
        };
        (0..self.mips).map(|l| 3 * bytes(l)).sum::<usize>() + (1..self.mips).map(bytes).sum::<usize>()
    }

    pub fn check(&self) -> Result<()> {
        if self.grid == 0 || self.grid > 64 || self.mips == 0 || self.mips > 12 || self.rx == 0 || self.ry == 0 || self.rx > 4096 || self.ry > 4096 {
            bail!("not an atlas shape: {self:?}");
        }
        for l in 0..self.mips {
            if !(self.rx >> l).is_multiple_of(4) || !(self.ry >> l).is_multiple_of(4) || (self.rx >> l) == 0 || (self.ry >> l) == 0 {
                bail!("frames of {}x{} texels do not hold {} levels of whole blocks", self.rx, self.ry, self.mips);
            }
        }
        Ok(())
    }
}

/// A parsed cache file: key, shape and the block stream.
pub struct Atlas<'a> {
    pub key: u64,
    pub shape: Shape,
    pub stream: &'a [u8],
}

pub fn parse(file: &[u8]) -> Result<Atlas<'_>> {
    if file.len() < SIIMP_HEADER || &file[..8] != SIIMP_MAGIC {
        bail!("not a SmartImposter atlas cache file (version 3)");
    }
    let u = |i: usize| u32::from_le_bytes(file[i..i + 4].try_into().unwrap());
    let shape = Shape { grid: u(16), rx: u(20), ry: u(24), mips: u(28) };
    shape.check()?;
    let len = u(32) as usize;
    if len != shape.stream_bytes() || file.len() != SIIMP_HEADER + len {
        bail!("atlas stream is {} bytes, its shape says {}", file.len() - SIIMP_HEADER, shape.stream_bytes());
    }
    Ok(Atlas { key: u64::from_le_bytes(file[8..16].try_into().unwrap()), shape, stream: &file[SIIMP_HEADER..] })
}

/// The cache file of a stream.
pub fn unparse(key: u64, shape: Shape, stream: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(SIIMP_HEADER + stream.len());
    out.extend_from_slice(SIIMP_MAGIC);
    out.extend_from_slice(&key.to_le_bytes());
    for v in [shape.grid, shape.rx, shape.ry, shape.mips, stream.len() as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(stream);
    out
}

/// BC4 index -> rank of its value, 0 (the low endpoint) to 7 (the high one), for blocks with
/// endpoint 0 > endpoint 1 (the 8-value mode; the atlas encoder writes no other).
pub const RANK: [u8; 8] = [7, 0, 6, 5, 4, 3, 2, 1];
pub const UNRANK: [u8; 8] = [1, 7, 6, 5, 4, 3, 2, 0];
/// BC1 index -> position along the endpoint line: 0 (endpoint 0), 1, 2, 3 (endpoint 1).
pub const POS: [u8; 4] = [0, 3, 1, 2];
pub const UNPOS: [u8; 4] = [0, 2, 3, 1];

/// A BC4 block: endpoints (hi > lo) and the rank of every texel.
#[derive(Clone, Copy, Default)]
pub struct Bc4 {
    pub hi: u8,
    pub lo: u8,
    pub rank: [u8; 16],
}

impl Bc4 {
    pub fn read(b: &[u8]) -> Self {
        let bits = u64::from_le_bytes([b[2], b[3], b[4], b[5], b[6], b[7], 0, 0]);
        let mut rank = [0u8; 16];
        for (i, r) in rank.iter_mut().enumerate() {
            *r = RANK[((bits >> (3 * i)) & 7) as usize];
        }
        Bc4 { hi: b[0], lo: b[1], rank }
    }
    pub fn write(&self, b: &mut [u8]) {
        let mut bits = 0u64;
        for (i, r) in self.rank.iter().enumerate() {
            bits |= (UNRANK[*r as usize] as u64) << (3 * i);
        }
        b[0] = self.hi;
        b[1] = self.lo;
        b[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
    }
    /// The value (0..=255) of rank r: the 8-value interpolation, rounded.
    #[inline]
    pub fn value(&self, r: u8) -> u8 {
        (self.lo as u32 + (self.hi.wrapping_sub(self.lo) as u32 * r as u32 + 3) / 7) as u8
    }
    /// Where value v falls on this block's scale of ranks.
    #[inline]
    pub fn rank_of(&self, v: u8) -> u8 {
        if v <= self.lo {
            return 0;
        }
        // (hi > lo in every block the codec holds; a damaged file must not divide by zero)
        let d = (self.hi.wrapping_sub(self.lo) as u32).max(1);
        (((v - self.lo) as u32 * 7 + d / 2) / d).min(7) as u8
    }
    /// The rank the atlas encoder (bcenc.comp) gives a texel of value `v` (0..1): its rounding.
    pub fn encoder_rank(&self, v: f32) -> u8 {
        let (f0, f1) = (self.hi as f32 / 255.0, self.lo as f32 / 255.0);
        let t = ((v - f1) / (f0 - f1)).clamp(0.0, 1.0) * 7.0;
        (t + 0.5).floor() as u8
    }
}

/// A BC1 colour block: endpoints as (r5, g6, b5) and the position of every texel.
#[derive(Clone, Copy, Default)]
pub struct Bc1 {
    pub c0: [u8; 3],
    pub c1: [u8; 3],
    pub pos: [u8; 16],
}

fn unpack565(p: u16) -> [u8; 3] {
    [(p >> 11) as u8 & 31, (p >> 5) as u8 & 63, p as u8 & 31]
}
fn pack565(c: [u8; 3]) -> u16 {
    (c[0] as u16) << 11 | (c[1] as u16) << 5 | c[2] as u16
}

impl Bc1 {
    pub fn read(b: &[u8]) -> Self {
        let idx = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        let mut pos = [0u8; 16];
        for (i, p) in pos.iter_mut().enumerate() {
            *p = POS[((idx >> (2 * i)) & 3) as usize];
        }
        Bc1 { c0: unpack565(u16::from_le_bytes([b[0], b[1]])), c1: unpack565(u16::from_le_bytes([b[2], b[3]])), pos }
    }
    pub fn write(&self, b: &mut [u8]) {
        let mut idx = 0u32;
        for (i, p) in self.pos.iter().enumerate() {
            idx |= (UNPOS[*p as usize] as u32) << (2 * i);
        }
        b[0..2].copy_from_slice(&pack565(self.c0).to_le_bytes());
        b[2..4].copy_from_slice(&pack565(self.c1).to_le_bytes());
        b[4..8].copy_from_slice(&idx.to_le_bytes());
    }
    /// The position the atlas encoder (bcenc.comp bc1) gives a texel of colour `c`: the nearest
    /// of the four palette colours, the first of equals.
    pub fn encoder_pos(&self, c: [f32; 3]) -> u8 {
        let f = |e: [u8; 3]| [e[0] as f32 / 31.0, e[1] as f32 / 63.0, e[2] as f32 / 31.0];
        let (p0, p1) = (f(self.c0), f(self.c1));
        let mix = |a: [f32; 3], b: [f32; 3]| [(2.0 * a[0] + b[0]) / 3.0, (2.0 * a[1] + b[1]) / 3.0, (2.0 * a[2] + b[2]) / 3.0];
        let pal = [p0, p1, mix(p0, p1), mix(p1, p0)];
        let (mut best, mut bd) = (0usize, 1e9f32);
        for (k, p) in pal.iter().enumerate() {
            let d = [c[0] - p[0], c[1] - p[1], c[2] - p[2]];
            let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if dd < bd {
                bd = dd;
                best = k;
            }
        }
        POS[best]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_tile_the_stream() {
        let s = Shape { grid: 12, rx: 32, ry: 64, mips: 3 };
        s.check().unwrap();
        let mut at = 0;
        for level in 0..3 {
            for img in 0..3 {
                let r = s.region(img, level);
                assert_eq!(r.offset, at);
                at += r.bw * r.bh * 16;
            }
        }
        for level in 0..2 {
            let r = s.region(3, level);
            assert_eq!(r.offset, at);
            assert_eq!((r.bw, r.bh), s.blocks(level + 1));
            at += r.bw * r.bh * 16;
        }
        assert_eq!(at, s.stream_bytes());
    }

    #[test]
    fn blocks_round_trip() {
        let raw = [200u8, 13, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc];
        let mut out = [0u8; 8];
        Bc4::read(&raw).write(&mut out);
        assert_eq!(raw, out);
        let raw = [0x1f, 0xa3, 0x02, 0x40, 0xde, 0xad, 0xbe, 0xef];
        Bc1::read(&raw).write(&mut out);
        assert_eq!(raw, out);
        let b = Bc4 { hi: 255, lo: 0, rank: [0; 16] };
        assert_eq!((b.value(0), b.value(7), b.rank_of(128), b.encoder_rank(0.5)), (0, 255, 4, 4));
    }
}
