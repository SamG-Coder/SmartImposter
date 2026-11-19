//! Frond: the model and the container. See FORMAT.md for the format as a specification.
//!
//! The idea: the four atlases of an impostor describe one silhouette. Where the coverage atlas
//! says "no surface", the other atlases hold nothing: a block without coverage is a constant, and
//! in a block that is partly covered, the texels without coverage carry a value the atlas encoder
//! derived from the block's endpoints alone. So every block is coded coverage first, and the
//! depth, occlusion, normal, colour and visibility blocks after it, conditioned on it.
//!
//! The second idea: the atlases are mip chains, and a texel of one level is the coverage-weighted
//! mean of the four texels under it. Levels are coded coarse to fine, and a texel is coded
//! knowing what it and its siblings not yet coded must average to for the level above to come
//! out as it did: the first of four is predicted loosely, the last almost exactly.

// (channels, colours and components are indexed together across several tables)
#![allow(clippy::needless_range_loop)]

use crate::bc::{Bc1, Bc4, Shape};
use crate::rans::{Coder, Decoder, Encoder, Model};
use anyhow::{bail, Result};

pub const MAGIC: &[u8; 8] = b"FROND\0\0\x01";

/// BC4 channels of a block, in coding order.
pub const COVERAGE: usize = 0;
pub const DEPTH: usize = 1;
pub const OCCLUSION: usize = 2;
pub const NORMAL_X: usize = 3;
pub const NORMAL_Y: usize = 4;
pub const VISIBILITY: usize = 5;
/// The value (0..1) the atlas bake leaves in a texel without coverage, per BC4 channel
/// (bake.rgen / mip.comp / bcenc.comp); coverage itself is not conditioned on anything.
const BLANK: [f32; 6] = [0.0, 0.0, 1.0, 0.5, 0.5, 1.0];
/// The same for the two BC1 colour blocks: albedo (black), visibility colour (mid grey).
const BLANK_COLOUR: [[f32; 3]; 2] = [[0.0; 3], [0.5; 3]];

/// Bit accounting categories (`frond stats`).
pub const CATEGORIES: [&str; 17] = [
    "block class",
    "coverage endpoints",
    "coverage indices",
    "depth endpoints",
    "depth indices",
    "occlusion endpoints",
    "occlusion indices",
    "normal x endpoints",
    "normal x indices",
    "normal y endpoints",
    "normal y indices",
    "visibility endpoints",
    "visibility indices",
    "albedo endpoints",
    "albedo indices",
    "vis colour endpoints",
    "vis colour indices",
];

/// Adaptation rates (a boundary moves 1 / 2^rate of the way): fast where there are many contexts
/// with few symbols each, slower for the stationary ones.
const RATE_CLASS: u32 = 5;
const RATE_END: u32 = 6;
const RATE_RANK: u32 = 6;
const RATE_HIT: u32 = 6;

/// A chunk: the blocks of rows row0..row0 + rows of every image at one level (albedo, normal and
/// depth of `level`, and the visibility image of `level - 1`, which has the same size). The
/// chunks of a level need the level above it and nothing else: they decode in parallel, coarse
/// levels first, and a reader that only wants the coarse levels (the renderer's base atlas)
/// stops there.
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub level: u32,
    pub row0: usize,
    pub rows: usize,
}

/// The chunks of an atlas, in file order: from the coarsest level to the finest; per level, one
/// per row of frames, or the whole level where that is small.
pub fn chunks(shape: &Shape) -> Vec<Chunk> {
    let mut out = Vec::new();
    for level in (0..shape.mips).rev() {
        let (bw, bh) = shape.blocks(level);
        let band = ((shape.ry >> level) / 4) as usize;
        if bw * band >= 4096 {
            out.extend((0..shape.grid as usize).map(|k| Chunk { level, row0: k * band, rows: band }));
        } else {
            out.push(Chunk { level, row0: 0, rows: bh });
        }
    }
    out
}

#[derive(Clone)]
struct Models {
    class: [Model<4>; 9],
    /// BC4 high endpoint: residual against the neighbour block's, high and low nibble
    hi_hi: [[Model<17>; 3]; 6],
    hi_lo: [[Model<17>; 16]; 6],
    /// BC4 endpoint distance - 1, by the neighbour's distance (log2 bucket)
    d_hi: [[Model<17>; 9]; 6],
    d_lo: [[Model<17>; 16]; 6],
    /// with a level above: how far the high endpoint is above the highest of the four texels
    /// over the block, and the low one below the lowest; by how far apart those are (log2 bucket)
    up_hi: [[[Model<17>; 9]; 2]; 6],
    up_lo: [[[Model<17>; 16]; 2]; 6],
    /// rank of a covered texel, by what the level above says it should be and the state of its
    /// left and up neighbours
    rank: [Vec<Model<9>>; 6],
    /// texel without coverage: is its rank the encoder's? and the rank where it is not
    hit: [Model<3>; 6],
    miss: [Model<9>; 6],
    /// BC1 endpoint 0: residual against the neighbour block's, per component (r / b: 5 bits, g: 6)
    c_rb: [[[Model<33>; 3]; 2]; 2],
    c_g: [[Model<65>; 3]; 2],
    /// BC1 endpoint 1: difference from endpoint 0
    c_drb: [[Model<33>; 2]; 2],
    c_dg: [Model<65>; 2],
    /// with a level above: endpoint 0 above the highest of the four colours over the block,
    /// endpoint 1 below the lowest, per component; by how far apart those are (log2 bucket)
    c_up_rb: [[[[Model<33>; 7]; 2]; 2]; 2],
    c_up_g: [[[Model<65>; 8]; 2]; 2],
    /// position of a covered texel, by its place among its siblings, what the level above
    /// expects and its left and up neighbours
    pos: [[Model<5>; 864]; 2],
    chit: [Model<3>; 2],
    cmiss: [Model<5>; 2],
}

impl Models {
    fn new() -> Box<Self> {
        Box::new(Models {
            class: [Model::new(); 9],
            hi_hi: [[Model::new(); 3]; 6],
            hi_lo: [[Model::new(); 16]; 6],
            d_hi: [[Model::new(); 9]; 6],
            d_lo: [[Model::new(); 16]; 6],
            up_hi: [[[Model::new(); 9]; 2]; 6],
            up_lo: [[[Model::new(); 16]; 2]; 6],
            rank: std::array::from_fn(|_| vec![Model::new(); 4000]),
            hit: [Model::new(); 6],
            miss: [Model::new(); 6],
            c_rb: [[[Model::new(); 3]; 2]; 2],
            c_g: [[Model::new(); 3]; 2],
            c_drb: [[Model::new(); 2]; 2],
            c_dg: [Model::new(); 2],
            c_up_rb: [[[[Model::new(); 7]; 2]; 2]; 2],
            c_up_g: [[[Model::new(); 8]; 2]; 2],
            pos: [[Model::new(); 864]; 2],
            chit: [Model::new(); 2],
            cmiss: [Model::new(); 2],
        })
    }
}

/// The texels of a band of block rows, or of a whole level: has coverage, and the value
/// (0..=255) per BC4 channel. Texels of empty and literal blocks: no coverage, 0.
struct Planes {
    w: usize,
    valid: Vec<u8>,
    val: [Vec<u8>; 6],
    /// and the colour (r5, g6, b5) per BC1 block
    col: [Vec<[u8; 3]>; 2],
}

impl Planes {
    fn new(w: usize, h: usize) -> Self {
        Planes { w, valid: vec![0; w * h], val: std::array::from_fn(|_| vec![0; w * h]), col: std::array::from_fn(|_| vec![[0; 3]; w * h]) }
    }
}

/// What the model remembers of the chunk coded so far.
struct State {
    /// the chunk's texels, and the texel row of the level its first row is
    p: Planes,
    y0: usize,
    /// per texel and BC1 block: the position
    pos: [Vec<u8>; 2],
    /// per block column: the class and endpoints of the last block coded there (this row left
    /// of the current block, the row above from it on)
    class: Vec<u8>,
    ends: [Vec<[u8; 2]>; 6],
    cends: [Vec<[u8; 3]>; 2],
}

/// The four images' blocks at one position, as fields.
#[derive(Default)]
struct Block {
    bc4: [Bc4; 6],
    bc1: [Bc1; 2],
}

const EMPTY: usize = 0;
const CODED: usize = 1;
const LITERAL: usize = 2;

fn log2_bucket(d: u8) -> usize {
    (8 - (d as u32 + 1).leading_zeros().saturating_sub(24)) as usize // 1..=8
}

/// Code one block's fields. `left` / `up`: a coded block is next to it (its endpoints predict
/// this one's). The same body encodes (fields in) and decodes (fields out).
#[allow(clippy::too_many_arguments)]
fn code_block<C: Coder>(c: &mut C, m: &mut Models, st: &mut State, above: Option<&Planes>, bx: usize, by: usize, left: bool, up: bool, vis: bool, b: &mut Block) {
    let w = st.p.w;
    let texel = |i: usize| (by * 4 + i / 4) * w + bx * 4 + i % 4;
    let pred = if left { 1 } else if up { 2 } else { 0 };
    let from = if left { bx - 1 } else { bx };
    let mut valid = [true; 16];
    // the weight of a texel in the level above: its coverage (mip.comp), 1 for coverage itself
    let mut weight = [1i32; 16];
    for ch in 0..6 {
        if ch == VISIBILITY && !vis {
            continue;
        }
        // ---- endpoints
        // The block's values run from lo to hi, and the four texels of the level above that lie
        // over it are means of its texels: hi is at or above the highest of them, lo at or below
        // the lowest (but for rounding). Without a level above (or with nothing there): the high
        // one against the neighbour block's, then the distance.
        // (coverage is 0 or 1 nearly everywhere: its endpoints follow the neighbour's best)
        let over = above.filter(|_| ch != COVERAGE).and_then(|pa| {
            let (x, y) = (bx * 2, (st.y0 + by * 4) / 2);
            let (mut mn, mut mx) = (255u8, 0u8);
            for n in [y * pa.w + x, y * pa.w + x + 1, (y + 1) * pa.w + x, (y + 1) * pa.w + x + 1] {
                if pa.valid[n] != 0 {
                    (mn, mx) = (mn.min(pa.val[ch][n]), mx.max(pa.val[ch][n]));
                }
            }
            (mn <= mx).then_some((mn, mx))
        });
        let (hi, lo) = if let Some((mn, mx)) = over {
            let bucket = if mx == mn { 0 } else { log2_bucket(mx - mn - 1) };
            let r = b.bc4[ch].hi.wrapping_sub(mx) as usize;
            let hn = c.sym(&mut m.up_hi[ch][0][bucket], RATE_END, r >> 4);
            let ln = c.sym(&mut m.up_lo[ch][0][hn], RATE_END, r & 15);
            let hi = ((hn << 4 | ln) as u8).wrapping_add(mx);
            let r = mn.wrapping_sub(b.bc4[ch].lo) as usize;
            let hn = c.sym(&mut m.up_hi[ch][1][bucket], RATE_END, r >> 4);
            let ln = c.sym(&mut m.up_lo[ch][1][hn], RATE_END, r & 15);
            (hi, mn.wrapping_sub((hn << 4 | ln) as u8))
        } else {
            let (ph, pl) = if pred > 0 { (st.ends[ch][from][0], st.ends[ch][from][1]) } else { (0, 0) };
            let r = b.bc4[ch].hi.wrapping_sub(ph) as usize;
            let hn = c.sym(&mut m.hi_hi[ch][pred], RATE_END, r >> 4);
            let ln = c.sym(&mut m.hi_lo[ch][hn], RATE_END, r & 15);
            let hi = ((hn << 4 | ln) as u8).wrapping_add(ph);
            let bucket = if pred > 0 { log2_bucket(ph.wrapping_sub(pl).wrapping_sub(1)) } else { 0 };
            let d = b.bc4[ch].hi.wrapping_sub(b.bc4[ch].lo).wrapping_sub(1) as usize;
            let hn = c.sym(&mut m.d_hi[ch][bucket], RATE_END, d >> 4);
            let ln = c.sym(&mut m.d_lo[ch][hn], RATE_END, d & 15);
            (hi, hi.wrapping_sub((hn << 4 | ln) as u8).wrapping_sub(1))
        };
        (b.bc4[ch].hi, b.bc4[ch].lo) = (hi, lo);
        st.ends[ch][bx] = [hi, lo];
        c.tally(1 + 2 * ch);
        // ---- ranks
        let blk = b.bc4[ch];
        // (a block the model cannot hold, endpoints not high > low, is never coded: LITERAL)
        let blank = if ch == COVERAGE { 0 } else { blk.encoder_rank(BLANK[ch]) };
        let mut value = [0i32; 16];
        let ones = [1i32; 16];
        let wt = if ch == COVERAGE { ones } else { weight };
        for i in 0..16 {
            let t = texel(i);
            let rank = if ch != COVERAGE && !valid[i] {
                // no coverage here: the encoder's rank for the blank value, nearly always
                if c.sym(&mut m.hit[ch], RATE_HIT, (b.bc4[ch].rank[i] == blank) as usize) == 1 {
                    blank
                } else {
                    c.sym(&mut m.miss[ch], RATE_RANK, b.bc4[ch].rank[i] as usize) as u8
                }
            } else {
                // by the neighbours' values on this block's scale: 0..=7, 8 none, 9 no coverage
                let state = |n: Option<usize>| match n {
                    None => 8,
                    Some(n) if ch != COVERAGE && st.p.valid[n] == 0 => 9,
                    Some(n) => blk.rank_of(st.p.val[ch][n]) as usize,
                };
                let l = state((bx * 4 + i % 4 > 0).then(|| t - 1));
                let u = state((by * 4 + i / 4 > 0).then(|| t - w));
                // What the level above expects here: its texel is the weighted mean of this one
                // and its three siblings (2 x 2, inside the block), so the texels not yet coded
                // must average to (mean x weights - the coded ones) / (their weights).
                // 0..=7 on this block's scale, 8 no level above, 9 nothing there.
                let a = match above {
                    None => 8,
                    Some(pa) => {
                        let n = ((st.y0 + by * 4 + i / 4) / 2) * pa.w + (bx * 4 + i % 4) / 2;
                        if ch != COVERAGE && pa.valid[n] == 0 {
                            9
                        } else {
                            let first = i - ((i % 4) & 1) - 4 * ((i / 4) & 1);
                            let sib = [first, first + 1, first + 4, first + 5];
                            let mut rest = pa.val[ch][n] as i32 * sib.iter().map(|&j| wt[j]).sum::<i32>();
                            let mut left_w = 0;
                            for &j in &sib {
                                if j < i {
                                    rest -= wt[j] * value[j];
                                } else {
                                    left_w += wt[j];
                                }
                            }
                            blk.rank_of((rest / left_w.max(1)).clamp(0, 255) as u8) as usize
                        }
                    }
                };
                let q = ((i % 4) & 1) + 2 * ((i / 4) & 1);
                c.sym(&mut m.rank[ch][((q * 10 + a) * 10 + l) * 10 + u], RATE_RANK, b.bc4[ch].rank[i] as usize) as u8
            };
            b.bc4[ch].rank[i] = rank;
            value[i] = blk.value(rank) as i32;
            st.p.val[ch][t] = value[i] as u8;
            if ch == COVERAGE {
                valid[i] = !(rank == 0 && lo == 0);
                st.p.valid[t] = valid[i] as u8;
                weight[i] = if valid[i] { value[i].max(1) } else { 0 };
            }
        }
        c.tally(2 + 2 * ch);
    }
    for col in 0..2 {
        if col == 1 && !vis {
            continue;
        }
        // ---- colour endpoints
        // As the BC4 endpoints: endpoint 0 is the block's highest colour (per component, a little
        // inset) and endpoint 1 its lowest, around the four colours of the level above. Without
        // one: endpoint 0 against the neighbour's, endpoint 1 against endpoint 0.
        let over = above.and_then(|pa| {
            let (x, y) = (bx * 2, (st.y0 + by * 4) / 2);
            let (mut mn, mut mx) = ([255u8; 3], [0u8; 3]);
            for n in [y * pa.w + x, y * pa.w + x + 1, (y + 1) * pa.w + x, (y + 1) * pa.w + x + 1] {
                if pa.valid[n] != 0 {
                    for k in 0..3 {
                        (mn[k], mx[k]) = (mn[k].min(pa.col[col][n][k]), mx[k].max(pa.col[col][n][k]));
                    }
                }
            }
            (mn[0] <= mx[0]).then_some((mn, mx))
        });
        let p = if pred > 0 { st.cends[col][from] } else { [0; 3] };
        for k in 0..3 {
            let mask = if k == 1 { 63u8 } else { 31u8 };
            if let Some((mn, mx)) = over {
                let bucket = if mx[k] == mn[k] { 0 } else { log2_bucket(mx[k] - mn[k] - 1) };
                let r = (b.bc1[col].c0[k].wrapping_sub(mx[k]) & mask) as usize;
                let r = if k == 1 { c.sym(&mut m.c_up_g[col][0][bucket], RATE_END, r) } else { c.sym(&mut m.c_up_rb[col][k / 2][0][bucket], RATE_END, r) };
                b.bc1[col].c0[k] = (r as u8).wrapping_add(mx[k]) & mask;
                let r = (mn[k].wrapping_sub(b.bc1[col].c1[k]) & mask) as usize;
                let r = if k == 1 { c.sym(&mut m.c_up_g[col][1][bucket], RATE_END, r) } else { c.sym(&mut m.c_up_rb[col][k / 2][1][bucket], RATE_END, r) };
                b.bc1[col].c1[k] = mn[k].wrapping_sub(r as u8) & mask;
            } else {
                let r = (b.bc1[col].c0[k].wrapping_sub(p[k]) & mask) as usize;
                let r = if k == 1 { c.sym(&mut m.c_g[col][pred], RATE_END, r) } else { c.sym(&mut m.c_rb[col][k / 2][pred], RATE_END, r) };
                let c0 = (r as u8).wrapping_add(p[k]) & mask;
                let d = (c0.wrapping_sub(b.bc1[col].c1[k]) & mask) as usize;
                let d = if k == 1 { c.sym(&mut m.c_dg[col], RATE_END, d) } else { c.sym(&mut m.c_drb[col][k / 2], RATE_END, d) };
                b.bc1[col].c0[k] = c0;
                b.bc1[col].c1[k] = c0.wrapping_sub(d as u8) & mask;
            }
        }
        st.cends[col][bx] = b.bc1[col].c0;
        c.tally(13 + 2 * col);
        // ---- positions
        let blank = b.bc1[col].encoder_pos(BLANK_COLOUR[col]);
        let (c0, c1) = (b.bc1[col].c0.map(|v| v as i32), b.bc1[col].c1.map(|v| v as i32));
        // the colour of a position, and the position nearest a colour along the endpoint line
        // (red and blue count double: they have half green's steps)
        let colour = |p: u8| -> [i32; 3] { std::array::from_fn(|k| (c0[k] * (3 - p as i32) + c1[k] * p as i32 + 1) / 3) };
        let scale = [2, 1, 2];
        let len2: i32 = (0..3).map(|k| ((c1[k] - c0[k]) * scale[k]).pow(2)).sum();
        let nearest = |v: [i32; 3]| -> usize {
            let dot: i32 = (0..3).map(|k| (v[k] - c0[k]) * (c1[k] - c0[k]) * scale[k] * scale[k]).sum();
            if len2 == 0 { 0 } else { ((3 * dot * 2 + len2) / (2 * len2)).clamp(0, 3) as usize }
        };
        let mut value = [[0i32; 3]; 16];
        for i in 0..16 {
            let t = texel(i);
            let pos = if !valid[i] {
                if c.sym(&mut m.chit[col], RATE_HIT, (b.bc1[col].pos[i] == blank) as usize) == 1 {
                    blank
                } else {
                    c.sym(&mut m.cmiss[col], RATE_RANK, b.bc1[col].pos[i] as usize) as u8
                }
            } else {
                let state = |n: Option<usize>| match n {
                    None => 4,
                    Some(n) if st.p.valid[n] == 0 => 5,
                    Some(n) => st.pos[col][n] as usize,
                };
                let l = state((bx * 4 + i % 4 > 0).then(|| t - 1));
                let u = state((by * 4 + i / 4 > 0).then(|| t - w));
                // what the level above expects, as for the BC4 ranks: 0..=3, 4 none, 5 nothing there
                let a = match above {
                    None => 4,
                    Some(pa) => {
                        let n = ((st.y0 + by * 4 + i / 4) / 2) * pa.w + (bx * 4 + i % 4) / 2;
                        if pa.valid[n] == 0 {
                            5
                        } else {
                            let first = i - ((i % 4) & 1) - 4 * ((i / 4) & 1);
                            let sib = [first, first + 1, first + 4, first + 5];
                            let all: i32 = sib.iter().map(|&j| weight[j]).sum();
                            let mut rest: [i32; 3] = std::array::from_fn(|k| pa.col[col][n][k] as i32 * all);
                            let mut left_w = 0;
                            for &j in &sib {
                                if j < i {
                                    (0..3).for_each(|k| rest[k] -= weight[j] * value[j][k]);
                                } else {
                                    left_w += weight[j];
                                }
                            }
                            nearest(rest.map(|v| v / left_w.max(1)))
                        }
                    }
                };
                let q = ((i % 4) & 1) + 2 * ((i / 4) & 1);
                c.sym(&mut m.pos[col][((q * 6 + a) * 6 + l) * 6 + u], RATE_RANK, b.bc1[col].pos[i] as usize) as u8
            };
            b.bc1[col].pos[i] = pos;
            st.pos[col][t] = pos;
            value[i] = colour(pos);
            st.p.col[col][t] = value[i].map(|v| v as u8);
        }
        c.tally(14 + 2 * col);
    }
}

/// The byte offsets of block (bx, row) in the four images of a chunk's level.
struct Places {
    bw: usize,
    offset: [usize; 4],
    vis: bool,
}

impl Places {
    fn new(shape: &Shape, level: u32) -> Self {
        let vis = level >= 1;
        let r: [_; 3] = std::array::from_fn(|img| shape.region(img, level));
        let v = if vis { shape.region(3, level - 1).offset } else { 0 };
        Places { bw: r[0].bw, offset: [r[0].offset, r[1].offset, r[2].offset, v], vis }
    }
    fn at(&self, img: usize, bx: usize, row: usize) -> usize {
        self.offset[img] + (row * self.bw + bx) * 16
    }
    fn images(&self) -> usize {
        3 + self.vis as usize
    }
}

fn new_state(bw: usize, ch: &Chunk) -> State {
    let n = bw * 4 * ch.rows * 4;
    State {
        p: Planes::new(bw * 4, ch.rows * 4),
        y0: ch.row0 * 4,
        pos: std::array::from_fn(|_| vec![0; n]),
        class: vec![EMPTY as u8; bw],
        ends: std::array::from_fn(|_| vec![[0; 2]; bw]),
        cends: std::array::from_fn(|_| vec![[0; 3]; bw]),
    }
}

/// What coding a chunk leaves for the chunks that build on it: its texels (for the level below)
/// and its models as they ended (the next level's chunks start from the last chunk's).
struct Done {
    planes: Planes,
    models: Box<Models>,
}

/// One chunk: u32 length of the literal blocks, the literal blocks, the rANS stream.
fn encode_chunk(stream: &[u8], shape: &Shape, ch: &Chunk, empty: &[[u8; 16]; 4], above: Option<&Planes>, start: &Models, stats: Option<&mut Vec<f64>>) -> (Vec<u8>, Done) {
    let pl = Places::new(shape, ch.level);
    let mut enc = if stats.is_some() { Encoder::with_stats(CATEGORIES.len()) } else { Encoder::default() };
    let mut m = Box::new(start.clone());
    let mut st = new_state(pl.bw, ch);
    let mut literals: Vec<u8> = Vec::new();
    let mut prev = vec![EMPTY as u8; pl.bw];
    for by in 0..ch.rows {
        for bx in 0..pl.bw {
            let raw: [&[u8]; 4] = std::array::from_fn(|img| {
                if img < pl.images() {
                    let o = pl.at(img, bx, ch.row0 + by);
                    &stream[o..o + 16]
                } else {
                    &empty[3][..]
                }
            });
            let mut b = Block {
                bc4: [Bc4::read(&raw[2][8..]), Bc4::read(&raw[2][..8]), Bc4::read(&raw[0][..8]), Bc4::read(&raw[1][..8]), Bc4::read(&raw[1][8..]), Bc4::read(&raw[3][..8])],
                bc1: [Bc1::read(&raw[0][8..]), Bc1::read(&raw[3][8..])],
            };
            let class = if (0..4).all(|i| raw[i] == &empty[i][..]) {
                EMPTY
            } else if b.bc4.iter().take(5 + pl.vis as usize).all(|c| c.hi > c.lo) {
                CODED
            } else {
                LITERAL
            };
            let (l, u) = (if bx > 0 { st.class[bx - 1] } else { 0 }, prev[bx]);
            enc.sym(&mut m.class[l as usize * 3 + u as usize], RATE_CLASS, class);
            enc.tally(0);
            match class {
                CODED => code_block(&mut enc, &mut m, &mut st, above, bx, by, l == CODED as u8, u == CODED as u8, pl.vis, &mut b),
                LITERAL => (0..pl.images()).for_each(|i| literals.extend_from_slice(raw[i])),
                _ => {}
            }
            st.class[bx] = class as u8;
        }
        prev.copy_from_slice(&st.class);
    }
    if let (Some(total), Some(mine)) = (stats, &enc.stats) {
        for (t, v) in total.iter_mut().zip(mine) {
            *t += v;
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(literals.len() as u32).to_le_bytes());
    out.extend_from_slice(&literals);
    out.extend_from_slice(&enc.finish());
    (out, Done { planes: st.p, models: m })
}

/// A pointer to the output stream that chunk decoders share: every chunk writes its own rows.
#[derive(Clone, Copy)]
struct Out(*mut u8, usize);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    /// SAFETY: [offset, offset + 16) is inside the stream and belongs to the calling chunk alone.
    unsafe fn block<'a>(self, offset: usize) -> &'a mut [u8] {
        assert!(offset + 16 <= self.1);
        unsafe { std::slice::from_raw_parts_mut(self.0.add(offset), 16) }
    }
}

fn decode_chunk(data: &[u8], shape: &Shape, ch: &Chunk, empty: &[[u8; 16]; 4], above: Option<&Planes>, start: &Models, out: Out) -> Result<Done> {
    if data.len() < 4 {
        bail!("chunk too short");
    }
    let nlit = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    if data.len() < 4 + nlit {
        bail!("chunk literals run past its end");
    }
    let mut literals = &data[4..4 + nlit];
    let mut dec = Decoder::new(&data[4 + nlit..]);
    let pl = Places::new(shape, ch.level);
    let mut m = Box::new(start.clone());
    let mut st = new_state(pl.bw, ch);
    let mut prev = vec![EMPTY as u8; pl.bw];
    for by in 0..ch.rows {
        for bx in 0..pl.bw {
            let (l, u) = (if bx > 0 { st.class[bx - 1] } else { 0 }, prev[bx]);
            let class = dec.sym(&mut m.class[l as usize * 3 + u as usize], RATE_CLASS, 0);
            // SAFETY: the chunk's rows of each image are its own (chunks() partitions the rows)
            let row = ch.row0 + by;
            let (d0, d1, d2) = unsafe { (out.block(pl.at(0, bx, row)), out.block(pl.at(1, bx, row)), out.block(pl.at(2, bx, row))) };
            let mut none = [0u8; 16];
            let d3 = if pl.vis { unsafe { out.block(pl.at(3, bx, row)) } } else { &mut none[..] };
            match class {
                CODED => {
                    let mut b = Block::default();
                    code_block(&mut dec, &mut m, &mut st, above, bx, by, l == CODED as u8, u == CODED as u8, pl.vis, &mut b);
                    b.bc4[OCCLUSION].write(&mut d0[..8]);
                    b.bc1[0].write(&mut d0[8..]);
                    b.bc4[NORMAL_X].write(&mut d1[..8]);
                    b.bc4[NORMAL_Y].write(&mut d1[8..]);
                    b.bc4[DEPTH].write(&mut d2[..8]);
                    b.bc4[COVERAGE].write(&mut d2[8..]);
                    if pl.vis {
                        b.bc4[VISIBILITY].write(&mut d3[..8]);
                        b.bc1[1].write(&mut d3[8..]);
                    }
                }
                LITERAL => {
                    let n = pl.images() * 16;
                    if literals.len() < n {
                        bail!("chunk has fewer literal blocks than its classes say");
                    }
                    d0.copy_from_slice(&literals[..16]);
                    d1.copy_from_slice(&literals[16..32]);
                    d2.copy_from_slice(&literals[32..48]);
                    if pl.vis {
                        d3.copy_from_slice(&literals[48..64]);
                    }
                    literals = &literals[n..];
                }
                _ => {
                    d0.copy_from_slice(&empty[0]);
                    d1.copy_from_slice(&empty[1]);
                    d2.copy_from_slice(&empty[2]);
                    if pl.vis {
                        d3.copy_from_slice(&empty[3]);
                    }
                }
            }
            st.class[bx] = class as u8;
        }
        prev.copy_from_slice(&st.class);
    }
    Ok(Done { planes: st.p, models: m })
}

/// The chunks of one level, coded in parallel by `code`, and what they leave for the next level:
/// the level's texels (the bands put together) and the last chunk's models.
fn code_level<T: Send>(
    shape: &Shape,
    list: &[Chunk],
    threads: usize,
    above: &mut Option<Planes>,
    models: &mut Box<Models>,
    code: impl Fn(&Chunk, Option<&Planes>, &Models) -> Result<(T, Done)> + Sync,
) -> Result<Vec<T>> {
    let (bw, bh) = shape.blocks(list[0].level);
    let results = for_each_parallel(list.len(), threads, |i| code(&list[i], above.as_ref(), models));
    let mut level = Planes::new(bw * 4, bh * 4);
    let mut out = Vec::with_capacity(list.len());
    for (ch, r) in list.iter().zip(results) {
        let (t, done) = r?;
        let at = ch.row0 * 4 * level.w;
        level.valid[at..at + done.planes.valid.len()].copy_from_slice(&done.planes.valid);
        for (dst, src) in level.val.iter_mut().zip(&done.planes.val) {
            dst[at..at + src.len()].copy_from_slice(src);
        }
        for (dst, src) in level.col.iter_mut().zip(&done.planes.col) {
            dst[at..at + src.len()].copy_from_slice(src);
        }
        *models = done.models;
        out.push(t);
    }
    *above = Some(level);
    Ok(out)
}

/// A fast 64-bit checksum of the stream (8 bytes at a time).
pub fn checksum(d: &[u8]) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ d.len() as u64;
    let mut it = d.chunks_exact(8);
    for w in &mut it {
        h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 32;
    }
    for &b in it.remainder() {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// The block an image is made of where the plant is not: the most common one among its first.
fn empty_block(stream: &[u8], offset: usize, blocks: usize) -> [u8; 16] {
    let mut seen: Vec<([u8; 16], usize)> = Vec::new();
    for k in 0..blocks.min(4096) {
        let b: [u8; 16] = stream[offset + k * 16..offset + k * 16 + 16].try_into().unwrap();
        if let Some(i) = seen.iter().position(|s| s.0 == b) {
            seen[i].1 += 1;
        } else if seen.len() < 64 {
            seen.push((b, 1));
        }
    }
    seen.into_iter().max_by_key(|s| s.1).map(|s| s.0).unwrap_or([0; 16])
}

fn for_each_parallel<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<T>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n {
                    break;
                }
                *out[i].lock().unwrap() = Some(f(i));
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

/// Compress an atlas stream. `stats`: bits per CATEGORIES entry, summed over the chunks.
pub fn encode(key: u64, shape: &Shape, stream: &[u8], threads: usize, stats: Option<&mut Vec<f64>>) -> Vec<u8> {
    assert_eq!(stream.len(), shape.stream_bytes());
    let mut empty = [[0u8; 16]; 4];
    for (img, e) in empty.iter_mut().enumerate() {
        if img < 3 || shape.mips > 1 {
            let r = shape.region(img, 0);
            *e = empty_block(stream, r.offset, r.bw * r.bh);
        }
    }
    let list = chunks(shape);
    let want_stats = stats.is_some();
    let (mut above, mut models) = (None, Models::new());
    let mut coded = Vec::new();
    for level in list.chunk_by(|a, b| a.level == b.level) {
        let r = code_level(shape, level, threads, &mut above, &mut models, |ch, above, start| {
            let mut st = want_stats.then(|| vec![0.0; CATEGORIES.len()]);
            let (bytes, done) = encode_chunk(stream, shape, ch, &empty, above, start, st.as_mut());
            Ok(((bytes, st), done))
        });
        coded.extend(r.expect("encoding cannot fail"));
    }
    if let Some(total) = stats {
        for (_, st) in &coded {
            for (t, v) in total.iter_mut().zip(st.as_ref().unwrap()) {
                *t += v;
            }
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&key.to_le_bytes());
    for v in [shape.grid, shape.rx, shape.ry, shape.mips] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&checksum(stream).to_le_bytes());
    for e in &empty {
        out.extend_from_slice(e);
    }
    out.extend_from_slice(&(coded.len() as u32).to_le_bytes());
    for (c, _) in &coded {
        out.extend_from_slice(&(c.len() as u32).to_le_bytes());
    }
    for (c, _) in &coded {
        out.extend_from_slice(c);
    }
    out
}

/// A Frond file's header: what it holds and where its chunks are.
pub struct Header {
    pub key: u64,
    pub shape: Shape,
    pub checksum: u64,
    pub empty: [[u8; 16]; 4],
    /// (chunk, byte range in the file)
    pub chunks: Vec<(Chunk, std::ops::Range<usize>)>,
}

pub fn header(file: &[u8]) -> Result<Header> {
    if file.len() < 108 || &file[..8] != MAGIC {
        bail!("not a Frond file (version 1)");
    }
    let u = |i: usize| u32::from_le_bytes(file[i..i + 4].try_into().unwrap());
    let shape = Shape { grid: u(16), rx: u(20), ry: u(24), mips: u(28) };
    shape.check()?;
    let list = chunks(&shape);
    if u(104) as usize != list.len() || file.len() < 108 + 4 * list.len() {
        bail!("Frond file has {} chunks, its shape says {}", u(104), list.len());
    }
    let mut at = 108 + 4 * list.len();
    let mut out = Vec::with_capacity(list.len());
    for (i, c) in list.iter().enumerate() {
        let n = u(108 + 4 * i) as usize;
        if at + n > file.len() {
            bail!("Frond file is truncated");
        }
        out.push((*c, at..at + n));
        at += n;
    }
    Ok(Header {
        key: u64::from_le_bytes(file[8..16].try_into().unwrap()),
        shape,
        checksum: u64::from_le_bytes(file[32..40].try_into().unwrap()),
        empty: std::array::from_fn(|i| file[40 + 16 * i..56 + 16 * i].try_into().unwrap()),
        chunks: out,
    })
}

/// Decompress into `stream` (shape.stream_bytes() long). `from_level`: only the chunks of that
/// level and coarser are decoded (the rest of the stream is left as it is): a base atlas.
pub fn decode_into(file: &[u8], h: &Header, stream: &mut [u8], from_level: u32, threads: usize) -> Result<()> {
    if stream.len() != h.shape.stream_bytes() {
        bail!("output is {} bytes, the atlas stream {}", stream.len(), h.shape.stream_bytes());
    }
    let out = Out(stream.as_mut_ptr(), stream.len());
    let (mut above, mut models) = (None, Models::new());
    for level in h.chunks.chunk_by(|a, b| a.0.level == b.0.level).filter(|l| l[0].0.level >= from_level) {
        let list: Vec<Chunk> = level.iter().map(|c| c.0).collect();
        let range = |ch: &Chunk| level.iter().find(|c| c.0.row0 == ch.row0).unwrap().1.clone();
        code_level(&h.shape, &list, threads, &mut above, &mut models, |ch, above, start| {
            decode_chunk(&file[range(ch)], &h.shape, ch, &h.empty, above, start, out).map(|done| ((), done))
        })?;
    }
    Ok(())
}

/// Decompress a whole file: (key, shape, stream), checked against the stored checksum.
pub fn decode(file: &[u8], threads: usize) -> Result<(u64, Shape, Vec<u8>)> {
    let h = header(file)?;
    let mut stream = vec![0u8; h.shape.stream_bytes()];
    decode_into(file, &h, &mut stream, 0, threads)?;
    if checksum(&stream) != h.checksum {
        bail!("the decoded stream does not match the file's checksum");
    }
    Ok((h.key, h.shape, stream))
}
