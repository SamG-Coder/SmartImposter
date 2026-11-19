//! The lawn: short grass as a tiling parallax layer instead of geometry.
//!
//! Short grass is the worst case for geometry: tens of thousands of blades per square metre,
//! everywhere, each a few pixels wide next to the camera and far below a pixel fifty metres
//! out. As cards it is any-hit work on every ray that grazes the ground; as boxes with baked
//! views (the plant impostors) it is one box per patch for rays to stop at.
//!
//! So the lawn is no geometry at all. One small square of real blades (`tile`) is baked once,
//! like a plant impostor, into views from a grid of directions: per view and texel, how high the
//! first blade is that a ray from that direction meets, its colour, normal, coverage and how
//! much sky it sees. The views wrap around (the square is baked between copies of itself), so the
//! square tiles the plane. At run time a ray that hits lawn-covered terrain is marched through
//! that height field in the slab of one blade height above the ground (tri.rchit `lawn`): the
//! parallax march of the plant impostors, in the hit shader, with wrap-around lookups. Nothing
//! is added to the acceleration structures, far lawn is a coarser mip of the same atlas (so it
//! cannot pop or flicker differently from near lawn), and the whole lawn costs one atlas.
//!
//! The frames of a view from direction f (towards the viewer) are laid out on the plane of the
//! blade tips: texel (u, v) is the line through the point (u, v) x TILE of that plane along f,
//! and stores where the first blade on that line is, as a height above the ground. For f straight
//! up that is a height map; for a slanting f it is the height map sheared along f.
//!
//! Which views: a ray that is not exactly along a view has to be marched through it, sideways
//! by (blade height) x (the difference of the two directions' slopes, cot of the elevation). A
//! lawn is mostly seen at a low angle, where the slope changes fast, and from further away the
//! lower the angle, where a texel covers more. So the views are spaced evenly in azimuth and
//! geometrically in slope (AZIMUTHS x SLOPES, cot from SLOPE_MIN to SLOPE_MAX, ~79 to ~2.4
//! degrees: a standing eye sees the lawn through its own views out to ~40 m): the march then
//! takes about the same number of steps at every distance. The plants' hemisphere grid has a
//! handful of views below 20 degrees; a lawn drawn with it smears into streaks along the view
//! direction.
//!
//! Near the eye the layer is flat: it has the blades' parallax but no silhouette, and a blade
//! does not stand in front of anything. The taller the grass the more that shows. So around the
//! camera the tile itself is drawn, as real blades (`near_radius`; renderer.rs near grass): the
//! same blades the layer was baked from, standing on the terrain, shrinking into the ground
//! towards the edge of that circle while the layer fades in under them.
//!
//! Repetition: one square repeated shows as rows, every clump lining up with its copies along
//! every direction of the grid of squares. So every copy of the square is shifted by its own
//! random amount (lawn.glsl lawnTile; the square wraps around, so a shifted copy is whole), with
//! ragged borders between the copies: nothing lines up, and where two copies meet, blades of one
//! arrangement simply stand next to blades of another.

use crate::asset::MeshData;
use glam::Vec3;
use std::sync::atomic::{AtomicU32, Ordering};

/// With the lawn on, how much of the short card grass (the biome's `Grass::Moist` patches) the
/// forest scene still places, as f32 bits: the lawn is the ground cover, the patches that are
/// left are its tufts. 1 = all of it (no lawn). Set by main before the world is generated.
static SHORT_GRASS: AtomicU32 = AtomicU32::new(0x3f80_0000);
pub fn set_short_grass(keep: f32) {
    SHORT_GRASS.store(keep.to_bits(), Ordering::Relaxed);
}
pub fn short_grass() -> f32 {
    f32::from_bits(SHORT_GRASS.load(Ordering::Relaxed))
}

/// Height of the grass (m): no blade tip is above it. A mown lawn by default; `--lawn meadow`,
/// `--lawn tall` or `--lawn-height` (cm) make it a meadow or tall grass: the same layer, with
/// the tile, the blades and their number following the height (as f32 bits, set by main before
/// the world and the renderer start).
static HEIGHT: AtomicU32 = AtomicU32::new(0x3d38_51ec); // 0.045
pub const MEADOW: f32 = 0.22;
pub const TALL: f32 = 0.45;
pub fn set_height(m: f32) {
    HEIGHT.store(m.clamp(0.02, 0.6).to_bits(), Ordering::Relaxed);
}
pub fn height() -> f32 {
    f32::from_bits(HEIGHT.load(Ordering::Relaxed))
}
/// Unmown (0 a lawn .. 1 a meadow): blades of every length, arching over, with seed stalks; and
/// in the forest scene it stands in for the tall and dry grass patches too.
pub fn wild() -> f32 {
    ((height() - 0.08) / 0.12).clamp(0.0, 1.0)
}
/// Around the camera the grass is real blades (the tile as geometry) out to this distance (m),
/// 0 = the layer only (`--lawn-near`; negative = by the height: 4 m of a lawn, 29 m of tall
/// grass). As f32 bits.
static NEAR: AtomicU32 = AtomicU32::new(0xbf80_0000); // -1
pub fn set_near(m: f32) {
    NEAR.store(m.min(40.0).to_bits(), Ordering::Relaxed);
}
pub fn near_radius() -> f32 {
    let m = f32::from_bits(NEAR.load(Ordering::Relaxed));
    if m < 0.0 {
        (64.0 * height()).clamp(4.0, 30.0)
    } else {
        m
    }
}

/// Edge of the square that is baked and repeated (m): four blade heights, 1.5 mm a texel for a
/// lawn and ~4 mm for tall grass (whose blades are that much wider).
pub fn tile_size() -> f32 {
    (4.0 * height()).clamp(0.42, 1.6)
}
/// Views: azimuths (evenly round the compass) x slopes; both multiples of 4 (the atlas is a 4 x 4
/// grid of blocks of views, so the plants' atlas code holds it).
pub const AZIMUTHS: u32 = 16;
pub const SLOPES: u32 = 12;
/// cot(elevation) of the steepest and the flattest views; the ones between are geometric.
pub const SLOPE_MIN: f32 = 0.2;
pub const SLOPE_MAX: f32 = 24.0;
/// Texels per view, and how many of them on each side repeat the neighbouring copy (so filtered
/// lookups at the tile's edge read the right texels; a texel of the coarsest level).
pub const RES: u32 = 320;
pub const GUTTER: u32 = 20;
/// Blades in the tile (a lawn: 30,000 a square metre; taller grass has larger, fewer blades).
const BLADES: usize = 5300;

/// How many copies of the tile the bake needs on each side of it: the flattest view's rays run
/// SLOPE_MAX blade heights sideways before they reach the ground.
pub fn copies() -> i32 {
    (height() * SLOPE_MAX / tile_size() + 0.05).ceil() as i32
}
struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// One square (`tile_size`) of grass blades, bases at y = 0: curved, tapering ribbons in clumps,
/// leaning every way; unmown grass also has straight stalks with a seed head. UV: x across the
/// blade, y from root (0) to tip (1), + 2 on a stalk; the material byte carries a random number
/// per blade (tri.rchit colours the blades from these).
pub fn tile() -> MeshData {
    #[allow(non_snake_case)]
    let TILE = tile_size();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (height, wild) = (height(), wild());
    // (blades grow with the grass: 2.4-3.6 mm wide in a lawn, 6-9 mm in tall grass)
    let grown = (height / 0.045).sqrt().min(2.6);
    let n = BLADES;
    // (a clump: where it stands and its shade of green, which its blades share)
    let clumps: Vec<[f32; 3]> = (0..n / 28).map(|_| [rng.f() * TILE, rng.f() * TILE, rng.f()]).collect();
    // Taller and shorter patches a hand across (wrapping round the tile): what is left of the
    // lawn's texture from further away than single blades show.
    let patches = |x: f32, z: f32| {
        let (u, v) = (x / TILE * std::f32::consts::TAU, z / TILE * std::f32::consts::TAU);
        0.5 + 0.25 * ((2.0 * u + 1.3).sin() * (3.0 * v + 0.4).cos() + (3.0 * u - 2.0 * v + 2.1).sin())
    };
    let mut m = MeshData { name: "lawn tile".into(), positions: Vec::new(), normals: Vec::new(), uvs: Vec::new(), indices: Vec::new(), materials: Vec::new(), grid: 0 };
    // (long blades arch: more segments)
    #[allow(non_snake_case)]
    let SEG: usize = if wild > 0.3 { 6 } else { 4 };
    for _ in 0..n {
        // most blades stand in clumps a couple of centimetres across, the rest anywhere
        let (bx, bz, shade) = if rng.f() < 0.7 {
            let c = clumps[(rng.f() * clumps.len() as f32) as usize % clumps.len()];
            let (r, a) = (0.018 * grown * rng.f().sqrt(), rng.f() * std::f32::consts::TAU);
            ((c[0] + r * a.cos()).rem_euclid(TILE), (c[1] + r * a.sin()).rem_euclid(TILE), c[2] + 0.3 * (rng.f() - 0.5))
        } else {
            (rng.f() * TILE, rng.f() * TILE, rng.f())
        };
        // a mown lawn: blades cut to about one length, standing; unmown: every length, arching
        let (r_len, r_bend) = (rng.f(), rng.f() * rng.f());
        let stalk = rng.f() < 0.022 * wild;
        let mut length = height * ((0.5 + 0.5 * r_len.powf(0.6)) * (1.0 - wild) + (0.32 + 0.68 * r_len.powf(0.8)) * wild) * (0.78 + 0.22 * patches(bx, bz));
        let mut bend = (0.12 + 0.8 * r_bend + wild * (0.1 + 0.45 * r_len)).min(0.95); // how far the tip hangs over, x its length
        let az = rng.f() * std::f32::consts::TAU;
        let lean = Vec3::new(az.cos(), 0.0, az.sin());
        let twist = (rng.f() - 0.5) * 1.2;
        let mut width = (0.0024 + 0.0012 * rng.f()) * grown;
        if stalk {
            // a seed stalk: thin, nearly straight, to the top of the grass
            length = height * (0.62 + 0.38 * r_len);
            bend = 0.04 + 0.5 * r_bend;
            width *= 0.4;
        }
        let leaf = (shade.clamp(0.0, 0.999) * 64.0) as u8 & 63;
        let first = (m.positions.len() / 3) as u32;
        for s in 0..=SEG {
            let t = s as f32 / SEG as f32;
            let centre = Vec3::new(bx, 0.0, bz) + lean * (length * bend * t * t) + Vec3::Y * (length * t * (1.0 - 0.3 * bend * t));
            // along the blade, and across it (turning a little towards the tip)
            let along = (lean * (2.0 * bend * t) + Vec3::Y * (1.0 - 0.6 * bend * t)).normalize();
            let a = az + std::f32::consts::FRAC_PI_2 + twist * t;
            let across = Vec3::new(a.cos(), 0.0, a.sin());
            let across = (across - along * across.dot(along)).normalize();
            let half = if stalk {
                // (the seed head: a spindle over the top quarter)
                0.5 * width * (1.0 + 2.2 * (std::f32::consts::PI * ((t - 0.72) / 0.28).clamp(0.0, 1.0)).sin())
            } else {
                0.5 * width * (1.0 - 0.8 * t.powf(1.6))
            };
            let mut face = across.cross(along).normalize();
            if face.y < 0.0 {
                face = -face;
            }
            for side in [-1.0f32, 1.0] {
                let p = centre + across * (half * side);
                // normals lean up (a blade is lit from above whichever way it faces), and the
                // two edges fold apart a little (the blade's V section)
                let nrm = (face * 0.6 + Vec3::Y * 0.4 + across * (0.3 * side)).normalize();
                m.positions.extend_from_slice(&p.to_array());
                m.normals.extend_from_slice(&nrm.to_array());
                m.uvs.extend_from_slice(&[0.5 + 0.5 * side, t + if stalk { 2.0 } else { 0.0 }]);
            }
        }
        for s in 0..SEG as u32 {
            let k = first + 2 * s;
            m.indices.extend_from_slice(&[k, k + 1, k + 2, k + 1, k + 3, k + 2]);
            // (material 1 = leaf, the blade's random number above it)
            m.materials.extend_from_slice(&[1 | leaf << 2; 2]);
        }
    }
    while !m.materials.len().is_multiple_of(4) {
        m.materials.push(0);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tile_is_blades_under_the_lawn_height() {
        // a lawn, a meadow, tall grass
        for h in [0.045, MEADOW, TALL] {
            set_height(h);
            let m = tile();
            assert!(m.tri_count() > 40_000);
            assert_eq!(m.materials.len() % 4, 0);
            assert!(m.materials.len() >= m.tri_count() as usize);
            let (lo, hi) = m.bounds();
            assert!(lo.y >= -1e-4 && hi.y <= height(), "{h}: {} .. {}", lo.y, hi.y);
            // blades lean over the tile's edge (the bake puts copies around it), but not far
            assert!(lo.x > -height() && hi.x < tile_size() + height());
            // the flattest view's rays end inside the copies the bake puts around the tile
            assert!(copies() as f32 * tile_size() >= height() * SLOPE_MAX);
        }
        set_height(0.045);
        assert!(AZIMUTHS.is_multiple_of(4) && SLOPES.is_multiple_of(4) && RES > 2 * GUTTER);
    }
}
