//! Cube-sphere quadtree: tile ids, cube -> sphere mapping, per-tile terrain meshes and the
//! per-frame selection of which tiles to draw.
//!
//! Coordinates are Earth-centred (ECEF, float64): +X at latitude 0 / longitude 0, +Y at
//! longitude 90 E, +Z the north pole. Tile meshes are float32 relative to the tile centre.

use super::dem::Terrain;
use glam::DVec3;
use std::f64::consts::PI;

pub const R_EARTH: f64 = 6_371_000.0;
/// Vertices per tile side (32 x 32 cells).
pub const GRID: usize = 33;
/// Finest level over ETOPO only: ~610 m tiles, ~19 m cells (the data is ~1.85 km per sample).
pub const MAX_LEVEL: u8 = 14;
/// Finest level over the 30 m data: ~150 m tiles, ~4.8 m cells.
pub const MAX_LEVEL_FINE: u8 = 16;
/// Split a tile while the camera is closer than this many tile edge lengths.
pub const SPLIT_K: f64 = 5.0;
/// Floats per vertex: position (3), normal (3), elevation (m), land cover (3 words of 4 unorm8:
/// tree shrub grass crop | built bare snow water | wetland moss has-data -), texture
/// coordinates in metres (2).
pub const VSTRIDE: usize = 12;
/// Texture coordinates repeat every TEX_PERIOD metres (a multiple of every texture repeat).
pub const TEX_PERIOD: f64 = 131_072.0;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, PartialOrd, Ord)]
pub struct TileId {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl TileId {
    pub fn root(face: u8) -> Self {
        TileId { face, level: 0, x: 0, y: 0 }
    }
    pub fn parent(&self) -> Option<TileId> {
        (self.level > 0).then(|| TileId { face: self.face, level: self.level - 1, x: self.x / 2, y: self.y / 2 })
    }
    pub fn children(&self) -> [TileId; 4] {
        let (l, x, y) = (self.level + 1, self.x * 2, self.y * 2);
        [
            TileId { face: self.face, level: l, x, y },
            TileId { face: self.face, level: l, x: x + 1, y },
            TileId { face: self.face, level: l, x, y: y + 1 },
            TileId { face: self.face, level: l, x: x + 1, y: y + 1 },
        ]
    }
    /// Cube-face coordinates of the tile's corner and its size, in [-1, 1].
    pub fn uv(&self) -> (f64, f64, f64) {
        let s = 2.0 / (1u64 << self.level) as f64;
        (-1.0 + self.x as f64 * s, -1.0 + self.y as f64 * s, s)
    }
    /// Tile edge length on the sphere (m).
    pub fn edge(&self) -> f64 {
        0.5 * PI * R_EARTH / (1u64 << self.level) as f64
    }
    pub fn center_dir(&self) -> DVec3 {
        let (u, v, s) = self.uv();
        face_dir(self.face, u + 0.5 * s, v + 0.5 * s)
    }
}

/// Cube face + (u, v) in [-1, 1] -> unit direction. Equal-angle warp (tan) keeps cells similar in
/// size across the face.
pub fn face_dir(face: u8, u: f64, v: f64) -> DVec3 {
    let (a, b) = ((u * PI / 4.0).tan(), (v * PI / 4.0).tan());
    let p = match face {
        0 => DVec3::new(1.0, a, b),
        1 => DVec3::new(-1.0, -a, b),
        2 => DVec3::new(-a, 1.0, b),
        3 => DVec3::new(a, -1.0, b),
        4 => DVec3::new(-b, a, 1.0),
        _ => DVec3::new(b, a, -1.0),
    };
    p.normalize()
}

pub fn lat_lon(d: DVec3) -> (f64, f64) {
    let d = d.normalize();
    (d.z.clamp(-1.0, 1.0).asin().to_degrees(), d.y.atan2(d.x).to_degrees())
}

pub fn dir_from_lat_lon(lat: f64, lon: f64) -> DVec3 {
    let (la, lo) = (lat.to_radians(), lon.to_radians());
    DVec3::new(la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin())
}

/// Ground radius at a direction: sea level over the oceans, terrain on land.
pub fn ground_radius(terrain: &Terrain, d: DVec3) -> f64 {
    let (lat, lon) = lat_lon(d);
    R_EARTH + terrain.height(lat, lon).max(0.0) as f64
}

/// Deepest level for a tile: finer where the 30 m data is.
pub fn max_level(id: TileId, terrain: &Terrain) -> u8 {
    if id.level < 8 {
        return MAX_LEVEL_FINE; // big tiles: decide further down
    }
    let (u, v, s) = id.uv();
    let mut b = [f64::MAX, f64::MIN, f64::MAX, f64::MIN];
    for (a, c) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)] {
        let (lat, lon) = lat_lon(face_dir(id.face, u + a * s, v + c * s));
        b = [b[0].min(lat), b[1].max(lat), b[2].min(lon), b[3].max(lon)];
    }
    if terrain.fine_in(b[0], b[1], b[2], b[3]) {
        MAX_LEVEL_FINE
    } else {
        MAX_LEVEL
    }
}

pub struct TileMesh {
    pub id: TileId,
    pub center: DVec3,
    /// GRID^2 surface vertices then 4 * GRID skirt vertices, VSTRIDE floats each.
    pub verts: Vec<f32>,
}

/// Build a tile: positions from ETOPO (oceans at sea level, flagged as water), normals from the
/// neighbouring samples (one sample beyond the edge, so they match across tiles), and skirts
/// hanging below the edges that hide the cracks between tiles of different levels.
pub fn build_tile(id: TileId, terrain: &Terrain) -> TileMesh {
    let (u0, v0, s) = id.uv();
    let n = GRID;
    let step = s / (n - 1) as f64;
    let m = n + 2; // with a one-sample apron
    let mut pos = vec![DVec3::ZERO; m * m];
    let mut elev = vec![0f32; m * m];
    for j in 0..m {
        for i in 0..m {
            let d = face_dir(id.face, u0 + (i as f64 - 1.0) * step, v0 + (j as f64 - 1.0) * step);
            let (lat, lon) = lat_lon(d);
            let e = terrain.height(lat, lon);
            elev[j * m + i] = e;
            pos[j * m + i] = d * (R_EARTH + e.max(0.0) as f64);
        }
    }
    let center = id.center_dir() * R_EARTH;
    // land cover per vertex, averaged over the vertex's footprint
    let cell = id.edge() / (n - 1) as f64;
    let pack = |f: &[f32]| u32::from_le_bytes([(f[0] * 255.0 + 0.5) as u8, (f[1] * 255.0 + 0.5) as u8, (f[2] * 255.0 + 0.5) as u8, (f[3] * 255.0 + 0.5) as u8]);
    let mut cover = vec![[0u32; 3]; m * m];
    for j in 1..m - 1 {
        for i in 1..m - 1 {
            let c = j * m + i;
            let (lat, lon) = lat_lon(pos[c]);
            if let Some(f) = terrain.cover.fractions(lat, lon, cell) {
                cover[c] = [pack(&f[0..4]), pack(&f[4..8]), pack(&[f[8], f[9], 1.0, 0.0])];
            }
        }
    }
    // texture coordinates: arc length along the cube face (m), relative to a multiple of
    // TEX_PERIOD near the tile so they stay small and continuous within the tile
    let arc = std::f64::consts::FRAC_PI_4 * R_EARTH;
    let (bu, bv) = ((u0 * arc / TEX_PERIOD).floor() * TEX_PERIOD, (v0 * arc / TEX_PERIOD).floor() * TEX_PERIOD);
    let tex = |i: usize, j: usize| [((u0 + i as f64 * step) * arc - bu) as f32, ((v0 + j as f64 * step) * arc - bv) as f32];
    let mut verts = Vec::with_capacity((n * n + 4 * n) * VSTRIDE);
    let push = |verts: &mut Vec<f32>, p: DVec3, nrm: DVec3, e: f32, c: [u32; 3], uv: [f32; 2]| {
        let q = p - center;
        verts.extend_from_slice(&[
            q.x as f32,
            q.y as f32,
            q.z as f32,
            nrm.x as f32,
            nrm.y as f32,
            nrm.z as f32,
            e,
            f32::from_bits(c[0]),
            f32::from_bits(c[1]),
            f32::from_bits(c[2]),
            uv[0],
            uv[1],
        ]);
    };
    let mut normals = vec![DVec3::ZERO; n * n];
    for j in 0..n {
        for i in 0..n {
            let c = (j + 1) * m + i + 1;
            let du = pos[c + 1] - pos[c - 1];
            let dv = pos[c + m] - pos[c - m];
            let mut nrm = du.cross(dv).normalize();
            if nrm.dot(pos[c]) < 0.0 {
                nrm = -nrm;
            }
            normals[j * n + i] = nrm;
            push(&mut verts, pos[c], nrm, elev[c], cover[c], tex(i, j));
        }
    }
    // skirts: the edge vertices again, lowered
    let drop = id.edge() * 0.03 + 30.0;
    let edge_vertex = |k: usize, side: usize| -> (usize, usize) {
        match side {
            0 => (k, 0),
            1 => (n - 1, k),
            2 => (n - 1 - k, n - 1),
            _ => (0, n - 1 - k),
        }
    };
    for side in 0..4 {
        for k in 0..n {
            let (i, j) = edge_vertex(k, side);
            let c = (j + 1) * m + i + 1;
            let p = pos[c] - pos[c].normalize() * drop;
            push(&mut verts, p, normals[j * n + i], elev[c], cover[c], tex(i, j));
        }
    }
    TileMesh { id, center, verts }
}

/// Shared index buffer for every tile (same topology): the grid, then the four skirts.
pub fn index_buffer() -> Vec<u32> {
    let n = GRID as u32;
    let mut idx = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = j * n + i;
            let (b, c, d) = (a + 1, a + n, a + n + 1);
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let surf = |i: u32, j: u32| j * n + i;
    let edge = |k: u32, side: u32| -> u32 {
        match side {
            0 => surf(k, 0),
            1 => surf(n - 1, k),
            2 => surf(n - 1 - k, n - 1),
            _ => surf(0, n - 1 - k),
        }
    };
    for side in 0..4u32 {
        let base = n * n + side * n;
        for k in 0..n - 1 {
            let (a, b) = (edge(k, side), edge(k + 1, side));
            let (sa, sb) = (base + k, base + k + 1);
            idx.extend_from_slice(&[a, sa, b, b, sa, sb]);
        }
    }
    idx
}

/// Tile selection for one frame. `resident(id)` tells whether a tile is on the GPU. Returns the
/// tiles to draw (each covering its area once) and the missing tiles to request, nearest first.
pub fn select(cam: DVec3, resident: &dyn Fn(TileId) -> bool, terrain: &Terrain) -> (Vec<TileId>, Vec<(f64, TileId)>) {
    let mut draw = Vec::new();
    let mut want = Vec::new();
    for f in 0..6 {
        visit(TileId::root(f), cam, resident, terrain, &mut draw, &mut want);
    }
    want.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (draw, want)
}

pub fn tile_distance(id: TileId, cam: DVec3) -> f64 {
    let c = id.center_dir() * R_EARTH;
    let bound = id.edge() * 0.75 + 9_000.0f64.min(id.edge());
    ((cam - c).length() - bound).max(0.0)
}

/// Behind the planet (over the horizon, with room for mountains sticking up beyond it)?
pub fn over_horizon(id: TileId, cam: DVec3) -> bool {
    let r = cam.length();
    if r <= R_EARTH + 100.0 {
        return false;
    }
    let theta = cam.normalize().dot(id.center_dir()).clamp(-1.0, 1.0).acos();
    let alpha = id.edge() * 0.75 / R_EARTH;
    let horizon = ((R_EARTH - 500.0) / r).clamp(-1.0, 1.0).acos() + (R_EARTH / (R_EARTH + 9_000.0)).acos();
    theta - alpha > horizon
}

fn visit(id: TileId, cam: DVec3, resident: &dyn Fn(TileId) -> bool, terrain: &Terrain, draw: &mut Vec<TileId>, want: &mut Vec<(f64, TileId)>) {
    if id.level > 1 && over_horizon(id, cam) {
        return;
    }
    let d = tile_distance(id, cam);
    let split = d < SPLIT_K * id.edge() && id.level < MAX_LEVEL_FINE && id.level < max_level(id, terrain);
    if split {
        let kids = id.children();
        let missing: Vec<TileId> = kids.iter().copied().filter(|k| !resident(*k)).collect();
        if missing.is_empty() {
            for k in kids {
                visit(k, cam, resident, terrain, draw, want);
            }
            return;
        }
        for k in missing {
            want.push((tile_distance(k, cam), k));
        }
    }
    if resident(id) {
        draw.push(id);
    } else {
        want.push((d, id));
    }
}
