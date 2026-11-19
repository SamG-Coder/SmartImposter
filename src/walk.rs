//! Walking: a first-person camera on the ground (`--walk`, key V; the default in `--park`).
//!
//! The eye is carried at head height over the terrain: W A S D move along the ground the way the
//! camera faces, Shift runs, Ctrl crouches, Space jumps, and gravity brings it back down. Tree
//! trunks are solid (`Trunks`: a cylinder per tree, measured from its bark); everything else can
//! be walked through. Over water the eye stays just above the surface.
//!
//! The eye comes to rest exactly (the still image accumulates while nothing moves).

use crate::asset::{MeshData, TreeAsset, TreePack};
use crate::catalog::{Catalog, Kind};
use crate::world::World;
use glam::{Vec2, Vec3};
use std::collections::HashMap;

/// Eye height standing and crouching (m), and the body's radius against trunks.
pub const EYE: f32 = 1.7;
pub const CROUCH: f32 = 0.75;
const BODY: f32 = 0.3;
/// Speeds (m/s): a brisk walk, a run, crouched.
const WALK: f32 = 2.8;
const RUN: f32 = 7.0;
const SNEAK: f32 = 1.3;
const JUMP: f32 = 4.2;
const GRAVITY: f32 = 9.81;

/// The trunks of the scene's trees as vertical cylinders (x, z, radius), in a grid.
pub struct Trunks {
    grid: HashMap<(i32, i32), Vec<[f32; 3]>>,
    pub count: usize,
}

/// Grid cell (m): larger than any trunk plus the body, so a 3 x 3 neighbourhood is enough.
const CELL: f32 = 8.0;

/// The trunk of a tree at breast height, in the tree's own space: its centre (x, z) and radius,
/// from the bark between 0.4 and 1.4 m above its base. The radius is on the low side of the
/// bark's distances from the centre: low branches and root flares do not widen it.
fn trunk(tree: &TreeAsset, meshes: &[MeshData], bounds: &mut HashMap<u32, (Vec3, Vec3)>) -> Option<(Vec2, f32)> {
    let (lo, hi) = (tree.bmin.y + 0.4, tree.bmin.y + 1.4);
    let mut pts: Vec<Vec2> = Vec::new();
    for part in &tree.parts {
        let m = &meshes[part.mesh as usize];
        // (the part's box first: most parts are branches and leaves far above)
        let (b0, b1) = *bounds.entry(part.mesh).or_insert_with(|| m.bounds());
        let ys = (0..8).map(|k| part.xform.transform_point3(Vec3::new(if k & 1 == 0 { b0.x } else { b1.x }, if k & 2 == 0 { b0.y } else { b1.y }, if k & 4 == 0 { b0.z } else { b1.z })).y);
        let (y0, y1) = ys.fold((f32::MAX, f32::MIN), |(a, b), y| (a.min(y), b.max(y)));
        if y1 < lo || y0 > hi {
            continue;
        }
        for (t, tri) in m.indices.chunks_exact(3).enumerate() {
            if m.materials.get(t).is_some_and(|b| b & 3 != 0) {
                continue; // bark only
            }
            let v = tri[0] as usize * 3;
            let p = part.xform.transform_point3(Vec3::new(m.positions[v], m.positions[v + 1], m.positions[v + 2]));
            if p.y >= lo && p.y <= hi {
                pts.push(Vec2::new(p.x, p.z));
            }
        }
    }
    if pts.len() < 3 {
        return None;
    }
    let median = |mut v: Vec<f32>, at: f32| -> f32 {
        v.sort_by(f32::total_cmp);
        v[((v.len() - 1) as f32 * at) as usize]
    };
    let centre = Vec2::new(median(pts.iter().map(|p| p.x).collect(), 0.5), median(pts.iter().map(|p| p.y).collect(), 0.5));
    let radius = median(pts.iter().map(|p| p.distance(centre)).collect(), 0.4);
    Some((centre, radius.clamp(0.05, 1.5)))
}

impl Trunks {
    pub fn none() -> Self {
        Trunks { grid: HashMap::new(), count: 0 }
    }

    /// The trunks of every tree (4 m and taller) of the world.
    pub fn build(world: &World, pack: &TreePack, cat: &Catalog) -> Self {
        let mut per_variant: Vec<Option<Option<(Vec2, f32)>>> = vec![None; pack.trees.len()];
        let mut bounds = HashMap::new();
        let mut out = Trunks::none();
        for t in &world.trees {
            let v = t.variant as usize;
            if v >= cat.variants.len() || cat.variants[v].kind != Kind::Tree || cat.variants[v].height < 4.0 {
                continue;
            }
            let Some((centre, radius)) = *per_variant[v].get_or_insert_with(|| trunk(&pack.trees[v], &pack.meshes, &mut bounds)) else { continue };
            let p = t.xform.transform_point3(Vec3::new(centre.x, 0.0, centre.y));
            out.add(p.x, p.z, radius * t.xform.matrix3.x_axis.length());
        }
        out
    }

    pub fn add(&mut self, x: f32, z: f32, radius: f32) {
        self.grid.entry(((x / CELL).floor() as i32, (z / CELL).floor() as i32)).or_default().push([x, z, radius]);
        self.count += 1;
    }

    /// Moves a body of radius `body` at (x, z) out of the trunks it stands in.
    pub fn push_out(&self, p: &mut Vec2, body: f32) {
        if self.count == 0 {
            return;
        }
        for _ in 0..2 {
            let (ci, cj) = ((p.x / CELL).floor() as i32, (p.y / CELL).floor() as i32);
            for j in cj - 1..=cj + 1 {
                for i in ci - 1..=ci + 1 {
                    for t in self.grid.get(&(i, j)).map_or(&[][..], |v| v.as_slice()) {
                        let c = Vec2::new(t[0], t[1]);
                        let (d, keep) = (*p - c, t[2] + body);
                        if d.length_squared() < keep * keep {
                            *p = c + if d.length_squared() > 1e-8 { d.normalize() } else { Vec2::X } * keep;
                        }
                    }
                }
            }
        }
    }
}

/// What the keys ask for this frame.
#[derive(Clone, Copy, Default)]
pub struct Input {
    pub forward: f32,
    pub right: f32,
    pub jump: bool,
    pub run: bool,
    pub crouch: bool,
}

pub struct Walk {
    /// walking (else the free fly camera)
    pub on: bool,
    /// the eye's height above the feet now (between CROUCH and EYE)
    eye: f32,
    grounded: bool,
}

impl Walk {
    pub fn new(on: bool) -> Self {
        Walk { on, eye: EYE, grounded: false }
    }

    /// One time step. `pos` is the eye, `vel` its velocity, `yaw` the camera's heading;
    /// `ground(x, z)` gives the height of the ground and of the surface there (the water's over
    /// water, else the ground again).
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, ground: impl Fn(f32, f32) -> (f32, f32), trunks: &Trunks, pos: &mut Vec3, vel: &mut Vec3, yaw: f32, input: Input, dt: f32) {
        let near = |rate: f32| 1.0 - (-rate * dt).exp();
        // ---- along the ground
        let (fwd, right) = (Vec2::new(yaw.sin(), -yaw.cos()), Vec2::new(yaw.cos(), yaw.sin()));
        let dir = (fwd * input.forward + right * input.right).normalize_or_zero();
        let speed = if input.crouch { SNEAK } else if input.run { RUN } else { WALK };
        let mut v = Vec2::new(vel.x, vel.z);
        // (quick on the feet, little steering in the air)
        v += (dir * speed - v) * near(if self.grounded { 11.0 } else { 1.5 });
        if dir == Vec2::ZERO && self.grounded && v.length() < 0.02 {
            v = Vec2::ZERO;
        }
        let mut p = Vec2::new(pos.x, pos.z) + v * dt;
        let before = p;
        trunks.push_out(&mut p, BODY);
        if p != before && dt > 0.0 {
            // (a trunk in the way takes the speed into it, the rest slides past)
            let n = (p - before).normalize_or_zero();
            v -= n * v.dot(n).min(0.0);
        }
        (pos.x, pos.z) = (p.x, p.y);
        (vel.x, vel.z) = (v.x, v.y);

        // ---- up and down
        let want = if input.crouch { CROUCH } else { EYE };
        self.eye += (want - self.eye) * near(10.0);
        if (want - self.eye).abs() < 1e-3 {
            self.eye = want;
        }
        let (bed, surface) = ground(pos.x, pos.z);
        // (wading: the eye stays above the water)
        let floor = (bed + self.eye).max(surface + 0.3);
        if self.grounded && input.jump && !input.crouch {
            vel.y = JUMP;
            self.grounded = false;
        }
        if self.grounded {
            // the eye follows the ground (smoothly over the terrain's facets) and comes to rest
            vel.y = 0.0;
            pos.y += (floor - pos.y) * near(16.0);
            if (floor - pos.y).abs() < 1e-4 {
                pos.y = floor;
            }
        } else {
            vel.y -= GRAVITY * dt;
            pos.y += vel.y * dt;
            if pos.y <= floor && vel.y <= 0.0 {
                // landed (or started below the ground: lifted onto it)
                self.grounded = true;
                vel.y = 0.0;
                if pos.y < floor - 0.5 {
                    pos.y = floor;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(w: &mut Walk, trunks: &Trunks, pos: &mut Vec3, vel: &mut Vec3, input: Input, secs: f32) {
        for _ in 0..(secs * 120.0) as u32 {
            w.step(|x, _| (0.1 * x, 0.1 * x), trunks, pos, vel, 0.0, input, 1.0 / 120.0);
        }
    }

    #[test]
    fn falls_walks_on_the_ground_and_comes_to_rest() {
        let (mut w, trunks) = (Walk::new(true), Trunks::none());
        let (mut pos, mut vel) = (Vec3::new(0.0, 12.0, 0.0), Vec3::ZERO);
        run(&mut w, &trunks, &mut pos, &mut vel, Input::default(), 3.0);
        assert!((pos.y - EYE).abs() < 1e-3, "landed at eye height, got {}", pos.y);
        // yaw 0 looks along -z; strafing right goes up the slope along +x
        run(&mut w, &trunks, &mut pos, &mut vel, Input { forward: 1.0, right: 1.0, ..Default::default() }, 2.0);
        assert!(pos.z < -3.0 && pos.x > 3.0);
        assert!((pos.y - (0.1 * pos.x + EYE)).abs() < 0.05, "follows the slope");
        run(&mut w, &trunks, &mut pos, &mut vel, Input::default(), 2.0);
        let rest = pos;
        run(&mut w, &trunks, &mut pos, &mut vel, Input::default(), 0.5);
        assert!(vel == Vec3::ZERO && pos == rest, "at rest exactly");
        assert_eq!(pos.y, 0.1 * pos.x + EYE);
    }

    #[test]
    fn jumps_and_crouches() {
        let (mut w, trunks) = (Walk::new(true), Trunks::none());
        let (mut pos, mut vel) = (Vec3::new(0.0, EYE, 0.0), Vec3::ZERO);
        run(&mut w, &trunks, &mut pos, &mut vel, Input::default(), 0.2);
        run(&mut w, &trunks, &mut pos, &mut vel, Input { jump: true, ..Default::default() }, 0.3);
        assert!(pos.y > EYE + 0.5, "in the air");
        run(&mut w, &trunks, &mut pos, &mut vel, Input::default(), 2.0);
        assert_eq!(pos.y, EYE);
        run(&mut w, &trunks, &mut pos, &mut vel, Input { crouch: true, ..Default::default() }, 2.0);
        assert_eq!(pos.y, CROUCH);
    }

    #[test]
    fn trunks_are_solid() {
        let mut trunks = Trunks::none();
        trunks.add(0.0, -5.0, 0.4);
        let (mut w, mut pos, mut vel) = (Walk::new(true), Vec3::new(0.05, EYE, 0.0), Vec3::ZERO);
        // walking straight at it (a little off centre): never inside, and past it in the end
        for _ in 0..1200 {
            w.step(|_, _| (0.0, 0.0), &trunks, &mut pos, &mut vel, 0.0, Input { forward: 1.0, ..Default::default() }, 1.0 / 120.0);
            assert!(Vec2::new(pos.x, pos.z + 5.0).length() >= 0.4 + BODY - 1e-4);
        }
        assert!(pos.z < -8.0, "slid round the trunk, at z {}", pos.z);
    }
}
