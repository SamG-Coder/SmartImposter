//! Looping showcase camera path: closed Catmull-Rom spline over (x, z, height above terrain),
//! re-parameterised by arc length so the camera moves at a constant speed.

use crate::world::World;
use glam::{Vec2, Vec3};

pub struct FlyPath {
    keys: Vec<Vec3>, // x, z, height above terrain
    table: Vec<(f32, f32)>, // (arc length, spline parameter)
    pub length: f32,
}

fn catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

impl FlyPath {
    pub fn showcase() -> Self {
        // (x, z, metres above terrain). Starts in the clearing looking down -Z.
        let keys = vec![
            // clearing -> climb out north
            Vec3::new(0.0, 18.0, 2.5),
            Vec3::new(-3.0, -2.0, 3.5),
            Vec3::new(3.0, -16.0, 13.0),
            // skim the canopy of the forest band north of the clearing
            Vec3::new(-10.0, -70.0, 24.0),
            Vec3::new(-30.0, -170.0, 24.0),
            Vec3::new(-40.0, -270.0, 27.0),
            // rise for a vista and turn west over the big forest
            Vec3::new(-100.0, -360.0, 42.0),
            Vec3::new(-230.0, -410.0, 60.0),
            Vec3::new(-390.0, -360.0, 48.0),
            // sweep south along the western forest at canopy height
            Vec3::new(-520.0, -250.0, 28.0),
            Vec3::new(-570.0, -120.0, 24.0),
            // drop out over the meadow, low
            Vec3::new(-520.0, 10.0, 12.0),
            Vec3::new(-340.0, 95.0, 6.0),
            Vec3::new(-200.0, 110.0, 12.0),
            // climb over the last band of trees and drop into the clearing
            Vec3::new(-110.0, 70.0, 30.0),
            Vec3::new(-45.0, 45.0, 32.0),
            Vec3::new(-6.0, 11.0, 9.0),
        ];
        let mut p = FlyPath { keys, table: Vec::new(), length: 0.0 };
        // Arc-length table measured on the horizontal plane + height.
        let n = p.keys.len() as f32;
        let samples = 4000;
        let mut prev = p.raw(0.0);
        let mut acc = 0.0;
        p.table.push((0.0, 0.0));
        for i in 1..=samples {
            let u = i as f32 / samples as f32 * n;
            let cur = p.raw(u);
            acc += cur.distance(prev);
            p.table.push((acc, u));
            prev = cur;
        }
        p.length = acc;
        p
    }

    fn raw(&self, u: f32) -> Vec3 {
        let n = self.keys.len();
        let i = (u.floor() as usize) % n;
        let t = u - u.floor();
        let k = |j: isize| self.keys[((i as isize + j).rem_euclid(n as isize)) as usize];
        catmull(k(-1), k(0), k(1), k(2), t)
    }

    fn param_at(&self, dist: f32) -> f32 {
        let d = dist.rem_euclid(self.length);
        let idx = self.table.partition_point(|&(l, _)| l < d).clamp(1, self.table.len() - 1);
        let (l0, u0) = self.table[idx - 1];
        let (l1, u1) = self.table[idx];
        u0 + (u1 - u0) * ((d - l0) / (l1 - l0).max(1e-6))
    }

    /// World-space camera position at arc length `dist` (wraps around).
    pub fn position(&self, world: &World, dist: f32) -> Vec3 {
        let p = self.raw(self.param_at(dist));
        Vec3::new(p.x, world.height_at(p.x, p.y) + p.z, p.y)
    }

    /// Position and smoothed look direction (looking at a point further along the path).
    pub fn camera(&self, world: &World, dist: f32) -> (Vec3, f32, f32) {
        let pos = self.position(world, dist);
        // Average a few look-ahead points for a steady gaze, aimed slightly down.
        let mut target = Vec3::ZERO;
        for k in 1..=4 {
            target += self.position(world, dist + 10.0 * k as f32);
        }
        target /= 4.0;
        target.y -= 4.0;
        let fwd = (target - pos).normalize_or_zero();
        let yaw = fwd.x.atan2(-fwd.z);
        // Never look at the sky while climbing; never stare at the ground either.
        let pitch = fwd.y.clamp(-0.99, 0.99).asin().clamp(-0.42, 0.03);
        (pos, yaw, pitch)
    }

    /// Report places where the camera passes through a tree crown.
    pub fn check(&self, world: &World) -> Vec<(f32, Vec3)> {
        let mut hits = Vec::new();
        let mut d = 0.0;
        while d < self.length {
            let p = self.position(world, d);
            for t in &world.trees {
                let base = t.xform.translation;
                let horiz = Vec2::new(p.x - base.x, p.z - base.z).length();
                let top = base.y + (t.center.y - base.y) * 2.0;
                if horiz < 3.5 && p.y > base.y && p.y < top + 0.5 {
                    hits.push((d, p));
                    break;
                }
            }
            d += 2.0;
        }
        hits
    }
}
