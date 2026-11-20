//! `--battle`, out of the aircraft (E): under a parachute, then standing on the ground. The view
//! is the pilot's own eyes, and in the pilot's hands is a gun with six turning barrels to fire at
//! the aircraft and the drones of the other squadrons. The gun and the parachute are ray traced
//! like everything else (aircraft.rs `extras`); the rounds are the battle's tracers.

use crate::aircraft::{Craft, FxGpu, Pose, KIND_BARRELS, KIND_CANOPY, KIND_MINIGUN};
use crate::battle::Battle;
use crate::gamepad::Pad;
use crate::pilot::Camera;
use glam::{Affine3A, DMat3, DVec3, Mat3};
use std::collections::HashSet;
use winit::keyboard::KeyCode;

/// rounds a second, and how fast the barrels turn to fire them (rad/s)
const RATE: f64 = 50.0;
const SPIN: f64 = 52.0;
/// the eye above the ground, standing (m)
const EYE: f64 = 1.7;

pub struct Foot {
    /// the pilot's eye (Earth axes) and how it moves
    pub pos: DVec3,
    pub vel: DVec3,
    /// which way the pilot looks: the heading from north, and up from level (rad)
    yaw: f64,
    pitch: f64,
    /// how far the parachute has opened (0..1); down on the ground, it is gone
    chute: f64,
    pub landed: bool,
    /// the trigger (the left mouse button)
    pub trigger: bool,
    time: f64,
    /// the barrels: their angle, how fast they are turning, and rounds owed
    angle: f64,
    turning: f64,
    owed: f64,
    pub fired: u64,
    firing: bool,
    /// the canopy keeps its own heading (it does not turn with the head)
    canopy: f64,
    prev: [Option<(DMat3, DVec3)>; 3],
}

impl Foot {
    /// Out of an aircraft at `pos`, going at `vel`.
    pub fn new(pos: DVec3, vel: DVec3) -> Self {
        let up = pos.normalize();
        let east = DVec3::Z.cross(up).normalize_or(DVec3::X);
        let north = up.cross(east);
        let yaw = vel.dot(east).atan2(vel.dot(north));
        // (thrown clear of it, upwards)
        Foot { pos: pos + up * 3.0, vel: vel + up * 18.0, yaw, pitch: -0.15, chute: 0.0, landed: false, trigger: false, time: 0.0, angle: 0.0, turning: 0.0, owed: 0.0, fired: 0, firing: false, canopy: yaw, prev: [None; 3] }
    }

    /// A testing aid (SI_BATTLE look=, lookup=, shoot=): look that many degrees round from the
    /// way the aircraft was going and that many up, and hold the trigger.
    pub fn script(&mut self, round: f64, lift: f64, shoot: bool) {
        self.yaw += round.to_radians();
        self.pitch = lift.to_radians().clamp(-1.45, 1.45);
        self.trigger = shoot;
    }

    pub fn mouse(&mut self, dx: f64, dy: f64) {
        self.yaw += dx * 0.0022;
        self.pitch = (self.pitch - dy * 0.0022).clamp(-1.45, 1.45);
    }

    /// Level ahead, to the right, up, and where the eyes point.
    fn axes(&self) -> (DVec3, DVec3, DVec3, DVec3) {
        let up = self.pos.normalize();
        let east = DVec3::Z.cross(up).normalize_or(DVec3::X);
        let north = up.cross(east);
        let ahead = north * self.yaw.cos() + east * self.yaw.sin();
        (ahead, ahead.cross(up), up, ahead * self.pitch.cos() + up * self.pitch.sin())
    }

    /// A step: the parachute opens and comes down (W A S D steer it a little), or, landed, the
    /// pilot walks; the mouse or the right stick looks about. Where the eyes are.
    pub fn update(&mut self, keys: &HashSet<KeyCode>, pad: &Pad, fire: bool, dt: f64, ground: impl Fn(DVec3) -> f64) -> Camera {
        self.time += dt;
        self.yaw += pad.right.0 * 2.4 * dt;
        self.pitch = (self.pitch + pad.right.1 * 1.8 * dt).clamp(-1.45, 1.45);
        let k = |c: KeyCode| keys.contains(&c) as u8 as f64;
        let (ahead, right, up, _) = self.axes();
        let go = (ahead * (k(KeyCode::KeyW) - k(KeyCode::KeyS) + pad.left.1) + right * (k(KeyCode::KeyD) - k(KeyCode::KeyA) + pad.left.0)).clamp_length_max(1.0);
        if !self.landed {
            // it opens a second after leaving the aircraft, and takes a second to fill
            if self.time > 0.9 {
                self.chute = (self.chute + dt / 1.1).min(1.0);
            }
            let drag = 0.003 + 0.22 * self.chute * self.chute;
            self.vel += (-up * 9.81 - self.vel * self.vel.length() * drag + go * 6.0 * self.chute) * dt;
            self.pos += self.vel * dt;
            let g = ground(self.pos);
            if self.pos.length() < g + EYE {
                self.pos = self.pos.normalize() * (g + EYE);
                self.vel = DVec3::ZERO;
                self.landed = true;
                println!("Down. W A S D to walk; E or R: into another aircraft.");
            }
        } else {
            let pace = if keys.contains(&KeyCode::ShiftLeft) { 9.0 } else { 5.0 };
            self.vel = go * pace;
            let to = self.pos + self.vel * dt;
            self.pos = to.normalize() * (ground(to) + EYE);
        }
        // the barrels spin up before the first round goes, and run down after the last
        self.firing = fire || self.trigger;
        let want = if self.firing { SPIN } else { 0.0 };
        self.turning += (want - self.turning).clamp(-70.0 * dt, 160.0 * dt);
        self.angle = (self.angle + self.turning * dt) % std::f64::consts::TAU;
        let (_, right, up, look) = self.axes();
        // (the gun shakes the view a little)
        let shake = if self.firing && self.turning > 0.8 * SPIN { 0.0025 } else { 0.0 };
        let look = (look + right * shake * (self.time * 173.0).sin() + up * shake * (self.time * 211.0).cos()).normalize();
        let r = look.cross(up).normalize_or(right);
        Camera { pos: self.pos, right: r, up: r.cross(look), fwd: look, fov_y: 74f32.to_radians() }
    }

    /// The gun's frame (its barrels along -z) and where it is held: low and to the right, its
    /// barrels a little inwards, towards what the eyes are on.
    fn gun(&self) -> (DMat3, DVec3) {
        let (_, right, up, look) = self.axes();
        let r = look.cross(up).normalize_or(right);
        let u = r.cross(look);
        let rot = DMat3::from_cols(r, u, -look) * DMat3::from_rotation_y(0.055) * DMat3::from_rotation_x(-0.035);
        let kick = if self.firing && self.turning > 0.8 * SPIN { 0.006 * (self.time * 311.0).sin().abs() } else { 0.0 };
        (rot, self.pos + r * 0.19 - u * 0.225 + look * (0.52 - kick))
    }

    /// The rounds this step's firing is good for go out of the muzzle (battle.rs `round`).
    pub fn shoot(&mut self, battle: &mut Battle, dt: f64) {
        if !self.firing || self.turning < 0.8 * SPIN {
            self.owed = 0.0;
            return;
        }
        let (rot, at) = self.gun();
        let (_, _, _, look) = self.axes();
        let muzzle = at - rot.z_axis * 1.0;
        self.owed += RATE * dt;
        while self.owed >= 1.0 {
            self.owed -= 1.0;
            self.fired += 1;
            // (it goes where the eyes are on, a long way out: not along the gun's own line)
            let to = (self.pos + look * 600.0 - muzzle).normalize();
            battle.round(muzzle, to, self.vel);
        }
    }

    /// The gun, the parachute and the flash at the muzzle, for this frame's picture (`frame`:
    /// Earth axes -> the renderer's frame at the camera, `cam` the camera's Earth position).
    pub fn draw(&mut self, frame: DMat3, cam: DVec3, crafts: &mut Vec<Craft>, fx: &mut Vec<FxGpu>) {
        let to = |p: DVec3| (frame * (p - cam)).as_vec3();
        let place = |r: DMat3, p: DVec3| {
            let m = frame * r;
            Affine3A::from_mat3_translation(Mat3::from_cols(m.x_axis.as_vec3(), m.y_axis.as_vec3(), m.z_axis.as_vec3()), to(p))
        };
        let mut put = |n: usize, kind: u8, rot: DMat3, pos: DVec3, prev: &mut [Option<(DMat3, DVec3)>; 3]| {
            let xform = place(rot, pos);
            let (pr, pp) = prev[n].unwrap_or((rot, pos));
            prev[n] = Some((rot, pos));
            let pose = Pose { xform, back: place(pr, pp) * xform.inverse(), lef: 0.0, flaperon: [0.0; 2], tail: [0.0; 2], rudder: 0.0 };
            crafts.push(Craft { pose, livery: 0, kind, tint: [1.0; 3] });
        };
        let (rot, at) = self.gun();
        let mut prev = self.prev;
        put(0, KIND_MINIGUN, rot, at, &mut prev);
        put(1, KIND_BARRELS, rot * DMat3::from_rotation_z(self.angle), at, &mut prev);
        let (_, _, up, look) = self.axes();
        if !self.landed && self.chute > 0.02 {
            // the canopy overhead, filling as it opens, swinging a little
            let east = DVec3::Z.cross(up).normalize_or(DVec3::X);
            let north = up.cross(east);
            let ahead = north * self.canopy.cos() + east * self.canopy.sin();
            let lean = (up + ahead * 0.05 * (self.time * 0.9).sin() + ahead.cross(up) * 0.04 * (self.time * 0.7).cos()).normalize();
            let side = ahead.cross(lean).normalize();
            let c = self.chute;
            let rot = DMat3::from_cols(side * c, lean * (0.45 + 0.55 * c), side.cross(lean) * c);
            put(2, KIND_CANOPY, rot, self.pos, &mut prev);
        }
        self.prev = prev;
        if self.firing && self.turning > 0.8 * SPIN {
            // the flash: never the same two frames running
            let muzzle = to(at - rot.z_axis * 1.0);
            let ahead = to(at - rot.z_axis * (1.28 + 0.2 * (self.time * 97.0).sin()));
            let flick = (0.6 + 0.4 * (self.time * 263.0).sin().abs()) as f32;
            let glow = |a: glam::Vec3, ra: f32, b: glam::Vec3, rb: f32, bright: f32| FxGpu { a: [a.x, a.y, a.z, ra], b: [b.x, b.y, b.z, rb], colour: [bright, 0.62 * bright, 0.22 * bright, 14.0], kind: [1.0, 0.0, 0.0, 0.0] };
            fx.push(glow(muzzle, 0.035, ahead, 0.012, 7.0 * flick));
            fx.push(glow(muzzle, 0.07 * flick, muzzle, 0.07 * flick, 4.0 * flick));
        }
        // a dot where the rounds go: there is nothing else to aim by
        let dot = to(self.pos + look * 40.0);
        fx.push(FxGpu { a: [dot.x, dot.y, dot.z, 0.04], b: [dot.x, dot.y, dot.z, 0.04], colour: [0.2, 2.6, 0.25, 30.0], kind: [1.0, 0.0, 0.0, 0.0] });
    }

    /// For the window's title.
    pub fn hud(&self, ground: f64) -> String {
        let above = self.pos.length() - ground - EYE;
        match self.landed {
            true => format!("On foot | {} rounds fired | E or R: into another aircraft", self.fired),
            false => format!("Parachute, {:.0} m up, {:.0} m/s down | {} rounds fired", above.max(0.0), -self.vel.dot(self.pos.normalize()), self.fired),
        }
    }
}
