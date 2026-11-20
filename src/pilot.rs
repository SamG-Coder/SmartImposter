//! `--test-flight`: flying the aircraft (flight.rs) over the globe. Keys to stick, pedals and
//! throttle; the cameras (chase, cockpit, fly-by); and the wake it leaves, kept for the shaders
//! (the hole it flies through cloud, its condensation trail).

use crate::aircraft::Pose;
use crate::flight::{Controls, Sim, MIL};
use crate::gamepad::Pad;
use crate::renderer::{Globals, WAKE_POINTS};
use glam::{Affine3A, DMat3, DVec3, Mat3};
use std::collections::{HashSet, VecDeque};
use winit::keyboard::KeyCode;

#[derive(Clone, Copy, PartialEq)]
pub enum View {
    Chase,
    Cockpit,
    FlyBy,
}

/// This frame's camera, in Earth axes.
pub struct Camera {
    pub pos: DVec3,
    pub right: DVec3,
    pub up: DVec3,
    pub fwd: DVec3,
    pub fov_y: f32,
}

struct WakePoint {
    pos: DVec3,
    time: f64,
    /// sample number (one every WAKE_STEP seconds)
    k: u64,
    contrail: f32,
    stir: f32,
}

/// seconds between the wake's samples; the newest FINE of them are all kept, older ones one in COARSE
const WAKE_STEP: f64 = 0.25;
const FINE: u64 = 11;
const COARSE: u64 = 18;
const WAKE_LIFE: f64 = 95.0;
/// the pilot's eye and the nozzle, model frame (x right, y up, z aft of the centre of gravity)
const EYE: DVec3 = DVec3::new(0.0, 1.03, -4.12);
const NOZZLE: DVec3 = DVec3::new(0.0, 0.085, 5.9);

pub struct Pilot {
    pub sim: Sim,
    stick: Controls,
    /// what the keys and the mouse ask of the stick and the pedals (pitch, roll, yaw), eased: a
    /// key is all or nothing
    hands: (f64, f64, f64),
    pub view: View,
    /// the chase camera's own forward and up: it turns after the aircraft with a lag, and keeps
    /// the horizon nearly level (it takes on a little of the aircraft's bank)
    cam_fwd: DVec3,
    cam_up: DVec3,
    /// looking round while the right mouse button is held (rad): about the aircraft in the chase
    /// view, the head in the cockpit. Let go, the view goes back behind the aircraft.
    look: (f64, f64),
    dist: f64,
    flyby_at: Option<(DVec3, f64)>,
    wake: VecDeque<WakePoint>,
    time: f64,
    samples: u64,
    prev: Option<(DMat3, DVec3)>,
    was_super: bool,
    /// the left mouse button is held: the mouse is the stick (pitch and roll)
    pub mouse_fly: bool,
    mouse_stick: (f64, f64),
    /// seconds since the mouse last moved the view: it goes back behind the aircraft after a while
    look_idle: f64,
    /// the aircraft is banked because A / D asked for a turn (it levels itself again after)
    auto_bank: bool,
    /// SI_FLIGHT (a testing aid): controls held without the keys, for scripted captures
    hold: Option<Controls>,
    /// `--battle`: hitting the ground does not put the aircraft back at the start; it is `down`,
    /// and the battle puts the pilot in another one (take_over)
    pub battle: bool,
    pub down: bool,
}

impl Pilot {
    pub fn new(pos: DVec3, heading: f64, speed: f64) -> Self {
        let sim = Sim::new(pos, heading, speed);
        let (cam_fwd, cam_up) = (sim.att * DVec3::X, pos.normalize());
        let stick = Controls { throttle: 0.6, ..Default::default() };
        let mut p = Pilot { sim, stick, hands: (0.0, 0.0, 0.0), view: View::Chase, cam_fwd, cam_up, look: (0.0, 0.0), dist: 19.0, flyby_at: None, wake: VecDeque::new(), time: 0.0, samples: 0, prev: None, was_super: false, mouse_fly: false, mouse_stick: (0.0, 0.0), look_idle: 9.0, auto_bank: false, hold: None, battle: false, down: false };
        p.sim.arcade = true;
        // SI_FLIGHT="throttle=1,pitch=0.4,roll=0,yaw=0,brake=0,view=flyby,look=1.2,dist=30": hold these
        if let Ok(script) = std::env::var("SI_FLIGHT") {
            let mut c = Controls { throttle: 0.6, ..Default::default() };
            for (k, v) in script.split(',').filter_map(|kv| kv.split_once('=')) {
                let x: f64 = v.parse().unwrap_or(0.0);
                match k.trim() {
                    "throttle" => c.throttle = x,
                    "pitch" => c.pitch = x,
                    "roll" => c.roll = x,
                    "yaw" => c.yaw = x,
                    "brake" => c.brake = x > 0.5,
                    "look" => p.look.0 = x,
                    "lookup" => p.look.1 = x,
                    "dist" => p.dist = x,
                    "view" => p.view = match v.trim() { "cockpit" => View::Cockpit, "flyby" => View::FlyBy, _ => View::Chase },
                    _ => {}
                }
            }
            p.sim.power = c.throttle;
            p.stick.throttle = c.throttle;
            p.hold = Some(c);
        }
        p
    }

    pub fn next_view(&mut self) {
        self.view = match self.view {
            View::Chase => View::Cockpit,
            View::Cockpit => View::FlyBy,
            View::FlyBy => View::Chase,
        };
        self.look = (0.0, 0.0);
        self.flyby_at = None;
        println!("View: {}", ["chase", "cockpit", "fly-by"][self.view as usize]);
    }

    pub fn reset(&mut self) {
        self.sim.reset();
        (self.cam_fwd, self.cam_up) = (self.sim.att * DVec3::X, self.sim.pos.normalize());
        self.stick = Controls { throttle: 0.6, ..Default::default() };
        self.hands = (0.0, 0.0, 0.0);
        self.auto_bank = false;
        self.was_super = false;
        self.wake.clear();
        self.prev = None;
        self.flyby_at = None;
    }

    /// `--battle`: the pilot is in another aircraft now, at `pos`, level on `heading` (degrees
    /// from north) at `speed` m/s.
    pub fn take_over(&mut self, pos: DVec3, heading: f64, speed: f64) {
        self.sim = Sim::new(pos, heading, speed);
        self.sim.arcade = true;
        self.sim.power = 0.8;
        self.reset();
        self.stick.throttle = 0.8;
        self.look = (0.0, 0.0);
        self.mouse_stick = (0.0, 0.0);
        self.down = false;
    }

    /// The mouse moved: it turns the view about the aircraft, or, with the left button held, it is
    /// the stick (forward: nose down; sideways: roll).
    pub fn mouse(&mut self, dx: f64, dy: f64) {
        if self.mouse_fly {
            self.mouse_stick.0 = (self.mouse_stick.0 + dx * 0.002).clamp(-1.0, 1.0);
            self.mouse_stick.1 = (self.mouse_stick.1 + dy * 0.002).clamp(-1.0, 1.0);
        } else {
            self.look.0 += dx * 0.0018;
            self.look.1 = (self.look.1 - dy * 0.0018).clamp(-1.4, 1.4);
            self.look_idle = 0.0;
        }
    }

    pub fn zoom(&mut self, steps: f64) {
        self.dist = (self.dist * 0.88f64.powf(steps)).clamp(12.0, 400.0);
    }

    /// The keys held, a time step: fly, and say where the camera is. The keys are Grand Theft
    /// Auto V's for aircraft: W / S throttle up and down (Shift / Ctrl too), A / D turn left and
    /// right (yaw), numpad 8 / 5 nose down and up, numpad 4 / 6 roll (the arrow keys do the
    /// same: Up nose down, Down nose up, Left / Right roll), Space speed brakes; with the left
    /// mouse button held the mouse is the stick. A controller the same way (gamepad.rs): right
    /// trigger faster, left slower, the left stick the stick, the bumpers the rudder, the right
    /// stick the view.
    pub fn update(&mut self, keys: &HashSet<KeyCode>, pad: &Pad, dt: f64, looking: bool, ground: impl Fn(DVec3) -> f64) -> Camera {
        if pad.view {
            self.next_view();
        }
        if pad.restart {
            self.reset();
        }
        if pad.zoom != 0.0 {
            self.zoom(pad.zoom * 5.0 * dt);
        }
        if pad.right != (0.0, 0.0) {
            // the right stick turns the view at a rate (and it goes back behind afterwards)
            self.look.0 += pad.right.0 * 2.6 * dt;
            self.look.1 = (self.look.1 + pad.right.1 * 1.8 * dt).clamp(-1.4, 1.4);
            self.look_idle = 0.0;
        }
        let k = |c: KeyCode| keys.contains(&c) as u8 as f64;
        // a key is all or nothing: the stick eases over to it, and back when it is let go
        let ease = |cur: &mut f64, to: f64, dt: f64| {
            let rate = if to == 0.0 { 3.0 } else { 1.5 };
            *cur += (to - *cur).clamp(-rate * dt, rate * dt);
        };
        // A key asks for half of what the aircraft can do (4.5 g, 145 deg/s): easy to hold a
        // turn or a climb with, not a 9 g snap. The mouse stick reaches all of it.
        const KEY_PITCH: f64 = 0.45;
        const KEY_ROLL: f64 = 0.45;
        let (mx, my) = self.mouse_stick;
        let roll_keys = k(KeyCode::Numpad6).max(k(KeyCode::ArrowRight)) - k(KeyCode::Numpad4).max(k(KeyCode::ArrowLeft));
        ease(&mut self.hands.0, (KEY_PITCH * (k(KeyCode::Numpad5).max(k(KeyCode::ArrowDown)) - k(KeyCode::Numpad8).max(k(KeyCode::ArrowUp))) + my).clamp(-1.0, 1.0), dt);
        ease(&mut self.hands.1, (KEY_ROLL * roll_keys + mx).clamp(-1.0, 1.0), dt);
        ease(&mut self.hands.2, (k(KeyCode::KeyD) - k(KeyCode::KeyA) + pad.rudder).clamp(-1.0, 1.0), dt);
        // (the controller's stick is a hand already: it is not eased, and it reaches all the
        // aircraft can do; forward is nose down)
        let (mut pitch, mut roll, mut yaw) = ((self.hands.0 - pad.left.1).clamp(-1.0, 1.0), (self.hands.1 + pad.left.0).clamp(-1.0, 1.0), self.hands.2);
        // A / D turn the aircraft the way an aircraft turns: it banks into the turn (to 40
        // degrees), pulls the little extra that keeps the nose from dropping, and levels its
        // wings again when the key is let go. Rolling by hand takes over.
        let (xb, yb, zb) = self.sim.axes();
        let up = self.sim.pos.normalize();
        let bank = (-yb.dot(up)).atan2(-zb.dot(up));
        if let Some(h) = self.hold {
            (pitch, roll, yaw) = (h.pitch, h.roll, h.yaw);
        }
        let manual = roll_keys != 0.0 || mx.abs() > 0.02 || pad.left.0 != 0.0 || self.hold.map(|h| h.roll != 0.0).unwrap_or(false);
        if manual || xb.dot(up).abs() > 0.8 {
            self.auto_bank = false;
        } else if yaw.abs() > 0.02 {
            self.auto_bank = true;
        }
        if self.auto_bank {
            // (added to what the hands ask this frame, not kept: the pull is the same at any
            // frame rate)
            let want = yaw * 40f64.to_radians();
            roll = (1.3 * (want - bank)).clamp(-0.4, 0.4);
            pitch = (pitch + (1.0 / bank.cos().max(0.6) - 1.0) / 8.0).min(1.0);
            if yaw.abs() < 0.02 && bank.abs() < 0.02 {
                self.auto_bank = false;
                roll = 0.0;
            }
        }
        (self.stick.pitch, self.stick.roll, self.stick.yaw) = (pitch, roll, yaw);
        if !self.mouse_fly {
            // (the mouse let go: its stick springs back)
            let back = 1.0 - (-dt * 8.0).exp();
            self.mouse_stick = (mx - mx * back, my - my * back);
        }
        let (faster, slower) = (k(KeyCode::KeyW).max(k(KeyCode::ShiftLeft)).max(k(KeyCode::ShiftRight)).max(pad.rt), k(KeyCode::KeyS).max(k(KeyCode::ControlLeft)).max(k(KeyCode::ControlRight)).max(pad.lt));
        self.stick.throttle = (self.stick.throttle + 0.6 * dt * (faster - slower)).clamp(0.0, 1.0);
        // (held back past idle, the throttle opens the speed brakes)
        self.stick.brake = keys.contains(&KeyCode::Space) || pad.brake || (slower > 0.5 && self.stick.throttle <= 0.0);
        if let Some(h) = self.hold {
            (self.stick.throttle, self.stick.brake) = (h.throttle, h.brake);
        }

        let steps = (dt * 240.0).ceil().max(1.0) as usize;
        let g = ground(self.sim.pos);
        for _ in 0..steps {
            self.sim.step(&self.stick, g, dt / steps as f64);
        }
        if self.sim.crashed && self.battle {
            self.down = true;
        } else if self.sim.crashed {
            println!("Hit the ground at {:.0} m/s. Back to the start.", self.sim.tas);
            self.reset();
        }
        self.time += dt;
        let supersonic = self.sim.mach >= 1.0;
        if supersonic != self.was_super {
            println!("{} (Mach {:.2} at {:.0} m)", if supersonic { "Supersonic" } else { "Subsonic again" }, self.sim.mach, self.sim.alt);
            self.was_super = supersonic;
        }
        self.leave_wake(dt);
        self.look_idle += dt;
        let _ = looking;
        if self.look_idle > 2.0 && self.hold.is_none() {
            // (the view goes back behind the aircraft a moment after the mouse stops, as in GTA V)
            let back = 1.0 - (-dt * 3.0).exp();
            self.look.0 -= self.look.0 * back;
            self.look.1 -= self.look.1 * back;
        }

        // ---- the camera
        let s = &self.sim;
        let model = s.model_rot();
        let (fwd, right, down) = s.axes();
        let speed_fov = 6.0 * ((s.mach - 0.6) / 1.2).clamp(0.0, 1.0);
        let (pos, vel) = (s.pos, s.vel);
        match self.view {
            View::Cockpit => {
                let (f, r, u) = self.cam_att_now(fwd, right, down);
                Camera { pos: pos + model * EYE, right: r, up: u, fwd: f, fov_y: 72f32.to_radians() }
            }
            View::Chase => {
                // Behind the aircraft, a little above it, looking the way it points. The camera
                // turns after the nose with a lag, so the aircraft is seen to pitch and roll
                // against a horizon that stays nearly level. Going straight up or down there is
                // no level to keep: the camera then holds the up it had, and rights itself after.
                let world_up = pos.normalize();
                self.cam_fwd = (self.cam_fwd + (fwd - self.cam_fwd) * (1.0 - (-dt * 5.0).exp())).normalize_or(fwd);
                let steep = self.cam_fwd.dot(world_up).abs();
                let level = (world_up - down * 0.08).normalize_or(world_up);
                let ease = (1.0 - (-dt * 2.5).exp()) * (1.0 - ((steep - 0.80) / 0.15).clamp(0.0, 1.0));
                self.cam_up = (self.cam_up + (level - self.cam_up) * ease).normalize_or(world_up);
                let cf = self.cam_fwd;
                let cr = cf.cross(self.cam_up).normalize_or(right);
                let cu = cr.cross(cf);
                self.cam_up = cu;
                let (sy, cy) = self.look.0.sin_cos();
                let (sp, cp) = (self.look.1 + 0.16).sin_cos();
                let cam = pos + ((-cf * cy + cr * sy) * cp + cu * sp) * self.dist;
                let f = (pos + cf * (0.3 * self.dist) * cy.max(0.0) + cu * (0.06 * self.dist) - cam).normalize();
                let r = f.cross(cu).normalize_or(cr);
                Camera { pos: cam, right: r, up: r.cross(f), fwd: f, fov_y: (50.0 + speed_fov as f32).to_radians() }
            }
            View::FlyBy => {
                // a camera standing in the air ahead and to the side: the aircraft comes past it
                let up = pos.normalize();
                let stale = match self.flyby_at {
                    Some((at, t)) => self.time - t > 9.0 || (at - pos).length() > 2500.0 + 6.0 * vel.length(),
                    None => true,
                };
                if stale {
                    let side = vel.cross(up).normalize_or(right);
                    self.flyby_at = Some((pos + vel * 3.2 + side * 55.0 + up * 12.0, self.time));
                }
                let at = self.flyby_at.unwrap().0;
                let f = (pos - at).normalize();
                let r = f.cross(up).normalize_or(right);
                let d = (pos - at).length();
                Camera { pos: at, right: r, up: r.cross(f), fwd: f, fov_y: ((60.0 / d.max(1.0)).atan() * 2.0).clamp(0.06, 1.0) as f32 }
            }
        }
    }

    /// The cockpit view's axes: the aircraft's, the head turned by the mouse.
    fn cam_att_now(&self, fwd: DVec3, right: DVec3, down: DVec3) -> (DVec3, DVec3, DVec3) {
        let (sy, cy) = self.look.0.clamp(-2.4, 2.4).sin_cos();
        let (sp, cp) = self.look.1.clamp(-0.6, 1.3).sin_cos();
        let f = (fwd * cy + right * sy) * cp - down * sp;
        let r = (right * cy - fwd * sy).normalize();
        (f, r, r.cross(f))
    }

    /// The wake: a sample every WAKE_STEP seconds where the nozzle is; the air (and the cloud in
    /// it) drifts with the wind of the low cloud layer (clouds.glsl cloudDrift), and so does the wake.
    fn leave_wake(&mut self, dt: f64) {
        let wind = |p: DVec3| {
            let up = p.normalize();
            let east = DVec3::Z.cross(up).normalize_or(DVec3::X);
            east * if up.z.abs() > 0.5 { 8.0 } else { -8.0 }
        };
        for w in self.wake.iter_mut() {
            w.pos += wind(w.pos) * dt;
        }
        let due = (self.time / WAKE_STEP) as u64;
        if due > self.samples || self.wake.is_empty() {
            self.samples = due;
            let pos = self.sim.pos + self.sim.model_rot() * NOZZLE;
            self.wake.push_front(WakePoint { pos, time: self.time, k: due, contrail: self.sim.contrail() as f32, stir: 1.0 });
        }
        let (now, newest) = (self.time, self.samples);
        self.wake.retain(|w| now - w.time < WAKE_LIFE && (newest - w.k <= FINE || w.k % COARSE == 0));
        self.wake.truncate(WAKE_POINTS - 1);
    }

    /// The aircraft for this frame's picture: where it is in the renderer's frame (`frame`: Earth
    /// axes -> the frame at the camera, `cam` the camera's Earth position), and what the shaders
    /// need of it.
    pub fn fill(&mut self, g: &mut Globals, frame: DMat3, cam: DVec3) -> Pose {
        let s = &self.sim;
        let model = s.model_rot();
        let to_frame = |r: DMat3, p: DVec3| {
            let m = frame * r;
            Affine3A::from_mat3_translation(Mat3::from_cols(m.x_axis.as_vec3(), m.y_axis.as_vec3(), m.z_axis.as_vec3()), (frame * (p - cam)).as_vec3())
        };
        let now = to_frame(model, s.pos);
        let inv = now.inverse();
        let (pr, pp) = self.prev.unwrap_or((model, s.pos));
        let back = to_frame(pr, pp) * inv;
        let (m, t) = (inv.matrix3, inv.translation);
        g.ac_inv = [[m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x], [m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y], [m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z]];
        self.prev = Some((model, s.pos));
        g.ac_state = [1.0, s.ab as f32, s.mach as f32, s.vapour_cone() as f32];
        let flow = model.transpose() * (-s.vel.normalize_or(DVec3::X));
        g.ac_vel = [flow.x as f32, flow.y as f32, flow.z as f32, s.tas as f32];
        // the wake, newest first: the nozzle now, then the samples
        let mut n = 0;
        let mut put = |g: &mut Globals, pos: DVec3, age: f64, contrail: f32, stir: f32| {
            let p = (frame * (pos - cam)).as_vec3();
            g.wake[n] = [p.x, p.y, p.z, age as f32];
            g.wake_b[n] = [contrail, stir, 0.0, 0.0];
            n += 1;
        };
        put(g, s.pos + model * NOZZLE, 0.0, 0.0, 1.0);
        for w in &self.wake {
            // (the trail forms a little way behind the engine, as the exhaust cools)
            let age = self.time - w.time;
            put(g, w.pos, age, w.contrail * ((age - 0.05) / 0.25).clamp(0.0, 1.0) as f32, w.stir);
        }
        // (z: the air's density against sea level's, for how strongly its shocks bend light)
        let thin = crate::flight::atmosphere(s.alt).rho / 1.225;
        g.ac_state2 = [s.wing_vapour() as f32, n as f32, thin as f32, (self.time % 1000.0) as f32];
        g.ac_state[1] = (s.ab as f32).max(0.0);
        g.ac_vel[3] = (s.power as f32).clamp(0.0, 1.0);
        let f = |v: f64| v as f32;
        Pose { xform: now, back, lef: f(s.surf.lef), flaperon: [f(s.surf.flaperon[0]), f(s.surf.flaperon[1])], tail: [f(s.surf.tail[0]), f(s.surf.tail[1])], rudder: f(s.surf.rudder) }
    }

    /// The instruments, for the window's title.
    pub fn hud(&self) -> String {
        let s = &self.sim;
        let up = s.pos.normalize();
        let climb = s.vel.dot(up);
        let power = if s.power >= MIL { format!("AB {:.0}%", s.ab * 100.0) } else { format!("{:.0}%", s.power / MIL * 100.0) };
        // heading from north
        let east = DVec3::Z.cross(up).normalize_or(DVec3::X);
        let north = up.cross(east);
        let f = s.axes().0;
        let hdg = f.dot(east).atan2(f.dot(north)).to_degrees().rem_euclid(360.0);
        format!(
            "F-16C | {:.0} kt | Mach {:.2} | {:.0} ft | {:+.1} g | AoA {:.0}\u{b0} | {} | hdg {:03.0} | {:+.0} ft/min{} | {}",
            s.cas * 1.94384,
            s.mach,
            s.alt * 3.28084,
            s.n,
            s.alpha.to_degrees(),
            power,
            hdg,
            climb * 196.85,
            if s.brake > 0.5 { " | BRAKES" } else { "" },
            ["chase", "cockpit", "fly-by"][self.view as usize],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::globe::tiles::R_EARTH;

    /// Fly `seconds` with these keys and this controller at `fps`.
    fn fly(keys: &[KeyCode], pad: Pad, seconds: f64, fps: f64) -> Pilot {
        let mut p = Pilot::new(DVec3::X * (R_EARTH + 3000.0), 90.0, 250.0);
        p.hold = None;
        let keys: HashSet<KeyCode> = keys.iter().copied().collect();
        for _ in 0..(seconds * fps) as usize {
            p.update(&keys, &pad, 1.0 / fps, false, |_| R_EARTH);
        }
        p
    }

    fn bank(p: &Pilot) -> f64 {
        let (_, yb, zb) = p.sim.axes();
        let up = p.sim.pos.normalize();
        (-yb.dot(up)).atan2(-zb.dot(up)).to_degrees()
    }

    #[test]
    fn a_turn_is_the_same_at_any_frame_rate() {
        // D held: a 40 degree bank and the small pull that holds the nose up, whether the
        // picture is drawn 60 times a second or 240
        for fps in [60.0, 240.0] {
            let p = fly(&[KeyCode::KeyD], Pad::default(), 5.0, fps);
            assert!((bank(&p) - 40.0).abs() < 4.0, "{fps} fps: bank {:.1}", bank(&p));
            assert!(p.stick.pitch > 0.02 && p.stick.pitch < 0.06, "{fps} fps: pitch {:.3}", p.stick.pitch);
            assert!(p.sim.n > 1.1 && p.sim.n < 1.7, "{fps} fps: {:.2} g", p.sim.n);
        }
    }

    #[test]
    fn controller_flies_it() {
        // the left stick back: the nose comes up, and it pulls g; the right trigger: full throttle
        let p = fly(&[], Pad { left: (0.0, -1.0), rt: 1.0, ..Default::default() }, 1.5, 120.0);
        assert!(p.sim.n > 5.0, "{:.1} g", p.sim.n);
        assert!(p.stick.throttle > 0.99);
        // the stick to the right: it rolls right, by hand (no bank hold)
        let p = fly(&[], Pad { left: (0.5, 0.0), ..Default::default() }, 0.5, 120.0);
        assert!(bank(&p) > 20.0 && !p.auto_bank, "bank {:.1}", bank(&p));
        // the right bumper: a turn to the right, banked as with D
        let p = fly(&[], Pad { rudder: 1.0, ..Default::default() }, 5.0, 120.0);
        assert!((bank(&p) - 40.0).abs() < 4.0, "bank {:.1}", bank(&p));
        // the left trigger held down to idle opens the speed brakes
        let p = fly(&[], Pad { lt: 1.0, ..Default::default() }, 2.0, 120.0);
        assert!(p.stick.throttle == 0.0 && p.stick.brake);
    }
}
