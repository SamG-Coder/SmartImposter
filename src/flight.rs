//! The flight model of `--test-flight`: an F-16C flown through its fly-by-wire control laws.
//!
//! What is modelled, and from what:
//! - The air: the 1976 standard atmosphere (density, temperature, speed of sound by height).
//! - Forces: lift, drag and side force from coefficients by angle of attack and Mach number
//!   (wing 27.87 m2, 10.4 t); thrust of an F110-GE-129 (76.3 kN dry, 131.6 kN with afterburner at
//!   sea level, static), falling with air density, rising with ram pressure, and losing inlet
//!   pressure recovery above Mach 1 (the fixed normal-shock inlet). The coefficients are fitted
//!   to the aircraft's published performance, not taken from wind tunnel tables. The model tops
//!   out at Mach 2.10 at 12 km and Mach 1.29 at sea level (the real aircraft: about 2.05 and
//!   1.2); it pulls 9 g near Mach 0.85 low down, and its lift coefficient is 1.45 at the 25
//!   degree limit. The transonic drag rise and the supersonic loss of lift slope are in
//!   them.
//! - Control laws, as the real flight control computer flies the (unstable) airframe: the stick
//!   commands load factor in pitch (+9 g to -3 g, 1 g with the stick free), limited to 25.2
//!   degrees angle of attack; roll rate about the flight path in roll (to 324 deg/s, less at
//!   high angle of attack and low airspeed); the rudder commands sideslip, which is otherwise
//!   held at zero. The response follows the commands with the lags of the airframe, and has less
//!   authority the lower the dynamic pressure: slow and mushy near the stall.
//!
//! `arcade` (what `--test-flight` flies) departs from the aircraft on purpose: more afterburner,
//! to about Mach 3, and pedals that turn it flat. The tests below are of the real numbers.
//!
//! What is not: the moments themselves (the computer's commands are followed, not derived from
//! surface deflections), fuel burn, stores, landing gear, wind, the Earth's rotation.

use crate::globe::tiles::R_EARTH;
use glam::{DMat3, DQuat, DVec3};

pub const G0: f64 = 9.80665;
const MASS: f64 = 10_400.0;
const S_REF: f64 = 27.87;
const T_MIL: f64 = 76_300.0;
const T_AB: f64 = 131_600.0;
/// throttle setting of full dry power: above it the afterburner
pub const MIL: f64 = 0.75;
const ALPHA_MAX: f64 = 25.2 * std::f64::consts::PI / 180.0;
const ALPHA_MIN: f64 = -10.0 * std::f64::consts::PI / 180.0;

pub struct Air {
    pub rho: f64,
    pub sound: f64,
    pub temp: f64,
    pub pressure: f64,
}

/// The 1976 standard atmosphere at height h (m).
pub fn atmosphere(h: f64) -> Air {
    let h = h.clamp(-500.0, 47_000.0);
    let (temp, pressure) = if h < 11_000.0 {
        let t = 288.15 - 0.0065 * h;
        (t, 101_325.0 * (t / 288.15).powf(5.25588))
    } else if h < 20_000.0 {
        (216.65, 22_632.06 * (-(h - 11_000.0) / 6341.62).exp())
    } else if h < 32_000.0 {
        let t = 216.65 + 0.001 * (h - 20_000.0);
        (t, 5474.89 * (t / 216.65).powf(-34.1632))
    } else {
        let t = 228.65 + 0.0028 * (h - 32_000.0);
        (t, 868.02 * (t / 228.65).powf(-12.2011))
    };
    Air { rho: pressure / (287.053 * temp), sound: (1.4 * 287.053 * temp).sqrt(), temp, pressure }
}

fn table(tab: &[(f64, f64)], x: f64) -> f64 {
    if x <= tab[0].0 {
        return tab[0].1;
    }
    for w in tab.windows(2) {
        if x <= w[1].0 {
            return w[0].1 + (w[1].1 - w[0].1) * (x - w[0].0) / (w[1].0 - w[0].0);
        }
    }
    tab[tab.len() - 1].1
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Lift slope (per rad) by Mach number.
fn lift_slope(m: f64) -> f64 {
    table(&[(0.0, 3.6), (0.5, 3.7), (0.9, 4.3), (1.0, 4.4), (1.2, 4.2), (1.5, 3.5), (2.0, 2.6), (2.5, 2.1)], m)
}

/// Lift coefficient by angle of attack (rad) and Mach number: the leading-edge flaps and the
/// strakes carry the lift to ~35 degrees at low speed; less of it supersonic.
fn lift(alpha: f64, m: f64) -> f64 {
    const TAB: [(f64, f64); 12] =
        [(0.0, 0.04), (5.0, 0.37), (10.0, 0.70), (15.0, 1.02), (20.0, 1.28), (25.0, 1.45), (30.0, 1.56), (35.0, 1.58), (40.0, 1.50), (50.0, 1.25), (60.0, 1.0), (90.0, 0.05)];
    let deg = alpha.to_degrees();
    let base = if deg >= 0.0 { table(&TAB, deg) } else { 0.08 - 0.85 * table(&TAB, -deg) };
    let cl = base * lift_slope(m) / 3.6;
    let cap = table(&[(0.0, 1.58), (0.9, 1.5), (1.2, 1.15), (2.0, 0.9)], m);
    cl / (1.0 + (cl / cap).abs().powi(8)).powf(0.125)
}

fn drag(alpha: f64, m: f64, cl: f64, brake: f64) -> f64 {
    let cd0 = table(&[(0.0, 0.0185), (0.8, 0.0185), (0.9, 0.021), (1.0, 0.036), (1.05, 0.043), (1.2, 0.044), (1.5, 0.040), (2.0, 0.036), (2.5, 0.034)], m);
    let k = table(&[(0.0, 0.125), (0.9, 0.13), (1.2, 0.19), (1.5, 0.24), (2.0, 0.31)], m);
    let a = alpha.abs();
    let induced = k * cl * cl;
    // (past ~15 degrees the lift is the pressure on a tilted plate: its drag is the lift's tangent)
    let plate = cl.abs() * a.min(1.2).tan();
    let t = smoothstep(15f64.to_radians(), 30f64.to_radians(), a);
    cd0 + (induced + (plate - induced).max(0.0) * t) + 1.2 * a.sin().powi(2) * smoothstep(35f64.to_radians(), 60f64.to_radians(), a) + 0.05 * brake
}

/// What the pilot holds: stick, pedals, throttle.
#[derive(Clone, Copy, Default)]
pub struct Controls {
    /// stick back (+1, nose up) .. forward (-1)
    pub pitch: f64,
    /// stick right (+1, right wing down) .. left
    pub roll: f64,
    /// right pedal (+1, nose right) .. left
    pub yaw: f64,
    /// 0 idle .. MIL full dry power .. 1 full afterburner
    pub throttle: f64,
    pub brake: bool,
}

/// Where the moving surfaces stand (rad), for the picture.
#[derive(Clone, Copy, Default)]
pub struct Surfaces {
    pub lef: f64,
    pub flaperon: [f64; 2],
    pub tail: [f64; 2],
    pub rudder: f64,
}

pub struct Sim {
    /// Earth-centred position (m) and velocity (m/s); the Earth does not turn
    pub pos: DVec3,
    pub vel: DVec3,
    /// body axes (x forward, y right, z down) -> Earth axes
    pub att: DQuat,
    /// body rates (rad/s): roll, pitch, yaw
    pub omega: DVec3,
    /// the engine's power setting as it has spooled (same scale as the throttle)
    pub power: f64,
    pub brake: f64,
    n_cmd: f64,
    alpha_int: f64,
    roll_rate: f64,
    // ---- read-outs of the last step
    pub mach: f64,
    pub alpha: f64,
    pub beta: f64,
    /// load factor (g) along the aircraft's up
    pub n: f64,
    pub qbar: f64,
    /// height above sea level (m)
    pub alt: f64,
    pub tas: f64,
    /// calibrated airspeed (m/s): what the airspeed indicator shows
    pub cas: f64,
    pub thrust: f64,
    /// afterburner 0..1
    pub ab: f64,
    pub temp: f64,
    pub surf: Surfaces,
    pub crashed: bool,
    /// Tuned to play like an arcade game's aircraft, not like the real one (pilot.rs sets it):
    /// 30% more afterburner and no loss of inlet pressure above Mach 1, so it reaches about Mach
    /// 3 at height where the real aircraft stops near Mach 2; and the pedals turn it flat, at up
    /// to 1.4 g sideways, where the real rudder only slips the nose a few degrees.
    pub arcade: bool,
    start: (DVec3, DVec3, DQuat),
}

impl Sim {
    /// Level at `pos`, heading `heading` degrees from north, at `speed` m/s.
    pub fn new(pos: DVec3, heading: f64, speed: f64) -> Self {
        let up = pos.normalize();
        let east = DVec3::Z.cross(up).normalize_or(DVec3::X);
        let north = up.cross(east);
        let (s, c) = heading.to_radians().sin_cos();
        let xb = north * c + east * s;
        let zb = -up;
        let yb = zb.cross(xb);
        let att = DQuat::from_mat3(&DMat3::from_cols(xb, yb, zb));
        let vel = xb * speed;
        Sim {
            pos,
            vel,
            att,
            omega: DVec3::ZERO,
            power: 0.6,
            brake: 0.0,
            n_cmd: 1.0,
            alpha_int: 0.0,
            roll_rate: 0.0,
            mach: 0.0,
            alpha: 0.0,
            beta: 0.0,
            n: 1.0,
            qbar: 0.0,
            alt: 0.0,
            tas: speed,
            cas: speed,
            thrust: 0.0,
            ab: 0.0,
            temp: 288.0,
            surf: Surfaces::default(),
            crashed: false,
            arcade: false,
            start: (pos, vel, att),
        }
    }

    pub fn reset(&mut self) {
        let (pos, vel, att) = self.start;
        let arcade = self.arcade;
        *self = Sim::new(pos, 0.0, 0.0);
        self.arcade = arcade;
        (self.pos, self.vel, self.att, self.start) = (pos, vel, att, (pos, vel, att));
        self.tas = vel.length();
    }

    /// Body axes in Earth axes: forward, right, down.
    pub fn axes(&self) -> (DVec3, DVec3, DVec3) {
        (self.att * DVec3::X, self.att * DVec3::Y, self.att * DVec3::Z)
    }

    /// The model's frame (x right, y up, z aft) -> Earth axes.
    pub fn model_rot(&self) -> DMat3 {
        let (x, y, z) = self.axes();
        DMat3::from_cols(y, -z, -x)
    }

    /// One step of `dt` seconds (keep it at 1/120 s or less). `ground`: the ground's radius under
    /// the aircraft.
    pub fn step(&mut self, c: &Controls, ground: f64, dt: f64) {
        if self.crashed {
            return;
        }
        let r = self.pos.length();
        let up = self.pos / r;
        self.alt = r - R_EARTH;
        let air = atmosphere(self.alt);
        let v = self.vel.length().max(0.5);
        let vhat = self.vel / v;
        let (xb, yb, zb) = self.axes();
        let (u, vy, w) = (self.vel.dot(xb), self.vel.dot(yb), self.vel.dot(zb));
        let alpha = w.atan2(u);
        let beta = (vy / v).clamp(-1.0, 1.0).asin();
        let mach = v / air.sound;
        let qbar = 0.5 * air.rho * v * v;

        // ---- the engine: it spools (seconds from idle to full dry power), the afterburner lights
        let want = c.throttle.clamp(0.0, 1.0);
        let rate = if want > self.power { if self.power >= MIL { 0.5 } else { 0.28 } } else { 0.45 };
        self.power += (want - self.power).clamp(-rate * dt, rate * dt);
        let dry = (self.power / MIL).min(1.0);
        let ab = ((self.power - MIL) / (1.0 - MIL)).clamp(0.0, 1.0);
        let lapse = (air.rho / 1.225).powf(0.72);
        let recovery = if mach > 1.0 { 1.0 - 0.075 * (mach - 1.0).powf(1.35) } else { 1.0 };
        let ram_dry = 1.0 - 0.25 * mach + 0.2 * mach * mach;
        let ram_ab = 1.0 - 0.1 * mach + 0.13 * mach * mach;
        let t_dry = T_MIL * ram_dry * (0.06 + 0.94 * dry.powf(1.5));
        let thrust = (t_dry + (T_AB * ram_ab - T_MIL * ram_dry).max(0.0) * ab) * lapse * if self.arcade { 1.0 + 0.3 * ab } else { recovery };

        // ---- the air's forces
        self.brake += ((c.brake as u8 as f64) - self.brake).clamp(-1.5 * dt, 1.5 * dt);
        let cl = lift(alpha, mach);
        let cd = drag(alpha, mach, cl, self.brake);
        let cy = (-1.1 * beta).clamp(-0.4, 0.4);
        let ldir = (-zb + vhat * vhat.dot(zb)).normalize_or(-zb);
        let sdir = (yb - vhat * vhat.dot(yb)).normalize_or(yb);
        let mut force = (ldir * cl - vhat * cd + sdir * cy) * (qbar * S_REF) + xb * thrust;
        if self.arcade {
            // (the flat turn: the nose follows the flight path round, below)
            force += sdir * (c.yaw * (v * 0.07).min(14.0) * (qbar / 6000.0).min(1.0) * MASS);
        }
        let gravity = -up * (G0 * (R_EARTH / r).powi(2));
        let acc = force / MASS + gravity;
        let n = force.dot(-zb) / (MASS * G0);

        // ---- the flight control computer
        // how much the surfaces can do: with the dynamic pressure
        let auth = (qbar / 9000.0).clamp(0.02, 1.0);
        // pitch: the stick commands load factor; the angle of attack that gives it, limited
        let n_target = 1.0 + if c.pitch >= 0.0 { 8.0 * c.pitch } else { 4.0 * c.pitch };
        self.n_cmd += (n_target - self.n_cmd) * (1.0 - (-dt / 0.12).exp());
        let cl_req = self.n_cmd * MASS * G0 / (qbar * S_REF).max(1.0);
        let alpha_ff = ((cl_req - 0.04) / lift_slope(mach)).clamp(-0.4, 0.8);
        let limited = (alpha_ff + self.alpha_int >= ALPHA_MAX && self.n_cmd > n) || (alpha_ff + self.alpha_int <= ALPHA_MIN && self.n_cmd < n);
        if !limited {
            self.alpha_int = (self.alpha_int + 0.03 * (self.n_cmd - n) * dt * auth).clamp(-0.2, 0.2);
        }
        let alpha_cmd = (alpha_ff + self.alpha_int).clamp(ALPHA_MIN, ALPHA_MAX);
        // the flight path's own turning (the nose follows it), plus what closes the errors
        let om_path = self.vel.cross(acc) / (v * v).max(900.0);
        let q_c = (4.0 * (alpha_cmd - alpha)).clamp(-1.0, 1.0);
        // sideslip: held at zero; the pedals ask for some (less of it supersonic)
        let beta_cmd = if self.arcade { 0.0 } else { -c.yaw * 8f64.to_radians() * (1.0 - 0.7 * smoothstep(0.9, 1.5, mach)) };
        let r_c = (3.0 * (beta - beta_cmd)).clamp(-0.8, 0.8);
        // roll: rate about the flight path
        let p_max = 324f64.to_radians() * (qbar / 14_000.0).clamp(0.12, 1.0) * (1.0 - 0.65 * smoothstep(15f64.to_radians(), 25f64.to_radians(), alpha.abs())) * (1.0 - 0.3 * smoothstep(1.2, 2.0, mach));
        self.roll_rate += (c.roll * p_max - self.roll_rate) * (1.0 - (-dt / 0.22).exp());
        let want = self.att.inverse() * (om_path + self.att * DVec3::new(0.0, q_c, r_c) + vhat * self.roll_rate);
        let tau = DVec3::new(0.05, 0.10 / auth.sqrt(), 0.15 / auth.sqrt());
        let lim = DVec3::new(14.0, 6.0 * auth + 0.3, 3.0 * auth + 0.2);
        let dom = ((want - self.omega) / tau).clamp(-lim, lim);
        self.omega += dom * dt;

        // ---- the surfaces, as the computer would hold them (for the picture)
        let deg = |d: f64| d.to_radians();
        let lef = (deg(1.38 * alpha.to_degrees() + 1.45)).clamp(deg(-2.0), deg(25.0)) * (1.0 - smoothstep(0.9, 1.1, mach)) + deg(-2.0) * smoothstep(0.9, 1.1, mach);
        let roll_def = (self.roll_rate / p_max.max(0.1)).clamp(-1.0, 1.0);
        let pitch_def = deg(-16.0) * (dom.y / (6.0 * auth + 0.3)).clamp(-1.0, 1.0) + deg(-0.9) * (self.n_cmd - 1.0);
        let ease = |cur: &mut f64, to: f64| *cur += (to - *cur) * (1.0 - (-dt / 0.06).exp());
        ease(&mut self.surf.lef, lef);
        ease(&mut self.surf.flaperon[0], deg(-20.0) * roll_def);
        ease(&mut self.surf.flaperon[1], deg(20.0) * roll_def);
        ease(&mut self.surf.tail[0], pitch_def + deg(-6.0) * roll_def);
        ease(&mut self.surf.tail[1], pitch_def + deg(6.0) * roll_def);
        ease(&mut self.surf.rudder, deg(-25.0) * c.yaw + deg(-10.0) * (r_c / 0.8));

        // ---- integrate
        self.vel += acc * dt;
        self.pos += self.vel * dt;
        self.att = (self.att * DQuat::from_scaled_axis(self.omega * dt)).normalize();
        if self.pos.length() < ground + 1.5 {
            self.crashed = true;
        }

        self.mach = mach;
        self.alpha = alpha;
        self.beta = beta;
        self.n = n;
        self.qbar = qbar;
        self.tas = v;
        let qc = air.pressure * ((1.0 + 0.2 * mach * mach).powf(3.5) - 1.0);
        self.cas = 340.294 * (5.0 * ((qc / 101_325.0 + 1.0).powf(2.0 / 7.0) - 1.0)).sqrt();
        self.thrust = thrust;
        self.ab = ab;
        self.temp = air.temp;
    }

    /// How humid the air is taken to be here (0..1): vapour forms low down.
    pub fn humid(&self) -> f64 {
        smoothstep(5000.0, 800.0, self.alt)
    }

    /// The cloud of condensation round the aircraft near the speed of sound (0..1).
    pub fn vapour_cone(&self) -> f64 {
        smoothstep(0.90, 0.97, self.mach) * smoothstep(1.09, 1.02, self.mach) * self.humid()
    }

    /// Vapour over the wings and in the tip vortices in hard turns (0..1).
    pub fn wing_vapour(&self) -> f64 {
        smoothstep(4.0, 7.5, self.n.abs()) * self.humid() * smoothstep(1.05, 0.95, self.mach)
    }

    /// A condensation trail forms behind the engine where the air is colder than about -40 C.
    pub fn contrail(&self) -> f64 {
        smoothstep(235.0, 229.0, self.temp) * smoothstep(0.1, 0.4, self.power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fly(alt: f64, speed: f64, throttle: f64, pitch: f64, secs: f64) -> Sim {
        let mut s = Sim::new(DVec3::X * (R_EARTH + alt), 90.0, speed);
        s.power = throttle;
        let c = Controls { throttle, pitch, ..Default::default() };
        for _ in 0..(secs * 240.0) as usize {
            s.step(&c, R_EARTH, 1.0 / 240.0);
        }
        s
    }

    #[test]
    fn standard_atmosphere() {
        let a = atmosphere(0.0);
        assert!((a.rho - 1.225).abs() < 0.001 && (a.sound - 340.3).abs() < 0.1);
        let a = atmosphere(11_000.0);
        assert!((a.rho - 0.3639).abs() < 0.001 && (a.temp - 216.65).abs() < 0.01);
    }

    #[test]
    fn holds_one_g_hands_off() {
        let s = fly(5000.0, 230.0, 0.55, 0.0, 20.0);
        assert!((s.n - 1.0).abs() < 0.05, "n = {}", s.n);
        assert!(s.alpha.to_degrees() > 0.5 && s.alpha.to_degrees() < 8.0, "alpha = {}", s.alpha.to_degrees());
    }

    /// Level at `alt` in full afterburner until it stops accelerating (a crude height hold on the stick).
    fn top_speed(alt: f64, speed: f64, arcade: bool) -> f64 {
        let mut s = Sim::new(DVec3::X * (R_EARTH + alt), 90.0, speed);
        s.power = 1.0;
        s.arcade = arcade;
        let mut c = Controls { throttle: 1.0, ..Default::default() };
        for _ in 0..(600.0 * 120.0) as usize {
            let climb = s.vel.dot(s.pos.normalize());
            c.pitch = (-(s.alt - alt) * 0.0004 - climb * 0.004).clamp(-0.3, 0.3);
            s.step(&c, R_EARTH - 1.0e4, 1.0 / 120.0);
        }
        assert!((s.alt - alt).abs() < 300.0, "height {} not held at {alt}", s.alt);
        s.mach
    }

    #[test]
    fn top_speed_is_about_mach_two() {
        let (high, low) = (top_speed(12_000.0, 450.0, false), top_speed(150.0, 300.0, false));
        println!("top speed: Mach {high:.2} at 12 km, Mach {low:.2} at sea level");
        assert!(high > 1.9 && high < 2.2, "Mach {high} at 12 km");
        assert!(low > 1.1 && low < 1.35, "Mach {low} at sea level");
        // the arcade tuning (--test-flight) goes on to about Mach 3
        let (high, low) = (top_speed(12_000.0, 450.0, true), top_speed(150.0, 300.0, true));
        println!("arcade top speed: Mach {high:.2} at 12 km, Mach {low:.2} at sea level");
        assert!(high > 2.7 && high < 3.4, "arcade Mach {high} at 12 km");
    }

    #[test]
    fn pulls_nine_g_and_no_more() {
        let s = fly(1500.0, 290.0, 1.0, 1.0, 3.0);
        assert!(s.n > 8.0 && s.n < 9.6, "n = {}", s.n);
        // slow: the angle of attack limit, not the g limit
        let s = fly(1500.0, 110.0, 1.0, 1.0, 3.0);
        assert!(s.alpha.to_degrees() < 26.5 && s.n < 4.0, "alpha {} n {}", s.alpha.to_degrees(), s.n);
    }
}
