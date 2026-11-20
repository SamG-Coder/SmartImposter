//! `--battle`: a stress test in the shape of a game. Eight made-up squadrons, each in its own
//! paint (shaders/livery.glsl), fight over the place the flight starts from: every aircraft
//! against every squadron but its own. They chase each other, fire rockets that home on their
//! target and guns at close range, throw flares when a rocket is after them, burn and fall when
//! they are hit, and come back at the edge of the fight, so the number in the air stays what
//! was asked for. The pilot (pilot.rs) flies one of them; shot down, the pilot is put in another
//! aircraft, of another squadron, somewhere else in the fight.
//!
//! What it stresses: every aircraft is the full model (nine instances of its parts in the
//! acceleration structure, rebuilt every frame), and every rocket trail, flame, tracer, flare,
//! fireball and smoke cloud is a volume traced by the same rays (shaders/fx.glsl).
//!
//! The other aircraft do not fly the flight model (flight.rs): they are points that turn at a
//! limited g and bank the way the turn asks, which is all that can be told from outside.

use crate::aircraft::{Craft, FxGpu, Pose, KIND_DRONE, KIND_GUN, KIND_JET, KIND_LAUNCHER, KIND_ROCKET};
use glam::{Affine3A, DMat3, DVec3, Mat3, Vec3};
use std::collections::VecDeque;

pub const SQUADRONS: [&str; 8] = ["Grey", "Desert", "Forest", "Sea", "Night", "Arctic", "Tiger", "Crimson"];
/// the nozzle and the tail of a rocket, model frame (x right, y up, z aft)
const NOZZLE: DVec3 = DVec3::new(0.0, 0.085, 5.85);
const ROCKET_TAIL: DVec3 = DVec3::new(0.0, 0.0, 1.5);
const ROCKET_SPEED: f64 = 980.0;
const ROCKET_LIFE: f64 = 9.0;
const TRACER_SPEED: f64 = 1050.0;
const G: f64 = 9.81;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
    /// A direction, any direction.
    fn dir(&mut self) -> DVec3 {
        loop {
            let v = DVec3::new(self.range(-1.0, 1.0), self.range(-1.0, 1.0), self.range(-1.0, 1.0));
            if v.length_squared() > 0.01 && v.length_squared() < 1.0 {
                return v.normalize();
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Flying,
    /// hit: burning, tumbling, on its way down
    Falling,
    /// gone until it comes back at the edge of the fight
    Out,
}

/// What a trail of smoke is made of.
#[derive(Clone, Copy, PartialEq)]
enum Smoke {
    Rocket,
    /// an aircraft that has been hit but still flies
    Damage,
    Wreck,
    /// the vapour a wing tip trails in a hard turn
    Vapour,
}

impl Smoke {
    /// seconds it lasts, seconds between its samples, radius at first (m), growth (m per root
    /// second), colour, density at one metre radius
    fn look(self) -> (f64, f64, f64, f64, [f32; 3], f32) {
        match self {
            Smoke::Rocket => (6.5, 0.07, 0.55, 1.5, [0.86, 0.86, 0.88], 0.42),
            Smoke::Damage => (4.0, 0.12, 0.8, 1.3, [0.42, 0.42, 0.43], 0.16),
            Smoke::Wreck => (13.0, 0.12, 1.8, 3.2, [0.035, 0.033, 0.032], 1.15),
            Smoke::Vapour => (1.5, 0.10, 0.20, 0.70, [0.95, 0.96, 0.98], 0.075),
        }
    }
}

/// A trail of smoke: where it was left and when, newest first.
struct Trail {
    kind: Smoke,
    /// position, the time it was left, the length of trail left before it, its number
    pts: VecDeque<(DVec3, f64, f64, u32)>,
    next: f64,
    arc: f64,
    count: u32,
    seed: f32,
}

impl Trail {
    fn new(kind: Smoke, seed: f32) -> Self {
        Trail { kind, pts: VecDeque::new(), next: 0.0, arc: 0.0, count: 0, seed }
    }

    /// A sample where the smoke comes from now, if one is due.
    fn leave(&mut self, pos: DVec3, now: f64) {
        if now < self.next {
            return;
        }
        if let Some(&(last, ..)) = self.pts.front() {
            self.arc += (pos - last).length();
        }
        self.next = now + self.kind.look().1;
        self.count += 1;
        self.pts.push_front((pos, now, self.arc, self.count));
    }

    /// Forget what has thinned to nothing; of the older samples keep one in three.
    fn age(&mut self, now: f64) {
        let (life, ..) = self.kind.look();
        let newest = self.count;
        self.pts.retain(|&(_, t, _, k)| now - t < life && (newest - k < 14 || k % 3 == 0));
    }

    /// Its pieces for the picture. `head`: where it comes from this instant (None: it has stopped).
    fn draw(&self, head: Option<DVec3>, now: f64, to: &dyn Fn(DVec3) -> Vec3, fx: &mut Vec<FxGpu>) {
        let step = self.kind.look().1;
        let (life, _, r0, grow, colour, dens) = self.kind.look();
        let at = |age: f64| {
            let r = r0 + grow * age.max(0.0).sqrt();
            let fade = (1.0 - age / life).clamp(0.0, 1.0).powf(1.5);
            (r as f32, dens * fade as f32 / r as f32)
        };
        let mut prev: Option<(DVec3, f64, f64)> = head.map(|h| (h, now, self.arc + self.pts.front().map(|p| (h - p.0).length()).unwrap_or(0.0)));
        for &(p, t, arc, _) in &self.pts {
            // (a gap in its samples is a gap in the trail: it stopped, and began again)
            if let Some((q, tq, arc_q)) = prev.filter(|p| p.1 - t < 3.5 * step) {
                let ((ra, da), (rb, db)) = (at(now - tq), at(now - t));
                let (a, b) = (to(q), to(p));
                if (a - b).length_squared() > 1e-4 {
                    fx.push(FxGpu { a: [a.x, a.y, a.z, ra], b: [b.x, b.y, b.z, rb], colour: [colour[0], colour[1], colour[2], 0.5 * (da + db)], kind: [0.0, self.seed, -(arc_q as f32), match self.kind { Smoke::Wreck => 0.6, Smoke::Vapour => 0.0, _ => 0.45 }] });
                }
            }
            prev = Some((p, t, arc));
        }
    }
}

struct Jet {
    pos: DVec3,
    vel: DVec3,
    /// the model's frame (x right, y up, z aft) -> Earth axes
    rot: DMat3,
    /// which way its top points (it banks into its turns)
    up: DVec3,
    /// a wreck's tumbling (rad/s about this axis)
    spin: DVec3,
    prev: Option<(DMat3, DVec3)>,
    team: u8,
    state: State,
    /// when it came to be in this state
    since: f64,
    hp: i8,
    target: Option<usize>,
    think: f64,
    fire: f64,
    gun: f64,
    /// a rocket is after it: where the rocket is, and until when to believe that
    threat: DVec3,
    threat_until: f64,
    /// when it may throw its next flare, and when it threw its last
    flare: f64,
    flared: f64,
    /// the ground's radius under it and a few seconds ahead, looked up now and then
    ground: f64,
    ground_at: f64,
    skill: f64,
    ab: f64,
    trail: Trail,
    /// the vapour off its wing tips, right and left, and whether it is pulling hard enough to make any
    vapour: [Trail; 2],
    pulling: bool,
}

struct Rocket {
    pos: DVec3,
    vel: DVec3,
    prev: Option<(DMat3, DVec3)>,
    target: Option<Target>,
    team: u8,
    mine: bool,
    born: f64,
    ground: f64,
    ground_at: f64,
    trail: Trail,
}

struct Blast {
    pos: DVec3,
    vel: DVec3,
    born: f64,
    size: f64,
    seed: f32,
    puffs: [DVec3; 4],
}

#[derive(Clone, Copy, PartialEq)]
enum SparkKind {
    Flare,
    Tracer,
    Debris,
}

struct Spark {
    kind: SparkKind,
    pos: DVec3,
    vel: DVec3,
    born: f64,
    life: f64,
    /// a tracer: what it was fired at (it can hit nothing else), and by which squadron
    target: Option<Target>,
    team: u8,
    mine: bool,
}

/// What a rocket or a round is after.
#[derive(Clone, Copy, PartialEq)]
enum Target {
    Jet(usize),
    Drone(usize),
}

/// The squadrons' colours, on what has no paint scheme of its own: their drones and trucks.
const COLOURS: [[f32; 3]; 8] = [[0.36, 0.38, 0.41], [0.52, 0.38, 0.20], [0.09, 0.17, 0.06], [0.08, 0.18, 0.42], [0.03, 0.032, 0.036], [0.78, 0.80, 0.83], [0.72, 0.36, 0.04], [0.45, 0.06, 0.04]];

/// A drone: one of a swarm, in its place in it until it is sent down at a truck.
struct Drone {
    pos: DVec3,
    vel: DVec3,
    prev: Option<(DMat3, DVec3)>,
    swarm: usize,
    /// its place in the swarm: to the right, up, and back from the swarm's point (m)
    place: DVec3,
    alive: bool,
    since: f64,
    /// the truck it is flying into
    dive: Option<usize>,
    ground: f64,
    ground_at: f64,
    phase: f64,
}

/// A swarm of drones: the point they keep their places about, and the truck it is going for.
struct Swarm {
    pos: DVec3,
    vel: DVec3,
    team: u8,
    goal: Option<usize>,
    /// where it goes when there is no truck to go for (from the middle of the fight)
    wander: DVec3,
    think: f64,
    /// when the next of them may go down at the goal
    peel: f64,
    ground: f64,
    ground_at: f64,
    height: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum SiteKind {
    /// sends rockets up after aircraft
    Launcher,
    /// fires at aircraft and drones in reach
    Gun,
}

/// A truck on the ground with a rocket launcher or an anti-aircraft gun on it.
struct Site {
    kind: SiteKind,
    team: u8,
    pos: DVec3,
    /// the lie of the ground it stands on
    up: DVec3,
    rot: DMat3,
    prev: Option<(DMat3, DVec3)>,
    /// it has a place (there is land for it within the fight)
    placed: bool,
    hp: i8,
    /// hit often enough: burning since then, and gone after a while
    burning: Option<f64>,
    aim: Option<Target>,
    think: f64,
    fire: f64,
    /// the gun: firing until then, and when its next round goes
    burst: f64,
    round: f64,
}

/// What the pilot's aircraft is doing, each step.
pub struct Flown {
    pub pos: DVec3,
    pub vel: DVec3,
    /// its model frame -> Earth axes
    pub rot: DMat3,
    pub fire: bool,
}

pub struct Battle {
    /// (the first is the pilot's)
    jets: Vec<Jet>,
    rockets: Vec<Rocket>,
    blasts: Vec<Blast>,
    sparks: Vec<Spark>,
    /// smoke left by what is no longer there
    ghosts: Vec<Trail>,
    pub time: f64,
    rng: Rng,
    centre: DVec3,
    radius: f64,
    pub kills: u32,
    pub lost: u32,
    scores: [u32; 8],
    /// the pilot cannot be hit until then (just put in a new aircraft)
    shield: f64,
    /// a testing aid (SI_BATTLE immortal=1): the pilot is never hit
    pub immortal: bool,
    reload: f64,
    report: f64,
    /// the camera that watches the fight (watch): whom it follows and since when, where it is
    watching: Option<(usize, f64)>,
    watch_at: DVec3,
    watch_what: u8,
    /// those of the pilot's squadron that fly in formation with the pilot
    wing: Vec<usize>,
    /// something to tell the pilot (the console)
    pub news: Vec<String>,
    /// the war below: drones in swarms, and trucks with launchers and guns on them
    drones: Vec<Drone>,
    swarms: Vec<Swarm>,
    sites: Vec<Site>,
    /// trucks burnt out, and rockets sent up from the ground, so far
    pub razed: u32,
    pub from_ground: u32,
}

fn heading_of(pos: DVec3, vel: DVec3) -> f64 {
    let up = pos.normalize();
    let east = DVec3::Z.cross(up).normalize_or(DVec3::X);
    let north = up.cross(east);
    vel.dot(east).atan2(vel.dot(north)).to_degrees().rem_euclid(360.0)
}

/// The model's frame for an aircraft going `fwd` with its top towards `up`.
fn rot_of(fwd: DVec3, up: DVec3) -> DMat3 {
    let u = (up - fwd * up.dot(fwd)).normalize_or(fwd.any_orthonormal_vector());
    DMat3::from_cols(fwd.cross(u), u, -fwd)
}

/// `from` turned towards `to` (both unit) by at most `angle`.
fn turn(from: DVec3, to: DVec3, angle: f64) -> DVec3 {
    let a = from.dot(to).clamp(-1.0, 1.0).acos();
    if a <= angle || a < 1e-6 {
        return to;
    }
    // (straight back: any way round will do)
    let to = if a > 3.1 { (to + from.any_orthonormal_vector() * 0.2).normalize() } else { to };
    let a = from.dot(to).clamp(-1.0, 1.0).acos();
    ((from * (a - angle).sin() + to * angle.sin()) / a.sin()).normalize()
}

impl Battle {
    /// `n` aircraft besides the pilot's, round `centre` (where the pilot starts, heading
    /// `heading` degrees), `drones` drones and `trucks` trucks (launchers and guns, half each).
    /// `ground`: the ground's radius under a point.
    pub fn new(centre: DVec3, heading: f64, n: usize, drones: usize, trucks: usize, ground: &dyn Fn(DVec3) -> f64) -> Self {
        let mut b = Battle {
            jets: Vec::new(),
            rockets: Vec::new(),
            blasts: Vec::new(),
            sparks: Vec::new(),
            ghosts: Vec::new(),
            time: 0.0,
            rng: Rng(0x9E37_79B9_7F4A_7C15),
            centre,
            radius: 1300.0 + 60.0 * (n as f64).sqrt(),
            kills: 0,
            lost: 0,
            scores: [0; 8],
            shield: 6.0,
            immortal: false,
            reload: 0.0,
            report: 30.0,
            watching: None,
            watch_at: DVec3::ZERO,
            watch_what: 0,
            wing: Vec::new(),
            news: Vec::new(),
            drones: Vec::new(),
            swarms: Vec::new(),
            sites: Vec::new(),
            razed: 0,
            from_ground: 0,
        };
        let up = centre.normalize();
        let east = DVec3::Z.cross(up).normalize_or(DVec3::X);
        let north = up.cross(east);
        let (s, c) = heading.to_radians().sin_cos();
        let ahead = north * c + east * s;
        let own = b.fresh(centre, ahead * 230.0, 0);
        b.jets.push(own);
        for i in 0..n {
            // the pilot's squadron comes in behind the pilot; the others from all round
            let team = ((i + 1) % 8) as u8;
            let bearing = heading.to_radians() + std::f64::consts::PI + team as f64 * std::f64::consts::TAU / 8.0;
            let out = north * bearing.cos() + east * bearing.sin();
            let side = out.cross(up);
            let dist = if team == 0 { b.rng.range(120.0, 900.0) } else { b.radius * b.rng.range(0.5, 0.9) };
            let p = centre + out * dist + side * b.rng.range(-0.12, 0.12) * b.radius.max(4.0 * dist.min(2000.0));
            let p = p.normalize() * (centre.length() + b.rng.range(-700.0, 700.0)).max(ground(p) + 600.0);
            let to = if team == 0 { ahead } else { (centre - p).normalize() };
            let to = (to - p.normalize() * to.dot(p.normalize())).normalize();
            let speed = b.rng.range(260.0, 330.0);
            let mut j = b.fresh(p, to * speed, team);
            j.ground = ground(p);
            j.fire = b.rng.range(4.0, 11.0);
            b.jets.push(j);
        }
        // the drones, 25 to a swarm, the swarms shared out among the squadrons
        const SWARM: usize = 25;
        for w in 0..drones.div_ceil(SWARM) {
            let out = b.rng.dir();
            let out = (out - up * out.dot(up)).normalize_or(east);
            let p = centre + out * b.radius * b.rng.range(0.3, 1.0);
            let height = b.rng.range(220.0, 520.0);
            let g = ground(p);
            let pos = p.normalize() * (g + height);
            let vel = -out.cross(up) * 78.0;
            b.swarms.push(Swarm { pos, vel, team: (w % 8) as u8, goal: None, wander: DVec3::ZERO, think: 0.0, peel: 0.0, ground: g, ground_at: 0.0, height });
            for k in 0..SWARM.min(drones - w * SWARM) {
                let place = DVec3::new(((k % 5) as f64 - 2.0) * 16.0, ((k / 5) % 2) as f64 * 9.0, (k / 5) as f64 * 14.0);
                let phase = b.rng.range(0.0, 6.0);
                b.drones.push(Drone { pos: pos + place, vel, prev: None, swarm: w, place, alive: true, since: 0.0, dive: None, ground: 0.0, ground_at: phase * 0.15, phase });
            }
        }
        // the trucks: they are stood on the ground in the first steps (ground_war)
        for k in 0..trucks {
            let kind = if k % 2 == 0 { SiteKind::Launcher } else { SiteKind::Gun };
            b.sites.push(Site { kind, team: ((k / 2) % 8) as u8, pos: centre, up, rot: DMat3::IDENTITY, prev: None, placed: false, hp: 4, burning: None, aim: None, think: 0.0, fire: 0.0, burst: 0.0, round: 0.0 });
        }
        b
    }

    fn fresh(&mut self, pos: DVec3, vel: DVec3, team: u8) -> Jet {
        let up = pos.normalize();
        Jet {
            pos,
            vel,
            rot: rot_of(vel.normalize_or(DVec3::X), up),
            up,
            spin: DVec3::ZERO,
            prev: None,
            team,
            state: State::Flying,
            since: self.time,
            hp: 3,
            target: None,
            think: self.time + self.rng.range(0.0, 2.0),
            fire: self.time + self.rng.range(2.0, 7.0),
            gun: 0.0,
            threat: DVec3::ZERO,
            threat_until: -1.0,
            flare: 0.0,
            flared: -1.0,
            ground: 0.0,
            ground_at: self.time + self.rng.range(0.0, 0.35),
            skill: self.rng.next(),
            ab: 0.0,
            trail: Trail::new(Smoke::Damage, self.rng.next() as f32),
            vapour: [Trail::new(Smoke::Vapour, 0.0), Trail::new(Smoke::Vapour, 0.5)],
            pulling: false,
        }
    }

    /// The squadron the pilot flies with now.
    pub fn team(&self) -> u8 {
        self.jets[0].team
    }

    /// The pilot's aircraft has been hit (or the pilot has flown it into the ground): the pilot
    /// needs another.
    pub fn down(&self) -> bool {
        self.jets[0].state != State::Flying
    }

    pub fn flying(&self) -> usize {
        self.jets.iter().filter(|j| j.state == State::Flying).count()
    }

    fn blast(&mut self, pos: DVec3, vel: DVec3, size: f64) {
        let puffs = [self.rng.dir(), self.rng.dir(), self.rng.dir(), self.rng.dir()];
        self.blasts.push(Blast { pos, vel, born: self.time, size, seed: self.rng.next() as f32, puffs });
        let pieces = (5.0 * size) as usize;
        for _ in 0..pieces {
            let v = vel * 0.5 + self.rng.dir() * self.rng.range(25.0, 110.0) * size.sqrt();
            self.sparks.push(Spark { kind: SparkKind::Debris, pos, vel: v, born: self.time, life: self.rng.range(0.8, 2.2), target: None, team: 0, mine: false });
        }
    }

    /// Aircraft `i` is hit and done for: by a squadron's weapon (and was it the pilot's), or by
    /// the ground.
    fn destroy(&mut self, i: usize, by: Option<u8>, mine: bool) {
        if self.jets[i].state != State::Flying || (i == 0 && (self.immortal || self.time < self.shield)) {
            return;
        }
        let (pos, vel, team) = (self.jets[i].pos, self.jets[i].vel, self.jets[i].team);
        self.blast(pos, vel, 1.0);
        let spin = self.rng.dir() * self.rng.range(1.0, 3.5);
        let seed = self.rng.next() as f32;
        let j = &mut self.jets[i];
        j.state = State::Falling;
        j.since = self.time;
        j.hp = 0;
        j.spin = spin;
        j.target = None;
        let old = std::mem::replace(&mut j.trail, Trail::new(Smoke::Wreck, seed));
        self.ghosts.push(old);
        if let Some(t) = by {
            self.scores[t as usize] += 1;
        }
        if mine {
            self.kills += 1;
            self.news.push(format!("Shot down a {} aircraft. {} so far.", SQUADRONS[team as usize], self.kills));
        }
        if i == 0 {
            self.lost += 1;
            self.news.push(match by {
                Some(t) => format!("Shot down by {} squadron.", SQUADRONS[t as usize]),
                None => "Hit the ground.".to_string(),
            });
        }
    }

    /// The pilot flew into the ground: no wreck to fall, a fire where it hit.
    pub fn crashed(&mut self, pos: DVec3) {
        if self.jets[0].state == State::Flying {
            self.blast(pos, DVec3::ZERO, 1.6);
            self.jets[0].state = State::Out;
            self.jets[0].since = self.time;
            self.lost += 1;
            self.news.push("Hit the ground.".to_string());
        }
    }

    /// Put the pilot in another aircraft: one of another squadron, some way from here. The pilot's
    /// old aircraft goes on falling in its place. Where the new one is, its heading (degrees) and
    /// its speed.
    pub fn take_over(&mut self) -> (DVec3, f64, f64) {
        let (old, team) = (self.jets[0].pos, self.jets[0].team);
        // (some way: half across the fight)
        let apart = 0.5 * self.radius;
        let ok = |j: &Jet, far: f64, other: bool| j.state == State::Flying && (j.pos - old).length() > far && (!other || j.team != team);
        let pick = |b: &mut Battle, far: f64, other: bool| {
            let c: Vec<usize> = (1..b.jets.len()).filter(|&k| ok(&b.jets[k], far, other)).collect();
            (!c.is_empty()).then(|| c[(b.rng.next() * c.len() as f64) as usize % c.len()])
        };
        let k = pick(self, apart, true).or_else(|| pick(self, 0.0, true)).or_else(|| pick(self, 0.0, false));
        match k {
            Some(k) => {
                self.jets.swap(0, k);
                // (what was after either of them is after the same aircraft still)
                let re = |t: usize| if t == 0 { k } else if t == k { 0 } else { t };
                let re_aim = |t: &mut Option<Target>| {
                    if let Some(Target::Jet(x)) = t {
                        *x = re(*x);
                    }
                };
                self.jets.iter_mut().for_each(|j| j.target = j.target.map(re));
                self.rockets.iter_mut().for_each(|r| re_aim(&mut r.target));
                self.sparks.iter_mut().for_each(|s| re_aim(&mut s.target));
                self.sites.iter_mut().for_each(|s| re_aim(&mut s.aim));
            }
            None => {
                // nobody left in the air: a new aircraft over the middle
                let p = self.centre.normalize() * (self.centre.length() + 1500.0);
                let v = self.centre.normalize().any_orthonormal_vector() * 250.0;
                self.jets[0] = self.fresh(p, v, (team + 1) % 8);
            }
        }
        let j = &mut self.jets[0];
        j.prev = None;
        j.hp = 3;
        // (it starts level: with room under it, and ahead)
        if j.ground > 0.0 && j.pos.length() < j.ground + 900.0 {
            j.pos = j.pos.normalize() * (j.ground + 900.0);
        }
        self.shield = self.time + 4.0;
        let (pos, vel) = (j.pos, j.vel);
        let far = (pos - old).length();
        self.news.push(format!("Now flying with {} squadron, {:.1} km away.", SQUADRONS[j.team as usize], far / 1000.0));
        (pos, heading_of(pos, vel), vel.length().clamp(180.0, 420.0))
    }

    /// What the pilot's rocket would go for: the aircraft or drone of another squadron nearest
    /// the nose, within reach. What, how far, and whose.
    fn in_sights(&self) -> Option<(Target, f64, u8)> {
        let me = &self.jets[0];
        let fwd = -me.rot.z_axis;
        let mut best: Option<(Target, f64, f64, u8)> = None;
        let mut look = |t: Target, p: DVec3, team: u8, reach: f64| {
            let d = p - me.pos;
            let dist = d.length();
            let c = d.dot(fwd) / dist.max(1.0);
            if team != me.team && dist < reach && c > 0.90 && best.map(|b| c > b.1).unwrap_or(true) {
                best = Some((t, c, dist, team));
            }
        };
        for (k, j) in self.jets.iter().enumerate().skip(1) {
            if j.state == State::Flying {
                look(Target::Jet(k), j.pos, j.team, 8000.0);
            }
        }
        for (k, d) in self.drones.iter().enumerate() {
            if d.alive {
                look(Target::Drone(k), d.pos, self.swarms[d.swarm].team, 4000.0);
            }
        }
        best.map(|b| (b.0, b.2, b.3))
    }

    fn launch(&mut self, from: usize, target: Option<Target>, mine: bool) {
        let j = &self.jets[from];
        let fwd = j.vel.normalize_or(-j.rot.z_axis);
        let side = if self.rng.next() < 0.5 { -1.0 } else { 1.0 };
        let pos = j.pos + j.rot * DVec3::new(2.6 * side, -0.5, 0.0);
        let vel = fwd * (j.vel.length() + 40.0) - j.rot.y_axis * 6.0;
        let seed = self.rng.next() as f32;
        self.rockets.push(Rocket { pos, vel, prev: None, target, team: j.team, mine, born: self.time, ground: j.ground, ground_at: self.time + 0.2, trail: Trail::new(Smoke::Rocket, seed) });
    }

    /// The war below the aircraft. Drones in swarms go for the trucks of the other squadrons
    /// and fly into them; the launchers send rockets up after aircraft; the guns fire at the
    /// aircraft and the drones in their reach. `water`: is that point of the ground water, or
    /// wooded (no place for a truck)?
    #[allow(clippy::too_many_arguments)]
    fn ground_war(&mut self, dt: f64, flown: &Flown, ground: &dyn Fn(DVec3) -> f64, water: &dyn Fn(DVec3) -> bool, seen: &[(DVec3, DVec3, u8, bool)], shielded: bool) {
        let now = self.time;
        let (centre, radius) = (self.centre, self.radius);
        let up = centre.normalize();
        let flat = |v: DVec3| v - up * v.dot(up);
        let ahead = flat(flown.vel).normalize_or(DVec3::ZERO);
        // ---- the trucks: stood where there is land for them, and somewhere new when the fight
        // has moved on or they have burnt out (a few a step: looking the ground up takes time)
        let mut budget = 8;
        let mut fired: Vec<(usize, usize)> = Vec::new();
        let room = self.rockets.len() < self.jets.len() * 3 / 4 + self.sites.len() / 2 + 8;
        for i in 0..self.sites.len() {
            let s = &self.sites[i];
            if !s.placed || flat(s.pos - centre).length() > 1.7 * radius || s.burning.map(|t| now - t > 25.0).unwrap_or(false) {
                if budget == 0 {
                    continue;
                }
                budget -= 1;
                let out = flat(self.rng.dir() + ahead * 0.5).normalize_or(up.any_orthonormal_vector());
                let far = radius * if s.placed { self.rng.range(0.7, 1.3) } else { 1.3 * self.rng.next().sqrt() };
                let p = centre + out * far;
                // (not on water or in a wood)
                if water(p) {
                    continue;
                }
                let p = p.normalize() * ground(p);
                let here = p.normalize();
                let east = DVec3::Z.cross(here).normalize_or(DVec3::X);
                let north = here.cross(east);
                let rise = |d: DVec3| (ground(p + d * 8.0) - ground(p - d * 8.0)) / 16.0;
                let lie = (here - east * rise(east) - north * rise(north)).normalize();
                // (nor on a slope a truck could not stand on)
                if lie.dot(here) < 0.9 {
                    continue;
                }
                let face = self.rng.dir();
                let face = (face - lie * face.dot(lie)).normalize_or(east);
                let wait = self.rng.next();
                let s = &mut self.sites[i];
                (s.pos, s.up, s.rot, s.prev) = (p, lie, rot_of(face, lie), None);
                (s.placed, s.burning, s.hp, s.aim) = (true, None, 6, None);
                (s.fire, s.think) = (now + 2.0 + 8.0 * wait, now + wait);
                continue;
            }
            if s.burning.is_some() {
                continue;
            }
            let s = &mut self.sites[i];
            let reach = if s.kind == SiteKind::Launcher { 5200.0 } else { 2300.0 };
            if now >= s.think {
                // the nearest aircraft of another squadron that is above its horizon
                s.think = now + 0.8 + 0.4 * (i % 7) as f64 / 7.0;
                let mut best: (f64, Option<Target>) = (reach, None);
                for (k, a) in seen.iter().enumerate() {
                    if !a.3 || a.2 == s.team || (k == 0 && shielded) {
                        continue;
                    }
                    let d = a.0 - s.pos;
                    let dist = d.length();
                    if dist < best.0 && d.dot(s.up) > 0.12 * dist {
                        best = (dist, Some(Target::Jet(k)));
                    }
                }
                if s.kind == SiteKind::Gun {
                    // (a drone in reach comes first: it is after a truck like this one)
                    let mut near: (f64, Option<Target>) = (1500.0, None);
                    for (k, d) in self.drones.iter().enumerate() {
                        let dist = (d.pos - s.pos).length();
                        if d.alive && dist < near.0 && self.swarms[d.swarm].team != s.team {
                            near = (dist, Some(Target::Drone(k)));
                        }
                    }
                    if near.1.is_some() {
                        best = near;
                    }
                }
                s.aim = best.1;
            }
            let Some(t) = s.aim else { continue };
            let (tp, tv, live) = match t {
                Target::Jet(k) => (seen[k].0, seen[k].1, seen[k].3),
                Target::Drone(k) => (self.drones[k].pos, self.drones[k].vel, self.drones[k].alive),
            };
            let d = tp - s.pos;
            let dist = d.length();
            if !live || dist > 1.1 * reach {
                s.aim = None;
                continue;
            }
            // it turns to face what it is after
            let fwd = -s.rot.z_axis;
            let to = (d - s.up * d.dot(s.up)).normalize_or(fwd);
            s.rot = rot_of(turn(fwd, to, 1.5 * dt), s.up);
            match (s.kind, t) {
                (SiteKind::Launcher, Target::Jet(k)) if room && now >= s.fire && to.dot(fwd) > 0.9 => {
                    s.fire = now + 7.0 + 7.0 * self.rng.next();
                    fired.push((i, k));
                }
                (SiteKind::Gun, _) => {
                    // in bursts, one barrel then the other
                    if now >= s.fire {
                        s.burst = now + 1.2;
                        s.fire = now + 3.0 + 2.5 * self.rng.next();
                    }
                    if now < s.burst && now >= s.round {
                        s.round = now + 0.06;
                        let side = if (now * 16.0) as i64 % 2 == 0 { 0.42 } else { -0.42 };
                        let muzzle = s.pos + s.rot * DVec3::new(side, 5.5, -2.1);
                        let lead = tp + tv * (dist / TRACER_SPEED);
                        let aim = ((lead - muzzle).normalize_or(s.up) + self.rng.dir() * 0.012).normalize();
                        self.sparks.push(Spark { kind: SparkKind::Tracer, pos: muzzle, vel: aim * TRACER_SPEED, born: now, life: 2.2, target: Some(t), team: s.team, mine: false });
                    }
                }
                _ => {}
            }
        }
        for (i, k) in fired {
            // a rocket leaves its tube, up and towards its target
            let s = &self.sites[i];
            let fwd = -s.rot.z_axis;
            let dir = (s.up * 0.8 + fwd * 0.6).normalize();
            let (pos, team, under) = (s.pos + s.up * 4.5 + fwd, s.team, s.pos.length() - 30.0);
            let seed = self.rng.next() as f32;
            self.rockets.push(Rocket { pos, vel: dir * 60.0, prev: None, target: Some(Target::Jet(k)), team, mine: false, born: now, ground: under, ground_at: now + 1.5, trail: Trail::new(Smoke::Rocket, seed) });
            self.blast(pos - dir * 3.0, DVec3::ZERO, 0.2);
            self.from_ground += 1;
        }
        // ---- the swarms: each goes for the nearest truck of another squadron
        let mut orders: Vec<(bool, Option<usize>)> = Vec::with_capacity(self.swarms.len());
        for w in 0..self.swarms.len() {
            let sw = &mut self.swarms[w];
            let open = |s: &Site| s.placed && s.burning.is_none() && s.team != sw.team;
            if now >= sw.think || sw.goal.map(|g| !open(&self.sites[g])).unwrap_or(false) {
                sw.think = now + 3.0;
                let at = sw.pos;
                sw.goal = (0..self.sites.len()).filter(|&g| open(&self.sites[g])).min_by(|&a, &b| (self.sites[a].pos - at).length_squared().total_cmp(&(self.sites[b].pos - at).length_squared()));
                if sw.goal.is_none() && (sw.wander == DVec3::ZERO || flat(centre + sw.wander - sw.pos).length() < 400.0) {
                    sw.wander = flat(self.rng.dir()).normalize_or(up.any_orthonormal_vector()) * 0.6 * radius;
                }
            }
            let to = match sw.goal {
                Some(g) => self.sites[g].pos,
                None => centre + sw.wander,
            };
            let off = flat(to - sw.pos);
            let level = flat(sw.vel).normalize_or(up.any_orthonormal_vector());
            // (over its goal it goes round and round; it turns and changes speed no faster than
            // its drones can follow)
            let going = flat(sw.vel).length().max(40.0);
            let dirn = turn(level, off.normalize_or(level), (11.0 / going).min(0.5) * dt);
            if now >= sw.ground_at {
                sw.ground = ground(sw.pos).max(ground(sw.pos + sw.vel * 5.0));
                sw.ground_at = now + 0.5;
            }
            let climb = ((sw.ground + sw.height - sw.pos.length()) * 0.6).clamp(-30.0, 45.0);
            let speed = if flat(sw.pos - centre).length() > 1.2 * radius { 160.0 } else { 78.0 };
            let speed = going + (speed - going).clamp(-14.0 * dt, 14.0 * dt);
            sw.vel = dirn * speed + sw.pos.normalize() * climb;
            sw.pos += sw.vel * dt;
            // left behind by the fight, it comes in again at the edge, ahead
            let mut jump = false;
            if flat(sw.pos - centre).length() > 2.2 * radius {
                let out = flat(self.rng.dir() + ahead * 0.7).normalize_or(level);
                let p = centre + out * 1.1 * radius;
                sw.ground = ground(p);
                sw.pos = p.normalize() * (sw.ground + sw.height);
                sw.vel = -out * 78.0;
                (sw.goal, sw.think, jump) = (None, now, true);
            }
            // near enough to its goal, one of them goes down at it, and then the next
            let mut send = None;
            if let Some(g) = sw.goal {
                if off.length() < 800.0 && now >= sw.peel {
                    sw.peel = now + 1.0;
                    send = Some(g);
                }
            }
            orders.push((jump, send));
        }
        // ---- the drones
        let mut hit: Vec<(usize, usize)> = Vec::new();
        let mut lost: Vec<usize> = Vec::new();
        for k in 0..self.drones.len() {
            let d = &mut self.drones[k];
            let sw = &self.swarms[d.swarm];
            let (jump, send) = &mut orders[d.swarm];
            let here = sw.pos.normalize();
            let level = (sw.vel - here * sw.vel.dot(here)).normalize_or(here.any_orthonormal_vector());
            let place = sw.pos + level.cross(here) * d.place.x + here * (d.place.y + 3.0 * (now * 0.7 + d.phase).sin()) - level * d.place.z;
            if *jump {
                (d.pos, d.vel, d.prev, d.dive) = (place, sw.vel, None, None);
            }
            if !d.alive {
                // another comes up from behind to take its place
                if now - d.since > 5.0 + 4.0 * d.phase.fract() {
                    (d.alive, d.pos, d.vel, d.prev, d.dive) = (true, place - level * 350.0, sw.vel, None, None);
                }
                continue;
            }
            if d.dive.is_none() {
                if let Some(g) = send.take() {
                    d.dive = Some(g);
                }
            }
            let mut want = sw.vel + (place - d.pos).clamp_length_max(90.0) * 1.2;
            let mut push = 45.0;
            if let Some(g) = d.dive {
                let s = &self.sites[g];
                if !s.placed || s.burning.is_some() {
                    d.dive = None;
                } else {
                    let to = s.pos + s.up * 1.5 - d.pos;
                    if to.length() < 9.0 + d.vel.length() * dt {
                        hit.push((k, g));
                        continue;
                    }
                    (want, push) = (to.normalize() * 135.0, 60.0);
                }
            }
            d.vel += (want - d.vel).clamp_length_max(push * dt);
            if now >= d.ground_at {
                d.ground = ground(d.pos);
                d.ground_at = now + if d.dive.is_some() { 0.25 } else { 0.9 };
            }
            let above = d.pos.length() - d.ground;
            if d.dive.is_none() && above < 40.0 {
                d.vel += d.pos.normalize() * 60.0 * dt;
            }
            d.pos += d.vel * dt;
            if above < 1.5 {
                lost.push(k);
            }
        }
        for (k, g) in hit {
            let at = self.sites[g].pos + self.sites[g].up * 2.0;
            self.blast(at, DVec3::ZERO, 0.6);
            let d = &mut self.drones[k];
            (d.alive, d.since, d.dive) = (false, now, None);
            let s = &mut self.sites[g];
            s.hp -= 1;
            if s.hp <= 0 && s.burning.is_none() {
                (s.burning, s.aim) = (Some(now), None);
                self.razed += 1;
                self.blast(at, DVec3::ZERO, 2.0);
            }
        }
        for k in lost {
            let at = self.drones[k].pos;
            self.blast(at, DVec3::ZERO, 0.3);
            let d = &mut self.drones[k];
            (d.alive, d.since, d.dive) = (false, now, None);
        }
    }

    /// A drone is hit (and was it by the pilot).
    fn drone_down(&mut self, k: usize, mine: bool) {
        if !self.drones[k].alive {
            return;
        }
        let (pos, vel, team) = (self.drones[k].pos, self.drones[k].vel, self.swarms[self.drones[k].swarm].team);
        self.blast(pos, vel, 0.3);
        let d = &mut self.drones[k];
        (d.alive, d.since, d.dive) = (false, self.time, None);
        if mine {
            self.kills += 1;
            self.news.push(format!("Shot down a {} drone. {} so far.", SQUADRONS[team as usize], self.kills));
        }
    }

    /// The pilot gets out. The aircraft goes on with nobody in it and falls; until take_over the
    /// pilot is on foot, and nothing in the fight is after the pilot.
    pub fn eject(&mut self) {
        if self.jets[0].state != State::Flying {
            return;
        }
        let (pos, vel, team, rot) = (self.jets[0].pos, self.jets[0].vel, self.jets[0].team, self.jets[0].rot);
        let mut wreck = self.fresh(pos, vel, team);
        (wreck.rot, wreck.state, wreck.hp) = (rot, State::Falling, 0);
        wreck.spin = self.rng.dir() * self.rng.range(0.3, 1.0);
        wreck.trail = Trail::new(Smoke::Wreck, self.rng.next() as f32);
        // (in the place of one that is out, where there is one: the number of them stays)
        match (1..self.jets.len()).find(|&k| self.jets[k].state == State::Out) {
            Some(k) => {
                let old = std::mem::replace(&mut self.jets[k], wreck);
                self.ghosts.push(old.trail);
            }
            None => self.jets.push(wreck),
        }
        self.jets[0].state = State::Out;
        self.jets[0].since = self.time;
        self.news.push("Out of the aircraft. E or R: into another.".to_string());
    }

    /// The pilot, on foot, fires a round from `at` along `dir`, moving at `vel`. Pointed within
    /// three degrees of an aircraft or a drone of another squadron (or of where that will be
    /// when the round gets there), the round is sent ahead of it for the pilot; else it goes
    /// where it is pointed, and hits nothing.
    pub fn round(&mut self, at: DVec3, dir: DVec3, vel: DVec3) {
        let team = self.jets[0].team;
        let mut best: (f64, Option<Target>, DVec3) = (0.9986, None, dir);
        let mut look = |t: Target, p: DVec3, v: DVec3| {
            let dist = (p - at).length();
            let lead = (p + v * (dist / TRACER_SPEED) - at).normalize_or(dir);
            let on = lead.dot(dir).max((p - at).normalize_or(dir).dot(dir));
            if dist < 3000.0 && on > best.0 {
                best = (on, Some(t), lead);
            }
        };
        for (k, j) in self.jets.iter().enumerate().skip(1) {
            if j.state == State::Flying && j.team != team {
                look(Target::Jet(k), j.pos, j.vel);
            }
        }
        for (k, d) in self.drones.iter().enumerate() {
            if d.alive && self.swarms[d.swarm].team != team {
                look(Target::Drone(k), d.pos, d.vel);
            }
        }
        let aim = (best.2 + self.rng.dir() * 0.004).normalize();
        self.sparks.push(Spark { kind: SparkKind::Tracer, pos: at, vel: vel + aim * TRACER_SPEED, born: self.time, life: 2.6, target: best.1, team, mine: true });
    }

    /// One step of `dt` seconds. `water`: is that point of the ground water or wooded (no truck stands there)?
    pub fn step(&mut self, dt: f64, flown: &Flown, ground: &dyn Fn(DVec3) -> f64, water: &dyn Fn(DVec3) -> bool) {
        self.time += dt;
        let now = self.time;
        if self.jets[0].state == State::Flying {
            let j = &mut self.jets[0];
            (j.pos, j.vel, j.rot) = (flown.pos, flown.vel, flown.rot);
        }
        // The fight follows the pilot: its middle drifts after the pilot's aircraft when that
        // strays, and those that come back come back round the new middle.
        // It stays close round the pilot, in height too, or all the pilot would see of it is smoke.
        // Its middle is kept ahead of the pilot, where the pilot is looking.
        {
            let ahead = flown.pos + flown.vel.normalize_or(DVec3::ZERO) * 0.3 * self.radius;
            self.centre += (ahead - self.centre) * (1.0 - (-dt * 1.2).exp());
        }
        // Four of the pilot's own squadron keep formation with the pilot: the nearest, and when
        // one is lost, the next.
        let own = self.jets[0].team;
        let led = self.jets[0].state == State::Flying;
        let jets = &self.jets;
        self.wing.retain(|&k| jets[k].state == State::Flying && jets[k].team == own && k != 0 && (jets[k].pos - flown.pos).length() < 2500.0);
        while led && self.wing.len() < 4 {
            let free = (1..jets.len()).filter(|k| jets[*k].state == State::Flying && jets[*k].team == own && !self.wing.contains(k));
            match free.min_by(|a, b| (jets[*a].pos - flown.pos).length_squared().total_cmp(&(jets[*b].pos - flown.pos).length_squared())) {
                Some(k) => self.wing.push(k),
                None => break,
            }
        }
        let wing = self.wing.clone();
        // (and the fight keeps the pilot's pace: flown fast, it is a fast fight, not one left behind)
        let pace = (flown.vel.length() - 280.0).max(0.0);
        // what every aircraft can see of the others: where, how fast, whose, and still flying
        let seen: Vec<(DVec3, DVec3, u8, bool)> = self.jets.iter().map(|j| (j.pos, j.vel, j.team, j.state == State::Flying)).collect();
        let shielded = self.immortal || now < self.shield;
        let mut launches: Vec<(usize, usize)> = Vec::new();
        let mut downed: Vec<usize> = Vec::new();
        let rocket_room = self.rockets.len() < self.jets.len() * 3 / 4 + self.sites.len() / 2 + 8;
        for i in 0..self.jets.len() {
            let centre = self.centre;
            let radius = self.radius;
            let j = &mut self.jets[i];
            match j.state {
                // (left far behind, it comes back in at the edge like those shot down)
                State::Flying if i > 0 && (j.pos - centre).length() > 2.2 * radius => {
                    j.state = State::Out;
                    j.since = now - 10.0;
                    continue;
                }
                State::Flying => {}
                State::Falling => {
                    // a wreck: it falls, tumbling, and the air slows it
                    let up = j.pos.normalize();
                    let speed = j.vel.length();
                    j.vel += (-up * G - j.vel * speed * 0.0009) * dt;
                    j.pos += j.vel * dt;
                    if j.spin.length_squared() > 1e-6 {
                        j.rot = DMat3::from_axis_angle(j.spin.normalize(), j.spin.length() * dt) * j.rot;
                    }
                    if now >= j.ground_at {
                        j.ground = ground(j.pos);
                        j.ground_at = now + 0.1;
                    }
                    if j.pos.length() < j.ground + 3.0 || now - j.since > 40.0 {
                        downed.push(i);
                    }
                    continue;
                }
                State::Out => continue,
            }
            if now >= j.ground_at {
                // (under it, and where it will be in four seconds: the mountains come up fast)
                j.ground = ground(j.pos).max(ground(j.pos + j.vel * 4.0));
                j.ground_at = now + 0.35;
            }
            if i == 0 {
                // the pilot flies this one; it throws its flares by itself
                if now < j.threat_until && now >= j.flare {
                    j.flare = now + 0.6;
                    j.flared = now;
                    self.sparks.push(Spark { kind: SparkKind::Flare, pos: j.pos - j.rot.y_axis * 1.0, vel: j.vel * 0.9 - j.rot.y_axis * 25.0, born: now, life: 3.5, target: None, team: j.team, mine: false });
                }
                continue;
            }
            // ---- who to go after
            if now >= j.think || j.target.map(|t| !seen[t].3 || seen[t].2 == j.team).unwrap_or(false) {
                j.think = now + 1.5 + 1.5 * j.skill;
                let fwd = j.vel.normalize_or(DVec3::X);
                let mut best = (f64::MAX, None);
                for (k, s) in seen.iter().enumerate() {
                    if !s.3 || s.2 == j.team || (k == 0 && shielded) {
                        continue;
                    }
                    let d = s.0 - j.pos;
                    let dist = d.length();
                    // (the one ahead is worth more than the one behind; a little of each to its
                    // own; the pilot draws more than a fair share of them, the pilot's formation less)
                    let score = dist * (1.6 - 0.6 * d.dot(fwd) / dist.max(1.0)) * (0.8 + 0.4 * ((k * 31 + i * 17) % 13) as f64 / 13.0) * if k == 0 { 0.6 } else if wing.contains(&k) { 2.0 } else { 1.0 };
                    if score < best.0 {
                        best = (score, Some(k));
                    }
                }
                j.target = best.1.filter(|_| best.0 < 30_000.0);
            }
            // ---- which way to go
            let up = j.pos.normalize();
            let speed = j.vel.length().max(1.0);
            let fwd = j.vel / speed;
            let above = j.pos.length() - j.ground;
            let off = j.pos - centre.normalize() * centre.normalize().dot(j.pos);
            let mut want = (-off).normalize_or(fwd);
            let mut want_speed = 300.0;
            let mut aim = None;
            if let Some(t) = j.target {
                let (tp, tv, ..) = seen[t];
                let dist = (tp - j.pos).length();
                let lead = tp + tv * (dist / TRACER_SPEED).min(2.5);
                want = (lead - j.pos).normalize_or(fwd);
                want_speed = if dist > 1500.0 { 390.0 } else { (tv.length() + 30.0).clamp(230.0, 380.0) };
                if dist < 220.0 {
                    want = (want + up * 0.7 + fwd.cross(up) * 0.5).normalize(); // not into it
                }
                aim = Some((t, dist, want.dot(fwd)));
            }
            let hunted = now < j.threat_until;
            // in formation: where its place beside the pilot is going, and a little towards it
            let place = wing.iter().position(|&k| k == i).filter(|_| led && !hunted);
            let mut limits = (6.0 + 3.0 * j.skill, 26.0 + 0.1 * pace, 520.0 + pace);
            want_speed += pace;
            if let Some(n) = place {
                // (a little ahead of the leader, where the chase camera sees them)
                const PLACES: [DVec3; 4] = [DVec3::new(-14.0, 0.5, -8.0), DVec3::new(14.0, 0.5, -8.0), DVec3::new(-29.0, 1.0, -20.0), DVec3::new(29.0, 1.0, -20.0)];
                let err = flown.pos + flown.rot * PLACES[n] - j.pos;
                let lead = flown.vel.normalize_or(fwd);
                let lead_speed = flown.vel.length();
                if err.length() > 1500.0 {
                    // a long way off: straight for it
                    want = (err + flown.vel * 3.0).normalize_or(fwd);
                    want_speed = lead_speed + 250.0;
                } else {
                    // the leader's way, and sideways into place; ahead of its place it slows, behind it hurries
                    let along = err.dot(lead);
                    let across = (err - lead * along).clamp_length_max(160.0);
                    want = (flown.vel + across * 0.9).normalize_or(fwd);
                    want_speed = lead_speed + (along * 0.8).clamp(-110.0, 250.0);
                }
                limits = (11.0, 75.0, 900.0);
                aim = aim.filter(|a| a.1 < 1500.0);
            }
            if hunted {
                // a rocket: put it on the beam and go down, as fast as it will go
                let from = (j.pos - j.threat).normalize_or(fwd);
                let side = from.cross(up).normalize_or(fwd) * if fwd.dot(from.cross(up)) < 0.0 { -1.0 } else { 1.0 };
                want = (side - up * 0.25).normalize();
                want_speed = 430.0;
                if now >= j.flare {
                    j.flare = now + 0.6;
                    j.flared = now;
                    self.sparks.push(Spark { kind: SparkKind::Flare, pos: j.pos - j.rot.y_axis * 1.0, vel: j.vel * 0.9 - j.rot.y_axis * 25.0, born: now, life: 3.5, target: None, team: j.team, mine: false });
                }
            }
            // Back towards the middle, the more the further out it is: left to themselves they
            // all chase the nearest, and the whole fight drifts off as one swarm.
            let out = j.pos - centre;
            if place.is_none() && !hunted && out.length() > 0.45 * radius {
                let pull = ((out.length() / radius - 0.45) / 0.55).min(1.6);
                want = (want - out.normalize() * 1.4 * pull).normalize_or(want);
            }
            // the ground: it climbs away from it; and there is a ceiling
            let floor = 420.0 + 3.0 * (-j.vel.dot(up)).max(0.0);
            if above < floor {
                want = ((fwd - up * fwd.dot(up)).normalize_or(fwd) + up * (1.2 + (floor - above) / floor)).normalize();
            } else if above < 2.0 * floor && want.dot(up) < 0.0 {
                want = (want - up * want.dot(up)).normalize_or(fwd);
            }
            let high = j.pos.length() - centre.length();
            if place.is_none() && high > 1500.0 && want.dot(up) > -0.1 {
                want = (want - up * (want.dot(up) + 0.15)).normalize_or(fwd);
            } else if place.is_none() && high < -1500.0 && want.dot(up) < 0.1 {
                want = (want - up * (want.dot(up) - 0.15)).normalize_or(fwd);
            }
            // ---- turn that way at the g it can pull, and bank as the turn asks
            let (g_max, thrust, fastest) = limits;
            let new = turn(fwd, want, g_max * G / speed * dt);
            let pull = (new - fwd) * speed / dt;
            let push = ((want_speed - speed) * if place.is_some() { 3.0 } else { 1.0 }).clamp(-thrust - 4.0, thrust);
            let speed = (speed + (push - if place.is_some() { 0.0 } else { 0.5 * G * new.dot(up) }) * dt).clamp(150.0, fastest);
            let hard = pull.length() > 3.0 * G;
            j.pulling = pull.length() > 6.5 * G;
            j.ab += ((if push > 8.0 || (hard && speed < 330.0) { 1.0 } else { 0.0 }) - j.ab) * (1.0 - (-dt * 2.5).exp());
            let lift = pull + up * G;
            let mut top = (lift - new * lift.dot(new)).normalize_or(up);
            if place.is_some() && pull.length() < 2.0 * G {
                top = (top + flown.rot.y_axis * 2.0).normalize_or(top); // wings as the leader's
            }
            j.up = (j.up + (top - j.up) * (1.0 - (-dt * 4.0).exp())).normalize_or(top);
            j.vel = new * speed;
            j.pos += j.vel * dt;
            j.rot = rot_of(new, j.up);
            if above < 4.0 {
                downed.push(i); // into the ground
                continue;
            }
            // ---- its weapons
            if let Some((t, dist, on)) = aim {
                if rocket_room && !hunted && now >= j.fire && dist > 450.0 && dist < 5500.0 && on > 0.97 {
                    j.fire = now + 5.0 + 7.0 * self.rng.next();
                    launches.push((i, t));
                }
                if dist < 800.0 && on > 0.9982 && now >= j.gun {
                    j.gun = now + 0.07;
                    let scatter = self.rng.dir() * 0.004;
                    let pos = j.pos + j.rot * DVec3::new(-0.6, 0.45, -3.0);
                    self.sparks.push(Spark { kind: SparkKind::Tracer, pos, vel: j.vel + (new + scatter).normalize() * TRACER_SPEED, born: now, life: 1.1, target: Some(Target::Jet(t)), team: j.team, mine: false });
                }
            }
        }
        for i in downed {
            let (pos, late) = (self.jets[i].pos, now - self.jets[i].since > 40.0);
            if !late {
                self.blast(pos, DVec3::ZERO, 1.8);
            }
            self.jets[i].state = State::Out;
            self.jets[i].since = now;
        }
        for (i, t) in launches {
            self.launch(i, Some(Target::Jet(t)), false);
        }
        self.ground_war(dt, flown, ground, water, &seen, shielded);
        // ---- the pilot's rockets
        if flown.fire && now >= self.reload && self.jets[0].state == State::Flying {
            self.reload = now + 0.45;
            let target = self.in_sights().map(|t| t.0);
            self.launch(0, target, true);
        }
        // ---- rockets: they turn after their target, and go off beside it
        let mut hits: Vec<(Target, u8, bool)> = Vec::new();
        let mut pops: Vec<(DVec3, DVec3, f64)> = Vec::new();
        for r in self.rockets.iter_mut() {
            let age = now - r.born;
            let speed = (r.vel.length() + 450.0 * dt).min(ROCKET_SPEED);
            let mut dir = r.vel.normalize_or(DVec3::X);
            let before = r.pos;
            if let Some(t) = r.target {
                let (tp, tv, live) = match t {
                    Target::Jet(k) => (self.jets[k].pos, self.jets[k].vel, self.jets[k].state == State::Flying && !(k == 0 && shielded)),
                    Target::Drone(k) => (self.drones[k].pos, self.drones[k].vel, self.drones[k].alive),
                };
                if !live {
                    r.target = None;
                } else {
                    let d = tp - r.pos;
                    let dist = d.length();
                    if age > 0.25 {
                        let closing = (r.vel - tv).dot(d / dist.max(1.0)).max(150.0);
                        let lead = tp + tv * (dist / closing).min(4.0);
                        dir = turn(dir, (lead - r.pos).normalize_or(dir), 24.0 * G / speed * dt);
                    }
                    if let Target::Jet(k) = t {
                        let j = &mut self.jets[k];
                        if dist < 3200.0 {
                            (j.threat, j.threat_until) = (r.pos, now + 0.6);
                        }
                        // (a flare thrown in front of it may take it)
                        if dist < 2500.0 && j.flared == now && (r.born * 977.0 + now * 131.0).fract() < 0.10 {
                            r.target = None;
                        }
                    }
                }
            }
            r.vel = dir * speed;
            r.pos += r.vel * dt;
            r.trail.leave(r.pos - dir * 1.6, now);
            if let Some(t) = r.target {
                // how near it came to its target in this step
                let (tp, tv) = match t {
                    Target::Jet(k) => (self.jets[k].pos, self.jets[k].vel),
                    Target::Drone(k) => (self.drones[k].pos, self.drones[k].vel),
                };
                let (a, b) = (before - (tp - tv * dt), r.pos - tp);
                let ab = b - a;
                let s = if ab.length_squared() > 1e-9 { (-a.dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
                if (a + ab * s).length() < 14.0 {
                    hits.push((t, r.team, r.mine));
                    r.born = -1e9;
                    continue;
                }
            }
            if now >= r.ground_at {
                r.ground = ground(r.pos);
                r.ground_at = now + 0.2;
            }
            if r.pos.length() < r.ground + 2.0 {
                pops.push((r.pos, DVec3::ZERO, 0.7));
                r.born = -1e9;
            } else if age > ROCKET_LIFE || (r.target.is_none() && age > 3.5) {
                pops.push((r.pos, r.vel * 0.3, 0.3));
                r.born = -1e9;
            }
        }
        for (t, team, mine) in hits {
            match t {
                Target::Jet(k) => self.destroy(k, Some(team), mine),
                Target::Drone(k) => self.drone_down(k, mine),
            }
        }
        for (pos, vel, size) in pops {
            self.blast(pos, vel, size);
        }
        let mut k = 0;
        while k < self.rockets.len() {
            if self.rockets[k].born < -1e8 {
                let r = self.rockets.swap_remove(k);
                self.ghosts.push(r.trail);
            } else {
                k += 1;
            }
        }
        // ---- tracers, flares and burning pieces
        let mut struck: Vec<(Target, u8, bool)> = Vec::new();
        for s in self.sparks.iter_mut() {
            let up = s.pos.normalize();
            match s.kind {
                SparkKind::Tracer => {
                    let before = s.pos;
                    s.pos += s.vel * dt;
                    // (where what it was fired at is, and how near is a hit)
                    let aim = match s.target {
                        Some(Target::Jet(k)) if self.jets[k].state == State::Flying => Some((self.jets[k].pos, self.jets[k].vel, 7.0)),
                        Some(Target::Drone(k)) if self.drones[k].alive => Some((self.drones[k].pos, self.drones[k].vel, 5.0)),
                        _ => None,
                    };
                    if let (Some((tp, tv, reach)), Some(what)) = (aim, s.target) {
                        let (a, b) = (before - (tp - tv * dt), s.pos - tp);
                        let ab = b - a;
                        let t = (-a.dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                        if (a + ab * t).length() < reach {
                            struck.push((what, s.team, s.mine));
                            s.life = 0.0;
                        }
                    }
                }
                _ => {
                    let speed = s.vel.length();
                    s.vel += (-up * G - s.vel * speed * if s.kind == SparkKind::Flare { 0.004 } else { 0.002 }) * dt;
                    s.pos += s.vel * dt;
                }
            }
        }
        self.sparks.retain(|s| now - s.born < s.life);
        for (what, team, mine) in struck {
            // (one round in three does harm; every one of the pilot's own does)
            let harm = mine || self.rng.next() < 0.33;
            match what {
                Target::Jet(t) if harm && !(t == 0 && shielded) => {
                    self.jets[t].hp -= 1;
                    if mine {
                        // (and is seen to)
                        let (pos, vel) = (self.jets[t].pos, self.jets[t].vel);
                        let v = vel + self.rng.dir() * 40.0;
                        self.sparks.push(Spark { kind: SparkKind::Debris, pos, vel: v, born: now, life: 0.7, target: None, team: 0, mine: false });
                    }
                    if self.jets[t].hp <= 0 {
                        self.destroy(t, Some(team), mine);
                    }
                }
                Target::Drone(k) if harm => self.drone_down(k, mine),
                _ => {}
            }
        }
        // ---- smoke; and those that are out come back at the edge of the fight
        for i in 0..self.jets.len() {
            let j = &mut self.jets[i];
            match j.state {
                State::Flying if j.hp < 3 => j.trail.leave(j.pos + j.rot * NOZZLE, now),
                State::Falling => j.trail.leave(j.pos, now),
                State::Out if i > 0 && now - j.since > 3.0 + 4.0 * j.skill => {
                    let team = j.team;
                    let old = std::mem::replace(&mut j.trail, Trail::new(Smoke::Damage, 0.0));
                    self.ghosts.push(old);
                    let c = self.centre;
                    let up = c.normalize();
                    // (more of them ahead of the pilot than behind)
                    let bearing = self.rng.dir() + flown.vel.normalize_or(DVec3::ZERO) * 0.7;
                    let out = (bearing - up * bearing.dot(up)).normalize_or(up.any_orthonormal_vector());
                    let p = c + out * self.radius * 0.95;
                    let p = p.normalize() * (c.length() + self.rng.range(-800.0, 800.0)).max(ground(p) + 600.0);
                    let v = (-out) * (self.rng.range(270.0, 340.0) + pace);
                    let mut fresh = self.fresh(p, v, team);
                    fresh.ground = ground(p);
                    self.jets[i] = fresh;
                }
                _ => {}
            }
            let j = &mut self.jets[i];
            j.trail.age(now);
            for (side, v) in [4.75, -4.75].into_iter().zip(j.vapour.iter_mut()) {
                if j.state == State::Flying && j.pulling && i > 0 {
                    v.leave(j.pos + j.rot * DVec3::new(side, 0.04, 2.1), now);
                }
                v.age(now);
            }
        }
        for r in self.rockets.iter_mut() {
            r.trail.age(now);
        }
        for g in self.ghosts.iter_mut() {
            g.age(now);
        }
        self.ghosts.retain(|g| !g.pts.is_empty());
        self.blasts.retain(|b| now - b.born < if b.size < 0.5 { 5.0 } else { 15.0 });
        if now >= self.report {
            self.report = now + 30.0;
            let board: Vec<String> = (0..8).map(|t| format!("{} {}", SQUADRONS[t], self.scores[t])).collect();
            self.news.push(format!("{:.0} s: {} flying, {} rockets in the air. Shot down by: {}", now, self.flying(), self.rockets.len(), board.join(", ")));
        }
    }

    /// Everything for this frame's picture: the other aircraft and the rockets, and the smoke and
    /// fire (`frame`: Earth axes -> the renderer's frame at the camera, `cam` the camera's Earth
    /// position).
    pub fn draw(&mut self, frame: DMat3, cam: DVec3, crafts: &mut Vec<Craft>, fx: &mut Vec<FxGpu>) {
        crafts.clear();
        fx.clear();
        let now = self.time;
        let to = |p: DVec3| (frame * (p - cam)).as_vec3();
        let place = |r: DMat3, p: DVec3| {
            let m = frame * r;
            Affine3A::from_mat3_translation(Mat3::from_cols(m.x_axis.as_vec3(), m.y_axis.as_vec3(), m.z_axis.as_vec3()), to(p))
        };
        let pose = |rot: DMat3, pos: DVec3, prev: &mut Option<(DMat3, DVec3)>| {
            let xform = place(rot, pos);
            let (pr, pp) = prev.unwrap_or((rot, pos));
            *prev = Some((rot, pos));
            Pose { xform, back: place(pr, pp) * xform.inverse(), lef: 0.0, flaperon: [0.0; 2], tail: [0.0; 2], rudder: 0.0 }
        };
        let ball = |p: Vec3, r: f32, c: [f32; 3], bright: f32, dens: f32, kind: f32, seed: f32, ragged: f32| FxGpu { a: [p.x, p.y, p.z, r], b: [p.x, p.y, p.z, r], colour: [c[0] * bright, c[1] * bright, c[2] * bright, dens], kind: [kind, seed, 0.0, ragged] };
        let tube = |a: Vec3, ra: f32, b: Vec3, rb: f32, c: [f32; 3], bright: f32, dens: f32, kind: f32, seed: f32| FxGpu { a: [a.x, a.y, a.z, ra], b: [b.x, b.y, b.z, rb], colour: [c[0] * bright, c[1] * bright, c[2] * bright, dens], kind: [kind, seed, 0.0, 0.0] };
        for (i, j) in self.jets.iter_mut().enumerate() {
            if j.state == State::Out {
                continue;
            }
            let head = match j.state {
                State::Flying if j.hp < 3 => Some(j.pos + j.rot * NOZZLE),
                State::Falling => Some(j.pos),
                _ => None,
            };
            j.trail.draw(head, now, &to, fx);
            if j.state == State::Falling {
                // it burns
                let flick = 0.8 + 0.4 * ((now * 23.0 + j.skill * 9.0).sin() * (now * 13.7).sin()) as f32;
                fx.push(ball(to(j.pos), 4.2 * flick, [1.0, 0.36, 0.07], 3.2, 0.45, 1.0, (now * 3.0).fract() as f32, 0.8));
            }
            if i == 0 {
                continue; // (the pilot's own aircraft is drawn by the pilot)
            }
            crafts.push(Craft { pose: pose(j.rot, j.pos, &mut j.prev), livery: j.team as u32, kind: KIND_JET, tint: [1.0; 3] });
            if j.state == State::Flying && j.ab > 0.05 {
                let ab = j.ab as f32;
                let (a, aft) = (j.pos + j.rot * NOZZLE, j.rot.z_axis);
                fx.push(tube(to(a), 0.30, to(a + aft * (2.5 + 6.5 * j.ab)), 0.09, [1.0, 0.42, 0.13], 0.8 + 2.0 * ab, 7.0, 1.0, j.skill as f32));
            }
        }
        // the trucks; one that is burnt out is black, and burns
        for (i, s) in self.sites.iter_mut().enumerate() {
            if !s.placed {
                continue;
            }
            let kind = if s.kind == SiteKind::Launcher { KIND_LAUNCHER } else { KIND_GUN };
            let tint = if s.burning.is_some() { [0.03; 3] } else { COLOURS[s.team as usize] };
            crafts.push(Craft { pose: pose(s.rot, s.pos, &mut s.prev), livery: 0, kind, tint });
            if s.burning.is_some() {
                let flick = 0.8 + 0.4 * ((now * 19.0 + i as f64).sin() * (now * 12.3).sin()) as f32;
                fx.push(ball(to(s.pos + s.up * 2.0), 3.6 * flick, [1.0, 0.38, 0.07], 3.2, 0.5, 1.0, (now * 2.0).fract() as f32, 0.8));
                // a column of smoke, going up and leaning with the wind
                let lean = s.up.any_orthonormal_vector();
                for k in 0..5 {
                    let h = (now * 0.12 + k as f64 / 5.0 + i as f64 * 0.37).fract();
                    let r = 4.0 + 12.0 * h;
                    let at = s.pos + s.up * (5.0 + 75.0 * h) + lean * 28.0 * h * h;
                    fx.push(ball(to(at), r as f32, [0.04; 3], 1.0, (0.9 * (1.0 - h) * (h * 8.0).min(1.0) / r) as f32, 0.0, k as f32 * 0.21, 0.8));
                }
            }
        }
        for r in self.rockets.iter_mut() {
            let dir = r.vel.normalize_or(DVec3::X);
            let rot = rot_of(dir, r.pos.normalize());
            crafts.push(Craft { pose: pose(rot, r.pos, &mut r.prev), livery: 0, kind: KIND_ROCKET, tint: [1.0; 3] });
            let tail = r.pos + rot * ROCKET_TAIL;
            r.trail.draw(Some(tail), now, &to, fx);
            fx.push(tube(to(tail), 0.26, to(tail - dir * 5.0), 0.07, [1.0, 0.66, 0.36], 5.0, 9.0, 1.0, 0.0));
            fx.push(ball(to(tail - dir * 0.8), 1.3, [1.0, 0.6, 0.3], 2.0, 0.7, 1.0, 0.0, 0.0));
        }
        // (after the rockets: the drones are the first to go when there are more than the renderer takes)
        for d in self.drones.iter_mut() {
            if d.alive {
                let rot = rot_of(d.vel.normalize_or(DVec3::X), d.pos.normalize());
                crafts.push(Craft { pose: pose(rot, d.pos, &mut d.prev), livery: 0, kind: KIND_DRONE, tint: COLOURS[self.swarms[d.swarm].team as usize] });
            }
        }
        for g in &self.ghosts {
            g.draw(None, now, &to, fx);
        }
        for b in &self.blasts {
            let (age, s) = (now - b.born, b.size);
            let up = b.pos.normalize();
            let at = b.pos + b.vel * (1.0 - (-2.0 * age).exp()) * 0.5;
            let burn = 1.3 * s.sqrt();
            if age < burn {
                // the fireball: white, then orange, then red, and gone
                let r = s * (4.0 + 15.0 * (1.0 - (-age * 5.0).exp()));
                let k = (age / burn) as f32;
                let c = [1.0, 0.58 - 0.34 * k, 0.22 - 0.18 * k];
                let bright = 5.5 * (-2.2 * k).exp() * (1.0 - k);
                fx.push(ball(to(at), r as f32, c, bright, 1.6 / r as f32, 1.0, b.seed, 0.75));
                if age < 0.14 {
                    fx.push(ball(to(at), 0.5 * r as f32, [1.0, 0.88, 0.62], 12.0 * (1.0 - age as f32 / 0.14), 2.5 / r as f32, 1.0, b.seed + 0.3, 0.0));
                }
            }
            // and the smoke it leaves, which spreads, rises a little and thins
            // (a small one leaves one puff, soon gone)
            let (puffs, life) = if s < 0.5 { (1, 5.0) } else { (4, 15.0) };
            let fade = ((1.0 - age / life).clamp(0.0, 1.0).powf(1.5) * (age * 3.0).min(1.0)) as f32;
            for (n, p) in b.puffs.iter().take(puffs).enumerate() {
                let r = s * (5.0 + 9.0 * (1.0 - (-age * 1.2).exp()) + 0.7 * age);
                let c = at + *p * s * 7.0 * (1.0 - (-age * 2.0).exp()) + up * 1.6 * age;
                let shade = 0.035 + 0.02 * n as f32;
                fx.push(ball(to(c), r as f32, [shade, shade, shade], 1.0, 1.2 * fade / r as f32, 0.0, b.seed + 0.17 * n as f32, 0.85));
            }
        }
        for s in &self.sparks {
            let age = ((now - s.born) / s.life) as f32;
            let p = to(s.pos);
            match s.kind {
                SparkKind::Tracer => {
                    // (it lights a few metres out of the muzzle, and its streak grows from there:
                    // not a bar of light through the gun and whoever holds it)
                    let gone = (now - s.born) * TRACER_SPEED - 7.0;
                    if gone > 1.0 {
                        // (the pilot's own are seen end on: thinner, or they are one bright ball)
                        let w = if s.mine { 0.3 } else { 1.0 };
                        let back = to(s.pos - s.vel.normalize_or(DVec3::X) * gone.min(32.0));
                        fx.push(tube(p, 0.20 * w, back, 0.10 * w, [1.0, 0.50, 0.14], 5.0, 12.0 / w, 1.0, 0.0));
                    }
                }
                SparkKind::Flare => {
                    let flick = 0.8 + 0.3 * ((now * 41.0 + s.born * 7.0).sin() as f32);
                    fx.push(ball(p, 0.9 * flick, [1.0, 0.78, 0.48], 10.0 * (1.0 - age * age), 2.5, 1.0, 0.0, 0.0));
                    let back = to(s.pos - s.vel * 0.45);
                    fx.push(tube(p, 0.4, back, 1.5, [0.8, 0.8, 0.8], 1.0, 0.22 * (1.0 - age), 0.0, s.born as f32 % 1.0));
                }
                SparkKind::Debris => {
                    fx.push(ball(p, 0.7, [1.0, 0.45, 0.10], 3.5 * (1.0 - age), 2.0, 1.0, 0.0, 0.0));
                    let back = to(s.pos - s.vel * 0.3);
                    fx.push(tube(p, 0.5, back, 1.4, [0.10, 0.10, 0.10], 1.0, 0.5 * (1.0 - age), 0.0, s.born as f32 % 1.0));
                }
            }
        }
        // (last, the wing tips' vapour: the first to go when there is more than the renderer takes)
        for j in self.jets.iter().skip(1) {
            for (side, v) in [4.75, -4.75].into_iter().zip(&j.vapour) {
                let head = (j.state == State::Flying && j.pulling).then(|| j.pos + j.rot * DVec3::new(side, 0.04, 2.1));
                v.draw(head, now, &to, fx);
            }
        }
    }

    /// A testing aid (SI_BATTLE team=3): the pilot flies with this squadron from the start.
    pub fn set_team(&mut self, team: u8) {
        self.jets[0].team = team % 8;
    }

    /// A camera that watches the fight instead of the pilot's aircraft: it flies behind one of
    /// the others that is after somebody, and moves on to another when that one is shot down or
    /// after a while.
    ///
    /// `what`: 1 the aircraft, 2 the drones (it flies with a swarm), 3 the trucks (it goes round
    /// one that is firing).
    pub fn watch(&mut self, dt: f64, what: u8) -> crate::pilot::Camera {
        let now = self.time;
        if what != self.watch_what {
            (self.watch_what, self.watching) = (what, None);
        }
        if what >= 2 {
            if let Some(c) = self.watch_below(what) {
                return c;
            }
        }
        let stale = match self.watching {
            Some((k, since)) => {
                let j = &self.jets[k];
                j.state == State::Out || (j.state == State::Falling && now - j.since > 3.5) || (j.state == State::Flying && (now - since > 12.0 || j.since > since))
            }
            None => true,
        };
        if stale {
            let near = |b: &Battle, k: usize| {
                let j = &b.jets[k];
                j.state == State::Flying && j.target.map(|t| (b.jets[t].pos - j.pos).length() < 1800.0).unwrap_or(false)
            };
            let mut c: Vec<usize> = (1..self.jets.len()).filter(|&k| near(self, k)).collect();
            if c.is_empty() {
                c = (0..self.jets.len()).filter(|&k| self.jets[k].state == State::Flying).collect();
            }
            let k = if c.is_empty() { 0 } else { c[(self.rng.next() * c.len() as f64) as usize % c.len()] };
            self.watching = Some((k, now));
            self.watch_at = DVec3::ZERO;
        }
        let j = &self.jets[self.watching.unwrap().0];
        let up = j.pos.normalize();
        let fwd = j.vel.normalize_or(-j.rot.z_axis);
        let side = fwd.cross(up).normalize_or(j.rot.x_axis);
        let want = j.pos - fwd * 34.0 + up * 8.0 + side * 13.0;
        // (it follows with a lag; a wreck it lets fall away)
        let rate: f64 = if j.state == State::Flying { 6.0 } else { 0.0 };
        self.watch_at = if self.watch_at == DVec3::ZERO { want } else { self.watch_at + j.vel * dt * rate.min(1.0) + (want - self.watch_at) * (1.0 - (-dt * rate).exp()) };
        let mut look = j.pos + fwd * 12.0;
        if let Some(t) = j.target.filter(|&t| (self.jets[t].pos - j.pos).length() < 900.0) {
            look = look * 0.7 + self.jets[t].pos * 0.3;
        }
        let f = (look - self.watch_at).normalize_or(fwd);
        let r = f.cross(up).normalize_or(side);
        crate::pilot::Camera { pos: self.watch_at, right: r, up: r.cross(f), fwd: f, fov_y: 52f32.to_radians() }
    }

    /// How close the fight is to the pilot: aircraft flying within `d` metres, and how far each
    /// of the formation is.
    pub fn near(&self, d: f64) -> (usize, Vec<f64>) {
        let at = self.jets[0].pos;
        let n = self.jets.iter().skip(1).filter(|j| j.state == State::Flying && (j.pos - at).length() < d).count();
        (n, self.wing.iter().map(|&k| (self.jets[k].pos - at).length().round()).collect())
    }

    /// The war below, in numbers: drones flying, trucks standing.
    pub fn below(&self) -> (usize, usize) {
        (self.drones.iter().filter(|d| d.alive).count(), self.sites.iter().filter(|s| s.placed && s.burning.is_none()).count())
    }

    /// The watching camera on the war below: with a swarm of drones, or round a truck.
    fn watch_below(&mut self, what: u8) -> Option<crate::pilot::Camera> {
        let now = self.time;
        let view = |at: DVec3, look: DVec3| {
            let up = at.normalize();
            let f = (look - at).normalize_or(up.any_orthonormal_vector());
            let r = f.cross(up).normalize_or(up.any_orthonormal_vector());
            crate::pilot::Camera { pos: at, right: r, up: r.cross(f), fwd: f, fov_y: 55f32.to_radians() }
        };
        let turn = (now as usize) / 14;
        if what == 2 {
            if self.watching.map(|w| w.0 >= self.swarms.len() || now - w.1 > 14.0).unwrap_or(true) {
                // a full swarm, one with a truck to go for if there is one; another each time
                let full = |w: usize| self.drones.iter().filter(|d| d.swarm == w && d.alive).count();
                let best = (0..self.swarms.len()).max_by_key(|&w| full(w) + if self.swarms[w].goal.is_some() { 50 } else { 0 } + (w * 7 + turn * 5) % 23)?;
                self.watching = Some((best, now));
            }
            let sw = &self.swarms[self.watching?.0];
            let up = sw.pos.normalize();
            let level = (sw.vel - up * sw.vel.dot(up)).normalize_or(up.any_orthonormal_vector());
            Some(view(sw.pos - level * 92.0 + up * 22.0 + level.cross(up) * 26.0, sw.pos - level * 26.0 + up * 5.0))
        } else {
            let busy = |s: &Site| s.placed && s.burning.is_none() && s.aim.is_some();
            if self.watching.map(|w| w.0 >= self.sites.len() || !self.sites[w.0].placed || now - w.1 > 14.0).unwrap_or(true) {
                let best = (0..self.sites.len()).filter(|&k| self.sites[k].placed).max_by_key(|&k| if busy(&self.sites[k]) { 100 } else { 0 } + (k * 7 + turn * 5) % 23)?;
                self.watching = Some((best, now));
            }
            let (s, since) = (&self.sites[self.watching?.0], self.watching?.1);
            // round it, slowly
            let a = 0.22 * (now - since) + 0.6;
            let (x, z) = (s.up.any_orthonormal_vector(), s.up.cross(s.up.any_orthonormal_vector()));
            Some(view(s.pos + (x * a.cos() + z * a.sin()) * 21.0 + s.up * 6.0, s.pos + s.up * 3.0))
        }
    }

    /// The fight, for the window's title.
    pub fn hud(&self) -> String {
        let me = &self.jets[0];
        let lock = match self.in_sights() {
            Some((what, dist, team)) => format!("{}{} in sights {:.1} km", SQUADRONS[team as usize], if matches!(what, Target::Drone(_)) { " drone" } else { "" }, dist / 1000.0),
            None => "nothing in sights".to_string(),
        };
        format!(
            "{} sqn{} | {} down, lost {} | {}{} | {} flying, {} drones, {} trucks, {} rockets",
            SQUADRONS[me.team as usize],
            if me.hp < 3 { " (hit)" } else { "" },
            self.kills,
            self.lost,
            lock,
            if self.time < me.threat_until { " | ROCKET" } else { "" },
            self.flying(),
            self.below().0,
            self.below().1,
            self.rockets.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::globe::tiles::R_EARTH;

    fn run(n: usize, seconds: f64) -> Battle {
        let centre = DVec3::X * (R_EARTH + 3000.0);
        let ground = |_: DVec3| R_EARTH;
        let mut b = Battle::new(centre, 0.0, n, 0, 0, &ground);
        b.immortal = true;
        let flown = Flown { pos: centre, vel: DVec3::Z * 230.0, rot: rot_of(DVec3::Z, DVec3::X), fire: false };
        for _ in 0..(seconds * 60.0) as usize {
            b.step(1.0 / 60.0, &flown, &ground, &|_| false);
        }
        b
    }

    #[test]
    fn they_fight_and_none_is_lost() {
        // a minute and a half of forty aircraft: rockets fly, some are shot down, and every one
        // is still in the air, falling, or on its way back
        let mut b = run(40, 90.0);
        assert_eq!(b.jets.len(), 41);
        let shot: u32 = b.scores.iter().sum();
        assert!(shot >= 3, "only {shot} shot down");
        assert!(b.flying() >= 20, "{} flying", b.flying());
        for j in &b.jets {
            assert!(j.pos.is_finite() && j.vel.is_finite());
            // none has gone through the ground, or off to somewhere else
            assert!(j.state != State::Flying || j.pos.length() > R_EARTH, "below ground");
            assert!((j.pos - b.centre).length() < 60_000.0);
        }
        // and the picture of it stays within what the renderer takes
        let (mut crafts, mut fx) = (Vec::new(), Vec::new());
        b.draw(DMat3::IDENTITY, b.centre, &mut crafts, &mut fx);
        assert!(crafts.len() >= 20 && crafts.len() < crate::renderer::MAX_CRAFT);
        assert!(!fx.is_empty() && fx.len() < crate::renderer::MAX_FX);
        assert!(fx.iter().all(|f| f.a.iter().chain(&f.b).chain(&f.colour).all(|v| v.is_finite())));
    }

    #[test]
    fn shot_down_the_pilot_flies_another() {
        let mut b = run(24, 2.0);
        b.immortal = false;
        b.shield = 0.0;
        let (team, at) = (b.team(), b.jets[0].pos);
        b.destroy(0, Some(3), false);
        assert!(b.down() && b.lost == 1);
        let (pos, heading, speed) = b.take_over();
        assert!(!b.down());
        assert!(b.team() != team, "the same squadron");
        assert!((pos - at).length() > 0.5 * b.radius, "only {:.0} m away", (pos - at).length());
        assert!((0.0..360.0).contains(&heading) && speed >= 180.0);
        // the old aircraft is still there, falling
        assert_eq!(b.jets.iter().filter(|j| j.state == State::Falling).count(), 1);
        assert_eq!(b.jets.len(), 25);
    }

    #[test]
    fn the_squadron_keeps_formation() {
        // nobody to fight: four of the pilot's squadron close up beside the pilot, and stay
        // there through a turn
        let centre = DVec3::X * (R_EARTH + 3000.0);
        let ground = |_: DVec3| R_EARTH;
        let mut b = Battle::new(centre, 0.0, 16, 0, 0, &ground);
        b.jets.iter_mut().for_each(|j| j.team = 0);
        let mut flown = Flown { pos: centre, vel: DVec3::Z * 250.0, rot: rot_of(DVec3::Z, DVec3::X), fire: false };
        for k in 0..60 * 40 {
            if k > 60 * 25 {
                // a gentle turn to the right (about 2 g)
                let fwd = turn(flown.vel.normalize(), flown.vel.normalize().cross(DVec3::X), 0.07 / 60.0);
                flown.vel = fwd * 250.0;
                flown.rot = rot_of(fwd, DVec3::X);
            }
            flown.pos += flown.vel / 60.0;
            b.step(1.0 / 60.0, &flown, &ground, &|_| false);
            if k == 60 * 25 || k == 60 * 40 - 1 {
                assert_eq!(b.wing.len(), 4);
                for &w in &b.wing {
                    let d = (b.jets[w].pos - flown.pos).length();
                    assert!(d > 12.0 && d < 90.0, "wingman {d:.0} m from the leader at step {k}");
                }
            }
        }
    }

    #[test]
    fn the_war_below() {
        // two minutes of 24 aircraft, 100 drones and 24 trucks over flat land: the trucks are
        // stood on the ground, rockets come up from them, drones fly into them and some burn
        // out; and there are as many drones and trucks at the end as at the start
        let centre = DVec3::X * (R_EARTH + 3000.0);
        let ground = |_: DVec3| R_EARTH;
        let mut b = Battle::new(centre, 0.0, 24, 100, 24, &ground);
        b.immortal = true;
        let mut flown = Flown { pos: centre, vel: DVec3::Z * 230.0, rot: rot_of(DVec3::Z, DVec3::X), fire: false };
        for _ in 0..120 * 60 {
            flown.pos += flown.vel / 60.0;
            b.step(1.0 / 60.0, &flown, &ground, &|_| false);
        }
        assert_eq!((b.drones.len(), b.sites.len()), (100, 24));
        assert!(b.sites.iter().filter(|s| s.placed).count() >= 20, "trucks not placed");
        for s in b.sites.iter().filter(|s| s.placed) {
            assert!((s.pos.length() - R_EARTH).abs() < 0.5, "a truck off the ground");
        }
        assert!(b.from_ground >= 3, "{} rockets from the ground", b.from_ground);
        assert!(b.razed >= 1, "no truck burnt out");
        assert!(b.below().0 >= 50, "{} drones flying", b.below().0);
        assert!(b.drones.iter().all(|d| d.pos.is_finite() && d.vel.is_finite() && (!d.alive || d.pos.length() > R_EARTH)));
        let (mut crafts, mut fx) = (Vec::new(), Vec::new());
        b.draw(DMat3::IDENTITY, flown.pos, &mut crafts, &mut fx);
        assert!(crafts.iter().filter(|c| c.kind == KIND_DRONE).count() >= 50);
        assert!(crafts.iter().any(|c| c.kind == KIND_LAUNCHER) && crafts.iter().any(|c| c.kind == KIND_GUN));
    }

    #[test]
    fn no_truck_stands_on_water() {
        let centre = DVec3::X * (R_EARTH + 3000.0);
        let ground = |_: DVec3| R_EARTH;
        let mut b = Battle::new(centre, 0.0, 0, 0, 16, &ground);
        let flown = Flown { pos: centre, vel: DVec3::Z * 230.0, rot: rot_of(DVec3::Z, DVec3::X), fire: false };
        for _ in 0..120 {
            b.step(1.0 / 60.0, &flown, &ground, &|_| true);
        }
        assert!(b.sites.iter().all(|s| !s.placed));
    }

    #[test]
    fn out_of_the_aircraft_and_shooting() {
        // the pilot gets out: the aircraft falls, nothing is after the pilot, and a round fired
        // from the ground at an aircraft overhead brings it down in the end
        let centre = DVec3::X * (R_EARTH + 3000.0);
        let ground = |_: DVec3| R_EARTH;
        let mut b = Battle::new(centre, 0.0, 0, 0, 0, &ground);
        let mut j = b.fresh(centre + DVec3::Z * 600.0 + DVec3::X * 200.0, DVec3::Y * 200.0, 3);
        (j.think, j.fire, j.flare) = (1e9, 1e9, 1e9);
        b.jets.push(j);
        b.eject();
        assert!(b.down() && b.jets.iter().filter(|j| j.state == State::Falling).count() == 1);
        let flown = Flown { pos: centre, vel: DVec3::ZERO, rot: rot_of(DVec3::Z, DVec3::X), fire: false };
        for _ in 0..240 {
            // (pointed at where it is, not ahead of it: the help does the rest)
            let at = b.jets[1].pos;
            b.round(centre, (at - centre).normalize(), DVec3::ZERO);
            b.step(1.0 / 60.0, &flown, &ground, &|_| false);
            if b.kills > 0 {
                break;
            }
        }
        assert_eq!(b.kills, 1);
        let (pos, ..) = b.take_over();
        assert!(!b.down() && pos.is_finite());
    }

    #[test]
    fn a_rocket_reaches_what_it_is_fired_at() {
        let centre = DVec3::X * (R_EARTH + 3000.0);
        let ground = |_: DVec3| R_EARTH;
        let mut b = Battle::new(centre, 0.0, 0, 0, 0, &ground);
        // one aircraft of another squadron two kilometres ahead, flying straight on
        let mut j = b.fresh(centre + DVec3::Z * 2000.0, DVec3::Z * 250.0, 3);
        (j.think, j.fire, j.flare) = (1e9, 1e9, 1e9);
        b.jets.push(j);
        let mut flown = Flown { pos: centre, vel: DVec3::Z * 250.0, rot: rot_of(DVec3::Z, DVec3::X), fire: true };
        for k in 0..600 {
            flown.pos += flown.vel / 60.0;
            flown.fire = k == 0;
            b.step(1.0 / 60.0, &flown, &ground, &|_| false);
        }
        assert_eq!(b.kills, 1);
    }
}
