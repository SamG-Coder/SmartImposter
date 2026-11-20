//! A game controller for `--test-flight`. The layout is Grand Theft Auto V's for aircraft: the
//! right trigger opens the throttle, the left closes it; the left stick is the stick (forward:
//! nose down); the bumpers are the rudder; the right stick looks round.
//!
//! Two ways in. gilrs, for whatever the system reports as a gamepad (Xbox controllers and the
//! like). And PlayStation controllers (DualSense, DualShock 4) read directly, from their own HID
//! reports: Windows does not always hand those to its gamepad interface (a DualSense on USB was
//! listed by the system and by Steam, and not by Windows Gaming Input), and their reports are
//! simple and known.

use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};
use std::sync::{Arc, Mutex};

/// What the controller asks for this frame. Sticks: x right, y forward / up, -1..1, after the
/// dead zone and the response curve. Triggers 0..1.
#[derive(Clone, Copy, Default)]
pub struct Pad {
    pub left: (f64, f64),
    pub right: (f64, f64),
    pub lt: f64,
    pub rt: f64,
    /// the bumpers, right minus left
    pub rudder: f64,
    /// A / Cross held: speed brakes
    pub brake: bool,
    /// View / Select (or Y / Triangle) pressed since the last frame: the next camera
    pub view: bool,
    /// Menu / Start pressed since the last frame: back to the start
    pub restart: bool,
    /// the D-pad, up minus down: the chase camera nearer or further
    pub zoom: f64,
    /// B / Circle or X / Square held: fire (--battle)
    pub fire: bool,
}

/// A stick's raw position to what it asks for: nothing inside the dead zone (a stick at rest is
/// never exactly centred), then a curve that is gentle near the centre and reaches all of it at
/// the rim, so a small movement is a small correction.
pub fn shape(x: f32, y: f32) -> (f64, f64) {
    const DEAD: f64 = 0.14;
    let (x, y) = (x as f64, y as f64);
    let m = (x * x + y * y).sqrt();
    if m <= DEAD {
        return (0.0, 0.0);
    }
    let t = ((m - DEAD) / (1.0 - DEAD)).min(1.0);
    let k = t.powf(1.7) / m;
    (x * k, y * k)
}

/// Sony's USB vendor number, and its controllers: DualSense, DualSense Edge, DualShock 4 (two
/// models) and its wireless adaptor.
const SONY: u16 = 0x054C;
const SONY_PADS: [u16; 5] = [0x0CE6, 0x0DF2, 0x05C4, 0x09CC, 0x0BA0];

/// A PlayStation controller's sticks, triggers and buttons, as its report gives them.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct SonyReport {
    /// left x, left y, right x, right y: 0..255, 128 at rest, y down
    sticks: [u8; 4],
    l2: u8,
    r2: u8,
    /// low four bits the D-pad (0 up, clockwise, 8 at rest), then Square, Cross, Circle, Triangle
    b0: u8,
    /// L1, R1, L2, R2, Create / Share, Options, L3, R3
    b1: u8,
}

/// One input report to what it says. The DualSense on USB (report 1, 64 bytes) and on Bluetooth
/// (report 0x31, one byte more in front) send sticks, triggers, a counter, buttons; the
/// DualShock 4 (report 1 on USB, 0x11 on Bluetooth), and a DualSense on Bluetooth before it is
/// asked for its full report, send sticks, buttons, triggers.
fn sony_report(dualsense: bool, r: &[u8]) -> Option<SonyReport> {
    let (at, sense) = match (r.first()?, dualsense) {
        (0x31, true) => (2, true),
        (0x01, true) if r.len() >= 60 => (1, true),
        (0x01, _) => (1, false),
        (0x11, false) => (3, false),
        _ => return None,
    };
    let r = r.get(at..at + 10)?;
    let sticks = [r[0], r[1], r[2], r[3]];
    Some(if sense { SonyReport { sticks, l2: r[4], r2: r[5], b0: r[7], b1: r[8] } } else { SonyReport { sticks, l2: r[7], r2: r[8], b0: r[4], b1: r[5] } })
}

/// The newest report of the PlayStation controller, and the presses not yet acted on.
#[derive(Default)]
struct SonyState {
    connected: bool,
    now: SonyReport,
    view: bool,
    restart: bool,
}

/// Find a PlayStation controller, read it until it goes, look again: on a thread of its own (a
/// read waits for the next report; a controller sends 250 to 1000 a second).
fn sony_thread(state: Arc<Mutex<SonyState>>) {
    let Ok(mut api) = hidapi::HidApi::new() else { return };
    loop {
        let _ = api.refresh_devices();
        let found = api.device_list().find(|d| d.vendor_id() == SONY && SONY_PADS.contains(&d.product_id())).cloned();
        let Some(dev) = found.as_ref().and_then(|info| info.open_device(&api).ok()) else {
            std::thread::sleep(std::time::Duration::from_millis(1000));
            continue;
        };
        let info = found.unwrap();
        let dualsense = matches!(info.product_id(), 0x0CE6 | 0x0DF2);
        if dualsense {
            // (on Bluetooth a DualSense sends its full report once this has been asked for)
            let mut cal = [0u8; 64];
            cal[0] = 0x05;
            let _ = dev.get_feature_report(&mut cal);
        }
        println!("Controller: {}", info.product_string().unwrap_or(if dualsense { "DualSense" } else { "DualShock 4" }));
        let mut buf = [0u8; 128];
        let mut last = SonyReport::default();
        let mut quiet = 0;
        loop {
            match dev.read_timeout(&mut buf, 500) {
                Ok(0) => {
                    // (nothing for a while: it has been switched off, or is gone)
                    quiet += 1;
                    if quiet > 6 {
                        break;
                    }
                }
                Ok(n) => {
                    quiet = 0;
                    let Some(r) = sony_report(dualsense, &buf[..n]) else { continue };
                    let mut s = state.lock().unwrap();
                    s.connected = true;
                    // Create / Share or Triangle: the next view. Options: back to the start.
                    let pressed = |r: &SonyReport| (r.b1 & 0x10 != 0 || r.b0 & 0x80 != 0, r.b1 & 0x20 != 0);
                    let (was, is) = (pressed(&last), pressed(&r));
                    s.view |= is.0 && !was.0;
                    s.restart |= is.1 && !was.1;
                    s.now = r;
                    last = r;
                }
                Err(_) => break,
            }
        }
        state.lock().unwrap().connected = false;
        println!("Controller disconnected");
    }
}

impl SonyState {
    /// What it asks for this frame (and the presses are taken).
    fn take(&mut self) -> Pad {
        let r = self.now;
        let axis = |v: u8| ((v as f32 - 127.5) / 127.5).clamp(-1.0, 1.0);
        let trigger = |v: u8| ((v as f64 / 255.0 - 0.06) / 0.94).clamp(0.0, 1.0);
        let bit = |b: u8, m: u8| (b & m != 0) as u8 as f64;
        let dpad = r.b0 & 15;
        let pad = Pad {
            left: shape(axis(r.sticks[0]), -axis(r.sticks[1])),
            right: shape(axis(r.sticks[2]), -axis(r.sticks[3])),
            lt: trigger(r.l2),
            rt: trigger(r.r2),
            rudder: bit(r.b1, 2) - bit(r.b1, 1),
            brake: r.b0 & 0x20 != 0,
            view: self.view,
            restart: self.restart,
            zoom: if matches!(dpad, 0 | 1 | 7) { 1.0 } else if matches!(dpad, 3..=5) { -1.0 } else { 0.0 },
            fire: r.b0 & 0x50 != 0,
        };
        (self.view, self.restart) = (false, false);
        pad
    }
}

pub struct Pads {
    gilrs: Option<Gilrs>,
    /// the controller last touched
    active: Option<GamepadId>,
    sony: Arc<Mutex<SonyState>>,
}

impl Pads {
    pub fn new() -> Self {
        let sony = Arc::new(Mutex::new(SonyState::default()));
        let state = sony.clone();
        let _ = std::thread::Builder::new().name("controller".into()).spawn(move || sony_thread(state));
        let gilrs = Gilrs::new().map_err(|e| println!("Controllers: PlayStation only ({e})")).ok();
        Pads { gilrs, active: None, sony }
    }

    pub fn poll(&mut self) -> Pad {
        {
            let mut s = self.sony.lock().unwrap();
            if s.connected {
                return s.take();
            }
        }
        let mut pad = Pad::default();
        let Some(gilrs) = self.gilrs.as_mut() else { return pad };
        while let Some(ev) = gilrs.next_event() {
            match ev.event {
                EventType::Connected => {
                    println!("Controller: {}", gilrs.gamepad(ev.id).name());
                    self.active = Some(ev.id);
                }
                EventType::Disconnected => {
                    println!("Controller disconnected");
                    if self.active == Some(ev.id) {
                        self.active = gilrs.gamepads().map(|(id, _)| id).find(|&id| id != ev.id);
                    }
                }
                EventType::ButtonPressed(b, _) => {
                    self.active = Some(ev.id);
                    match b {
                        Button::Select | Button::North => pad.view = true,
                        Button::Start => pad.restart = true,
                        _ => {}
                    }
                }
                EventType::AxisChanged(..) | EventType::ButtonChanged(..) => self.active = Some(ev.id),
                _ => {}
            }
        }
        if self.active.is_none() {
            self.active = gilrs.gamepads().next().map(|(id, _)| id);
        }
        let Some(g) = self.active.and_then(|id| gilrs.connected_gamepad(id)) else { return pad };
        pad.left = shape(g.value(Axis::LeftStickX), g.value(Axis::LeftStickY));
        pad.right = shape(g.value(Axis::RightStickX), g.value(Axis::RightStickY));
        // (a trigger is a button with a value on most controllers, an axis on some)
        let trigger = |b: Button, a: Axis| {
            let v = g.button_data(b).map(|d| d.value()).unwrap_or(0.0).max(g.value(a)).max(if g.is_pressed(b) { 1.0 } else { 0.0 });
            ((v as f64 - 0.06) / 0.94).clamp(0.0, 1.0)
        };
        pad.lt = trigger(Button::LeftTrigger2, Axis::LeftZ);
        pad.rt = trigger(Button::RightTrigger2, Axis::RightZ);
        let held = |b: Button| g.is_pressed(b) as u8 as f64;
        pad.rudder = held(Button::RightTrigger) - held(Button::LeftTrigger);
        pad.brake = g.is_pressed(Button::South);
        pad.zoom = held(Button::DPadUp) - held(Button::DPadDown);
        pad.fire = g.is_pressed(Button::East) || g.is_pressed(Button::West);
        pad
    }
}

#[cfg(test)]
mod tests {
    use super::shape;

    /// What the system reports, and what a controller sends, for ten seconds:
    /// `cargo test --release controller_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn controller_probe() {
        let mut g = gilrs::Gilrs::new().unwrap();
        let t = std::time::Instant::now();
        let mut listed = false;
        let mut events = 0;
        while t.elapsed().as_secs_f32() < 10.0 {
            while let Some(ev) = g.next_event() {
                events += 1;
                if events < 40 {
                    println!("{:.2}s {:?} {:?}", t.elapsed().as_secs_f32(), ev.id, ev.event);
                }
            }
            if !listed && t.elapsed().as_secs_f32() > 1.0 {
                listed = true;
                println!("gamepads: {}", g.gamepads().count());
                for (id, pad) in g.gamepads() {
                    println!("  {id:?} {:?} os name {:?} mapping {:?} uuid {:02x?}", pad.name(), pad.os_name(), pad.mapping_source(), pad.uuid());
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        println!("{events} events");
    }

    /// The PlayStation reader for ten seconds, a line whenever what it asks changes:
    /// `cargo test --release sony_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sony_probe() {
        let mut pads = super::Pads::new();
        let t = std::time::Instant::now();
        let mut last = String::new();
        let mut lines = 0;
        while t.elapsed().as_secs_f32() < 10.0 {
            let connected = pads.sony.lock().unwrap().connected;
            let p = pads.poll();
            let line = format!("connected {connected} left ({:+.2} {:+.2}) right ({:+.2} {:+.2}) lt {:.2} rt {:.2} rudder {:+.0} brake {} view {} restart {} zoom {:+.0}", p.left.0, p.left.1, p.right.0, p.right.1, p.lt, p.rt, p.rudder, p.brake, p.view, p.restart, p.zoom);
            if line != last && lines < 60 {
                println!("{:.2}s {line}", t.elapsed().as_secs_f32());
                lines += 1;
                last = line;
            }
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
    }

    #[test]
    fn playstation_reports() {
        use super::{sony_report, SonyReport, SonyState};
        // a DualSense on USB: report 1, then sticks, triggers, a counter, buttons
        let mut usb = [0u8; 64];
        usb[..11].copy_from_slice(&[0x01, 10, 20, 30, 40, 50, 60, 99, 0x28, 0x21, 0x00]);
        let want = SonyReport { sticks: [10, 20, 30, 40], l2: 50, r2: 60, b0: 0x28, b1: 0x21 };
        assert_eq!(sony_report(true, &usb), Some(want));
        // the same on Bluetooth: report 0x31 and one byte more in front
        let mut bt = [0u8; 78];
        bt[..12].copy_from_slice(&[0x31, 0x01, 10, 20, 30, 40, 50, 60, 99, 0x28, 0x21, 0x00]);
        assert_eq!(sony_report(true, &bt), Some(want));
        // a DualShock 4 on USB: sticks, buttons, triggers
        let mut ds4 = [0u8; 64];
        ds4[..10].copy_from_slice(&[0x01, 10, 20, 30, 40, 0x28, 0x21, 0x00, 50, 60]);
        assert_eq!(sony_report(false, &ds4), Some(want));
        assert_eq!(sony_report(true, &[0x09, 1, 2, 3]), None);

        // at rest: nothing asked. Stick forward: y up. R2 down: the throttle. Cross: brakes.
        let rest = SonyReport { sticks: [128, 127, 128, 128], b0: 8, ..Default::default() };
        let mut s = SonyState { connected: true, now: rest, ..Default::default() };
        let p = s.take();
        assert!(p.left == (0.0, 0.0) && p.right == (0.0, 0.0) && p.lt == 0.0 && p.rt == 0.0 && !p.brake && p.zoom == 0.0 && p.rudder == 0.0);
        s.now = SonyReport { sticks: [128, 0, 255, 128], r2: 255, b0: 0x20, b1: 0x02, ..rest };
        s.view = true;
        let p = s.take();
        assert!(p.left.1 > 0.99 && p.left.0.abs() < 0.02 && p.right.0 > 0.99, "{:?} {:?}", p.left, p.right);
        assert!(p.rt == 1.0 && p.brake && p.rudder == 1.0 && p.zoom == 1.0 && p.view && !p.fire);
        // (a press is acted on once)
        assert!(!s.take().view);
    }

    #[test]
    fn stick_curve() {
        // at rest (never exactly centred): nothing
        assert_eq!(shape(0.08, -0.05), (0.0, 0.0));
        // at the rim: all of it, along the way it is pushed
        let (x, y) = shape(1.0, 0.0);
        assert!((x - 1.0).abs() < 1e-9 && y == 0.0);
        let (x, y) = shape(-0.6, 0.8);
        assert!((x + 0.6).abs() < 1e-6 && (y - 0.8).abs() < 1e-6);
        // half way: well under half (gentle near the centre), and it only ever grows
        let (h, _) = shape(0.5, 0.0);
        assert!(h > 0.15 && h < 0.35, "half stick asks {h}");
        let mut last = 0.0;
        for i in 0..=100 {
            let (v, _) = shape(i as f32 / 100.0, 0.0);
            assert!(v >= last);
            last = v;
        }
    }
}
