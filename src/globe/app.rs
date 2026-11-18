//! `--globe`: the 1:1 Earth. A float64 camera that flies from orbit to the ground over the
//! cube-sphere terrain (ETOPO 2022), with the sky from a physically based atmosphere.
//!
//! Controls: right mouse (hold) or left click (capture, Esc releases) to look; WASD move;
//! Space / E up, Q / C down; Shift x5, Ctrl x0.2; wheel = speed. Speed follows the height above
//! the ground. Arrow keys move the sun. F12 screenshot. 1 / 2 / 0 debug views. -/= exposure.

use super::dem::Terrain;
use super::render::{GlobeGlobals, GlobeRenderer};
use super::tiles::{self, dir_from_lat_lon, lat_lon, R_EARTH};
use anyhow::Result;
use glam::DVec3;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

pub struct GlobeArgs {
    width: u32,
    height: u32,
    /// lat, lon (deg), altitude above the ground (m), heading (deg from north), pitch (deg)
    at: [f64; 5],
    views: Vec<[f64; 5]>,
    shot: Option<String>,
    /// subsolar point lat, lon (deg); None = afternoon sun for each view
    sun: Option<[f64; 2]>,
    exposure: f32,
    vsync: bool,
}

pub fn parse(args: &[String]) -> GlobeArgs {
    let mut a = GlobeArgs {
        width: 1920,
        height: 1080,
        at: [46.55, 8.0, 2_500_000.0, 0.0, -90.0],
        views: Vec::new(),
        shot: None,
        sun: None,
        exposure: 0.4,
        vsync: false,
    };
    let floats = |s: &str| -> Vec<f64> { s.split(',').filter_map(|v| v.trim().parse().ok()).collect() };
    let mut i = 0;
    while i < args.len() {
        let next = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--size" => {
                let v: Vec<u32> = next.split('x').filter_map(|v| v.parse().ok()).collect();
                if v.len() == 2 {
                    a.width = v[0];
                    a.height = v[1];
                }
            }
            "--at" => {
                let v = floats(&next);
                if v.len() == 5 {
                    a.at = [v[0], v[1], v[2], v[3], v[4]];
                }
            }
            "--views" => a.views = next.split(';').map(floats).filter(|v| v.len() == 5).map(|v| [v[0], v[1], v[2], v[3], v[4]]).collect(),
            "--shot" => a.shot = Some(next.clone()),
            "--sun" => {
                let v = floats(&next);
                if v.len() == 2 {
                    a.sun = Some([v[0], v[1]]);
                }
            }
            "--exposure" => a.exposure = next.parse().unwrap_or(0.4),
            "--vsync" => a.vsync = true,
            _ => {}
        }
        i += 1;
    }
    a
}

struct Cam {
    pos: DVec3,
    heading: f64,
    pitch: f64,
}

impl Cam {
    fn frame(&self) -> (DVec3, DVec3, DVec3) {
        let up = self.pos.normalize();
        let mut east = DVec3::Z.cross(up);
        if east.length_squared() < 1e-12 {
            east = DVec3::X;
        }
        let east = east.normalize();
        let north = up.cross(east);
        (up, east, north)
    }
    /// forward, right, camera up
    fn basis(&self) -> (DVec3, DVec3, DVec3) {
        let (up, east, north) = self.frame();
        let h = north * self.heading.cos() + east * self.heading.sin();
        let fwd = (h * self.pitch.cos() + up * self.pitch.sin()).normalize();
        let right = fwd.cross(up).normalize();
        let cup = right.cross(fwd);
        (fwd, right, cup)
    }
}

struct State {
    window: Arc<Window>,
    r: GlobeRenderer,
    etopo: Arc<Terrain>,
    cam: Cam,
    keys: HashSet<KeyCode>,
    looking: bool,
    captured: bool,
    last: Instant,
    speed_mul: f64,
    sun: [f64; 2],
    exposure: f32,
    debug: u32,
    fps_t: Instant,
    fps_n: u32,
    view: usize,
    view_frames: u32,
    shots: u32,
    capture_next: bool,
}

struct App {
    args: GlobeArgs,
    state: Option<State>,
}

/// Mid-afternoon summer sun for a view: subsolar point 15 N, 40 degrees west of it.
fn sun_for(v: [f64; 5]) -> [f64; 2] {
    [15.0, v[1] - 40.0]
}

fn place(etopo: &Terrain, v: [f64; 5]) -> Cam {
    let d = dir_from_lat_lon(v[0], v[1]);
    let ground = tiles::ground_radius(etopo, d);
    Cam { pos: d * (ground + v[2].max(1.7)), heading: v[3].to_radians(), pitch: v[4].to_radians().clamp(-1.5707, 1.5707) }
}

impl App {
    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let a = &self.args;
        let window = Arc::new(el.create_window(
            Window::default_attributes().with_title("SmartImposter Earth (loading...)").with_inner_size(winit::dpi::PhysicalSize::new(a.width, a.height)),
        )?);
        let etopo = Arc::new(Terrain::load(std::path::Path::new("assets/earth"))?);
        let size = window.inner_size();
        let t = Instant::now();
        let r = GlobeRenderer::new(window.display_handle()?.as_raw(), window.window_handle()?.as_raw(), size.width, size.height, etopo.clone(), a.vsync)?;
        println!("Globe: {} base tiles in {:.1}s", r.tile_count(), t.elapsed().as_secs_f32());
        let first = a.views.first().copied().unwrap_or(a.at);
        let cam = place(&etopo, first);
        self.state = Some(State {
            window,
            r,
            etopo,
            cam,
            keys: HashSet::new(),
            looking: false,
            captured: false,
            last: Instant::now(),
            speed_mul: 1.0,
            sun: a.sun.unwrap_or_else(|| sun_for(first)),
            exposure: a.exposure,
            debug: 0,
            fps_t: Instant::now(),
            fps_n: 0,
            view: 0,
            view_frames: 0,
            shots: 0,
            capture_next: false,
        });
        Ok(())
    }

    fn frame(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let views = self.args.views.clone();
        let shot = self.args.shot.clone();
        let Some(s) = self.state.as_mut() else { return Ok(()) };
        if s.r.need_resize {
            let sz = s.window.inner_size();
            if sz.width == 0 || sz.height == 0 {
                return Ok(());
            }
            s.r.resize(sz.width, sz.height)?;
        }
        let now = Instant::now();
        let dt = (now - s.last).as_secs_f64().min(0.1);
        s.last = now;

        // movement: speed scales with the height above the ground
        let ground = tiles::ground_radius(&s.etopo, s.cam.pos);
        let alt = s.cam.pos.length() - ground;
        let k = |c| s.keys.contains(&c);
        let (fwd, right, _) = s.cam.basis();
        let (up, _, _) = s.cam.frame();
        let mut mv = DVec3::ZERO;
        if k(KeyCode::KeyW) { mv += fwd; }
        if k(KeyCode::KeyS) { mv -= fwd; }
        if k(KeyCode::KeyD) { mv += right; }
        if k(KeyCode::KeyA) { mv -= right; }
        if k(KeyCode::Space) || k(KeyCode::KeyE) { mv += up; }
        if k(KeyCode::KeyQ) || k(KeyCode::KeyC) { mv -= up; }
        let mut speed = alt.max(3.0) * 0.8 * s.speed_mul;
        if k(KeyCode::ShiftLeft) { speed *= 5.0; }
        if k(KeyCode::ControlLeft) { speed *= 0.2; }
        if mv.length_squared() > 0.0 {
            s.cam.pos += mv.normalize() * speed * dt;
        }
        let ground = tiles::ground_radius(&s.etopo, s.cam.pos);
        if s.cam.pos.length() < ground + 1.7 {
            s.cam.pos = s.cam.pos.normalize() * (ground + 1.7);
        }
        let sun_step = 20.0 * dt;
        if k(KeyCode::ArrowLeft) { s.sun[1] -= sun_step; }
        if k(KeyCode::ArrowRight) { s.sun[1] += sun_step; }
        if k(KeyCode::ArrowUp) { s.sun[0] = (s.sun[0] + sun_step).min(89.0); }
        if k(KeyCode::ArrowDown) { s.sun[0] = (s.sun[0] - sun_step).max(-89.0); }

        let (fwd, right, cup) = s.cam.basis();
        let f = |v: DVec3, w: f32| [v.x as f32, v.y as f32, v.z as f32, w];
        let sun = dir_from_lat_lon(s.sun[0], s.sun[1]);
        let g = GlobeGlobals {
            cam_right: f(right, 0.0),
            cam_up: f(cup, 0.0),
            cam_fwd: f(fwd, (30f64.to_radians()).tan() as f32),
            sun: f(sun, 20.0),
            planet: f(-s.cam.pos, R_EARTH as f32),
            params: [s.exposure, 0.0, s.debug as f32, 0.0],
            addr: [0; 4],
        };

        // batch views: wait until the tiles for each view are in, then capture
        let mut capture = s.capture_next;
        if !views.is_empty() {
            s.view_frames += 1;
            if s.r.waiting == 0 && s.view_frames > 8 || s.view_frames > 900 {
                capture = true;
            }
        }
        let img = s.r.render(g, s.cam.pos, capture)?;
        if let Some(img) = img {
            let (w, h) = s.r.extent();
            let path = match (&shot, views.is_empty()) {
                (Some(p), false) => format!("{}_{}.png", p.trim_end_matches(".png"), s.view),
                (Some(p), true) => p.clone(),
                (None, _) => format!("shots/globe_{:03}.png", s.shots),
            };
            crate::save_png(&path, w, h, &img)?;
            println!("saved {path} ({} tiles drawn, {} cached)", s.r.drawn, s.r.tile_count());
            s.shots += 1;
            s.capture_next = false;
            if !views.is_empty() {
                s.view += 1;
                s.view_frames = 0;
                if s.view >= views.len() {
                    el.exit();
                    return Ok(());
                }
                s.cam = place(&s.etopo, views[s.view]);
                if self.args.sun.is_none() {
                    s.sun = sun_for(views[s.view]);
                }
            }
        }

        s.fps_n += 1;
        let el_t = s.fps_t.elapsed().as_secs_f32();
        if el_t > 0.5 {
            let (lat, lon) = lat_lon(s.cam.pos);
            let alt = s.cam.pos.length() - tiles::ground_radius(&s.etopo, s.cam.pos);
            let alt_s = if alt > 10_000.0 { format!("{:.0} km", alt / 1000.0) } else { format!("{:.0} m", alt) };
            s.window.set_title(&format!(
                "SmartImposter Earth | {:.0} fps | {:.4}, {:.4} | {} above ground | {} tiles drawn, {} cached, {} waiting",
                s.fps_n as f32 / el_t,
                lat,
                lon,
                alt_s,
                s.r.drawn,
                s.r.tile_count(),
                s.r.waiting
            ));
            s.fps_t = Instant::now();
            s.fps_n = 0;
        }
        Ok(())
    }

    fn key(&mut self, code: KeyCode, pressed: bool, el: &ActiveEventLoop) {
        let Some(s) = self.state.as_mut() else { return };
        if pressed {
            s.keys.insert(code);
        } else {
            s.keys.remove(&code);
            return;
        }
        match code {
            KeyCode::Escape => {
                if s.captured {
                    s.captured = false;
                    let _ = s.window.set_cursor_grab(CursorGrabMode::None);
                    s.window.set_cursor_visible(true);
                } else {
                    el.exit();
                }
            }
            KeyCode::Digit0 => s.debug = 0,
            KeyCode::Digit1 => s.debug = 1,
            KeyCode::Digit2 => s.debug = 2,
            KeyCode::Minus => s.exposure /= 1.15,
            KeyCode::Equal => s.exposure *= 1.15,
            KeyCode::F12 => s.capture_next = true,
            KeyCode::KeyP => {
                let (lat, lon) = lat_lon(s.cam.pos);
                let alt = s.cam.pos.length() - tiles::ground_radius(&s.etopo, s.cam.pos);
                println!("--at {:.5},{:.5},{:.1},{:.1},{:.1}", lat, lon, alt, s.cam.heading.to_degrees(), s.cam.pitch.to_degrees());
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_none() {
            if let Err(e) = self.init(el) {
                eprintln!("globe init failed: {e:?}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(_) => {
                if let Some(s) = self.state.as_mut() {
                    s.r.need_resize = true;
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.frame(el) {
                    eprintln!("frame failed: {e:?}");
                    el.exit();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.key(code, event.state == ElementState::Pressed, el);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(s) = self.state.as_mut() {
                    match button {
                        MouseButton::Left if state == ElementState::Pressed && !s.captured => {
                            s.captured = true;
                            if s.window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                                let _ = s.window.set_cursor_grab(CursorGrabMode::Confined);
                            }
                            s.window.set_cursor_visible(false);
                        }
                        MouseButton::Right if !s.captured => {
                            s.looking = state == ElementState::Pressed;
                            let _ = s.window.set_cursor_grab(if s.looking { CursorGrabMode::Confined } else { CursorGrabMode::None });
                            s.window.set_cursor_visible(!s.looking);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(s) = self.state.as_mut() {
                    let steps = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y as f64,
                        MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                    };
                    s.speed_mul = (s.speed_mul * 1.25f64.powf(steps)).clamp(0.01, 100.0);
                }
            }
            WindowEvent::Focused(false) => {
                if let Some(s) = self.state.as_mut() {
                    s.keys.clear();
                    s.captured = false;
                    s.looking = false;
                    let _ = s.window.set_cursor_grab(CursorGrabMode::None);
                    s.window.set_cursor_visible(true);
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _el: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if let Some(s) = self.state.as_mut() {
                if s.looking || s.captured {
                    s.cam.heading += delta.0 * 0.0022;
                    s.cam.pitch = (s.cam.pitch - delta.1 * 0.0022).clamp(-1.5707, 1.5707);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(s) = self.state.as_ref() {
            s.window.request_redraw();
        }
    }
}

pub fn run(args: GlobeArgs) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { args, state: None };
    event_loop.run_app(&mut app)?;
    if let Some(s) = app.state.as_ref() {
        unsafe {
            let _ = s.r.gpu.device.device_wait_idle();
        }
    }
    std::process::exit(0);
}
