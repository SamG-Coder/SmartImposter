//! SmartImposter: a hardware ray traced forest renderer.
//!
//! Trees closer than the LOD distance are traced as real geometry (every twig instance of the
//! Megaplant in the TLAS). Trees beyond it are single AABBs whose intersection shader ray-marches
//! a baked parallax octahedral impostor.

mod accel;
mod asset;
mod bc7;
mod catalog;
mod dlss;
mod erosion;
mod flypath;
mod globe;
mod textures;
mod gpu;
mod renderer;
mod world;

use anyhow::Result;
use glam::{DMat3, DVec3, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use renderer::{Globals, ImpostorSettings, Renderer};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

struct Args {
    asset: String,
    width: u32,
    height: u32,
    shot: Option<String>,
    /// A real place (`--site lat,lon`): terrain, water and forest from the Earth data.
    site: Option<[f64; 2]>,
    /// The 1:1 globe (`--globe`): terrain and trees stream in from the Earth data around the camera.
    globe: bool,
    /// Globe start: lat, lon (deg), height above the ground (m), heading (deg from north), pitch (deg).
    at: [f64; 5],
    /// Globe clock: start time (UTC "YYYY-MM-DDTHH:MM"; default 10:30 local solar time on the June
    /// solstice at the start) and speed (simulated seconds per second; default real time, or
    /// stopped when capturing).
    time: Option<String>,
    time_rate: Option<f64>,
    /// Several cameras in one run (`--views`): x, height above ground, z, yaw, pitch.
    views: Vec<[f32; 7]>, // + optional mesh LOD distance and debug mode (-1 = unchanged)
    frames: u32,
    cam: Option<[f32; 5]>,
    lod: f32,
    debug: u32,
    ao_rays: u32,
    sun: [f32; 2],
    imp: ImpostorSettings,
    species: Option<String>,
    lod2: f32,
    lod_band: f32,
    lod2_band: f32,
    spacing: f32,
    radius: f32,
    /// Aerial perspective strength (1 = clear day, 0 = off).
    haze: f32,
    dump_atlas: Option<String>,
    validation: bool,
    vsync: bool,
    gallery: bool,
    bench_move: f32,
    fly: bool,
    fly_speed: f32,
    fly_start: f32,
    ss: f32,
    fullscreen: bool,
    no_taa: bool,
    /// globe: volumetric clouds (`--clouds on|off`, key U) and their cover (`--cloud-cover 0..1`)
    clouds: bool,
    cloud_cover: f32,
    /// globe: thin distant trees smoothly with distance (no density step at tree tile edges,
    /// ~0.3-0.5 ms) instead of one density per tile (`--tree-thinning smooth|stepped`, key N)
    smooth_thin: bool,
    /// DLSS Ray Reconstruction mode (index into dlss::MODES), None = TAA + supersampling.
    dlss: Option<usize>,
    sharpen: f32,
}

fn parse_args() -> Args {
    let mut a = Args {
        asset: "assets/black_alder.mtree".into(),
        width: 1920,
        height: 1080,
        shot: None,
        views: Vec::new(),
        site: None,
        globe: false,
        at: [46.05, 9.25, 300.0, 200.0, -8.0],
        frames: 1,
        cam: None,
        lod: 50.0,
        debug: 0,
        ao_rays: 1,
        sun: [35.0, 38.0],
        time: None,
        time_rate: None,
        imp: ImpostorSettings { grid: 12, texel: 0.055, max_res: 448, ss: 3, ao_rays: 6, ao_range: 1.5, vis_rays: 12 },
        species: None,
        lod2: 100.0,
        lod_band: 12.0,
        lod2_band: 24.0,
        spacing: 7.0,
        radius: 8000.0,
        haze: 1.0,
        dump_atlas: None,
        validation: std::env::var("SI_VALIDATION").is_ok(),
        vsync: false,
        gallery: false,
        bench_move: 0.0,
        fly: false,
        fly_speed: 13.0,
        fly_start: 0.0,
        ss: 1.5,
        fullscreen: false,
        no_taa: false,
        smooth_thin: true,
        clouds: true,
        cloud_cover: 0.45,
        dlss: Some(1),
        sharpen: 0.5,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let floats = |s: &str| -> Vec<f32> { s.split(',').filter_map(|v| v.trim().parse().ok()).collect() };
    while i < args.len() {
        let next = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--asset" => a.asset = next,
            "--size" => {
                let v: Vec<u32> = next.split('x').filter_map(|v| v.parse().ok()).collect();
                if v.len() == 2 {
                    a.width = v[0];
                    a.height = v[1];
                }
            }
            "--shot" => a.shot = Some(next),
            "--views" => {
                a.views = next
                    .split(';')
                    .map(floats)
                    .filter(|v| v.len() >= 5)
                    .map(|v| [v[0], v[1], v[2], v[3], v[4], *v.get(5).unwrap_or(&-1.0), *v.get(6).unwrap_or(&-1.0)])
                    .collect();
            }
            "--frames" => a.frames = next.parse().unwrap_or(1),
            "--globe" => {
                a.globe = true;
                i -= 1; // no value
            }
            "--at" => {
                let v: Vec<f64> = next.split(',').filter_map(|v| v.trim().parse().ok()).collect();
                if v.len() == 5 {
                    a.at = [v[0], v[1], v[2], v[3], v[4]];
                }
            }
            "--site" => {
                let v: Vec<f64> = next.split(',').filter_map(|v| v.trim().parse().ok()).collect();
                if v.len() == 2 {
                    a.site = Some([v[0], v[1]]);
                }
            }
            "--cam" => {
                let v = floats(&next);
                if v.len() == 5 {
                    a.cam = Some([v[0], v[1], v[2], v[3], v[4]]);
                }
            }
            "--lod" => a.lod = next.parse().unwrap_or(50.0),
            "--debug" => a.debug = next.parse().unwrap_or(0),
            "--ao" => a.ao_rays = next.parse().unwrap_or(1),
            "--time" => a.time = Some(next),
            "--time-rate" => a.time_rate = next.parse().ok(),
            "--sun" => {
                let v = floats(&next);
                if v.len() == 2 {
                    a.sun = [v[0], v[1]];
                }
            }
            "--imp-res" => a.imp.max_res = next.parse().unwrap_or(448),
            "--imp-texel" => a.imp.texel = next.parse().unwrap_or(0.05),
            "--imp-grid" => a.imp.grid = next.parse().unwrap_or(12),
            "--species" => a.species = Some(next),
            "--lod2" => a.lod2 = next.parse().unwrap_or(100.0),
            "--lod-band" => a.lod_band = next.parse().unwrap_or(12.0),
            "--lod2-band" => a.lod2_band = next.parse().unwrap_or(24.0),
            "--spacing" => a.spacing = next.parse().unwrap_or(7.0),
            "--radius" => a.radius = next.parse().unwrap_or(8000.0),
            "--haze" => a.haze = next.parse().unwrap_or(1.0),
            "--dump-atlas" => a.dump_atlas = Some(next),
            "--bench-move" => a.bench_move = next.parse().unwrap_or(0.3),
            "--fly-speed" => a.fly_speed = next.parse().unwrap_or(13.0),
            "--fly-start" => a.fly_start = next.parse().unwrap_or(0.0),
            "--ss" => a.ss = next.parse().unwrap_or(1.5),
            "--fly" => {
                a.fly = true;
                i -= 1;
            }
            "--sharpen" => a.sharpen = next.parse().unwrap_or(0.5),
            "--dlss" => {
                a.dlss = match next.to_ascii_lowercase().as_str() {
                    "off" | "none" | "0" => None,
                    "dlaa" => Some(0),
                    "balanced" => Some(2),
                    "performance" | "perf" => Some(3),
                    "ultra" | "ultra-performance" => Some(4),
                    _ => Some(1),
                };
            }
            "--no-taa" => {
                a.no_taa = true;
                i -= 1;
            }
            "--tree-thinning" => a.smooth_thin = next != "stepped",
            "--clouds" => a.clouds = next != "off",
            "--cloud-cover" => a.cloud_cover = next.parse().unwrap_or(0.45),
            "--fullscreen" => {
                a.fullscreen = true;
                i -= 1;
            }
            "--validation" => {
                a.validation = true;
                i -= 1;
            }
            "--vsync" => {
                a.vsync = true;
                i -= 1;
            }
            "--gallery" => {
                a.gallery = true;
                i -= 1;
            }
            other => eprintln!("unknown argument {other}"),
        }
        i += 2;
    }
    a
}

struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    fov_y: f32,
}

impl Camera {
    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let fwd = Vec3::new(self.yaw.sin() * self.pitch.cos(), self.pitch.sin(), -self.yaw.cos() * self.pitch.cos());
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        (fwd, right, up)
    }
}

struct State {
    window: Arc<Window>,
    r: Renderer,
    cam: Camera,
    keys: HashSet<KeyCode>,
    looking: bool,
    last: Instant,
    frame: u32,
    accum: u32,
    fingerprint: Vec<u32>,
    lod: f32,
    lod2: f32,
    debug: u32,
    ao_rays: u32,
    sun_az: f32,
    sun_el: f32,
    exposure: f32,
    smooth_thin: bool,
    clouds: bool,
    fps_time: Instant,
    fps_frames: u32,
    shot_counter: u32,
    capture_next: bool,
    rendered: u32,
    frame_times: Vec<f32>,
    prev_cam: Option<[[f32; 4]; 4]>, // last frame's cam_pos/right/up/fwd for TAA reprojection
    path: flypath::FlyPath,
    fly_on: bool,
    fly_dist: f32,
    // manual fly-cam
    captured: bool,
    yaw_target: f32,
    pitch_target: f32,
    vel: Vec3,
    base_speed: f32,
    cinematic: bool,
    /// Last DLSS mode, restored by L.
    dlss_last: usize,
    haze: f32,
    /// Globe mode: the camera on the Earth (the render frame is rebuilt around it every frame).
    geo: Option<Geo>,
    /// globe: frames a capture waited for streaming
    settle: u32,
}

/// Globe camera: position in float64 Earth-centred coordinates; the renderer's world is the
/// east / up / south frame at the camera, origin at the camera (Camera.pos stays 0).
struct Geo {
    ecef: DVec3,
    /// simulated UTC time (seconds since 1970) and clock speed (simulated seconds per second)
    time: f64,
    rate: f64,
    /// eye adaptation: exposure factor (1 in daylight, up by ~10^4 under a full moon)
    adapt: f32,
    /// last frame's camera position and right / up / forward vectors, Earth-fixed (motion vectors)
    prev: Option<[DVec3; 4]>,
    /// the renderer's frame at the camera (rows east, up, south): set from north at a jump, then
    /// carried along with the camera (scene::transport), so flights over the poles stay straight
    frame: DMat3,
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn geo_place(terrain: &globe::dem::Terrain, lat: f64, lon: f64, height: f64) -> DVec3 {
    let d = globe::tiles::dir_from_lat_lon(lat, lon);
    d * (globe::tiles::ground_radius(terrain, d) + height.max(1.7))
}

struct App {
    args: Args,
    state: Option<State>,
}

pub fn save_png(path: &str, w: u32, h: u32, rgba: &[u8]) -> Result<()> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let f = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(rgba)?;
    Ok(())
}

impl App {
    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let a = &self.args;
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("SmartImposter (loading...)")
                    .with_inner_size(winit::dpi::PhysicalSize::new(a.width, a.height)),
            )?,
        );
        if let Some(prefix) = &a.dump_atlas {
            std::env::set_var("SI_DUMP_ATLAS", prefix);
        }
        let t = Instant::now();
        let (pack, cat) = catalog::load("assets/species", &a.asset, "assets/textures", a.species.as_deref())?;
        let tris: usize = pack.meshes.iter().map(|m| m.tri_count() as usize).sum();
        println!(
            "Catalog: {} species, {} variants, {} unique meshes, {:.1}M unique triangles",
            cat.species.len(),
            pack.trees.len(),
            pack.meshes.len(),
            tris as f64 / 1e6
        );
        for sp in &cat.species {
            let hs: Vec<String> = sp.variants.clone().map(|v| format!("{:.1}", cat.variants[v].height)).collect();
            println!("  {:<24} {:?} heights [{}] m", sp.name, sp.kind, hs.join(", "));
        }
        let (mut pack, mut cat) = (pack, cat);
        let globe_terrain = if a.globe { Some(std::sync::Arc::new(globe::dem::Terrain::load(std::path::Path::new("assets/earth"))?)) } else { None };
        if let Some([lat, lon]) = a.site {
            let terrain = std::sync::Arc::new(globe::dem::Terrain::load(std::path::Path::new("assets/earth"))?);
            world::set_site(terrain, lat, lon);
            println!("Site: {lat:.5}, {lon:.5}");
        }
        let world = if a.globe {
            world::World::empty()
        } else if a.gallery {
            world::World::gallery(&pack.trees, &cat)
        } else {
            world::World::generate(&pack.trees, &cat, a.radius)
        };
        println!("World: {} trees, terrain {} tris ({:.2}s)", world.trees.len(), world.terrain.tri_count(), t.elapsed().as_secs_f32());

        let size = window.inner_size();
        let r = Renderer::new(
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            size.width,
            size.height,
            pack,
            cat,
            world,
            a.imp,
            a.vsync,
            a.validation,
            a.ss,
            a.dlss,
            globe_terrain.clone(),
        )?;
        let start_y = r.world.surface_at(0.0, 0.0) + 1.8;
        let cam = match a.cam {
            Some(c) => Camera { pos: Vec3::new(c[0], c[1], c[2]), yaw: c[3].to_radians(), pitch: c[4].to_radians(), fov_y: 60f32.to_radians() },
            None => Camera { pos: Vec3::new(0.0, start_y, 0.0), yaw: 0.0, pitch: 0.05, fov_y: 60f32.to_radians() },
        };
        let (cam, geo) = match &globe_terrain {
            Some(t) => {
                let ecef = geo_place(t, a.at[0], a.at[1], a.at[2]);
                let c = Camera { pos: Vec3::ZERO, yaw: a.at[3].to_radians() as f32, pitch: a.at[4].to_radians() as f32, fov_y: 60f32.to_radians() };
                // the clock: given, or 10:30 local solar time on the June solstice here
                let time = a.time.as_deref().and_then(globe::sky::parse_utc).unwrap_or_else(|| {
                    globe::sky::parse_utc("2026-06-21").unwrap() + (10.5 - a.at[1] / 15.0).rem_euclid(24.0) * 3600.0
                });
                let capture = a.shot.is_some() || a.bench_move != 0.0;
                let rate = a.time_rate.unwrap_or(if capture { 0.0 } else { 1.0 });
                (c, Some(Geo { ecef, time, rate, adapt: 0.0, prev: None, frame: globe::scene::frame_at(ecef) }))
            }
            None => (cam, None),
        };
        let (cam_yaw, cam_pitch) = (cam.yaw, cam.pitch);
        let mut state = State {
            window,
            r,
            cam,
            keys: HashSet::new(),
            looking: false,
            last: Instant::now(),
            frame: 0,
            accum: 0,
            fingerprint: vec![],
            lod: a.lod,
            lod2: a.lod2,
            debug: a.debug,
            ao_rays: a.ao_rays,
            sun_az: a.sun[0],
            sun_el: a.sun[1],
            exposure: 2.2,
            smooth_thin: a.smooth_thin,
            clouds: a.clouds,
            fps_time: Instant::now(),
            fps_frames: 0,
            shot_counter: 0,
            capture_next: false,
            rendered: 0,
            frame_times: Vec::new(),
            prev_cam: None,
            path: flypath::FlyPath::showcase(),
            fly_on: a.fly,
            fly_dist: a.fly_start,
            captured: false,
            yaw_target: cam_yaw,
            pitch_target: cam_pitch,
            vel: Vec3::ZERO,
            base_speed: 10.0,
            cinematic: false,
            dlss_last: a.dlss.unwrap_or(1),
            haze: a.haze,
            geo,
            settle: 0,
        };
        if std::env::var("SI_FOREST_MAP").is_ok() {
            // Coarse tree-count map, 50 m cells over [-700, 700]^2, north (-Z) at the top.
            let cell = 50.0;
            let n = 28;
            let mut counts = vec![0u32; n * n];
            for t in &state.r.world.trees {
                let p = t.xform.translation;
                let i = ((p.x + 700.0) / cell) as isize;
                let j = ((p.z + 700.0) / cell) as isize;
                if (0..n as isize).contains(&i) && (0..n as isize).contains(&j) {
                    counts[j as usize * n + i as usize] += 1;
                }
            }
            println!("forest map: x from -700 (left) to 700, z from -700 (top) to 700; '#' dense '+' medium '.' sparse");
            for j in 0..n {
                let row: String = (0..n)
                    .map(|i| match counts[j * n + i] { c if c > 35 => '#', c if c > 15 => '+', c if c > 0 => '.', _ => ' ' })
                    .collect();
                println!("{:5.0} |{}|", -700.0 + j as f32 * cell, row);
            }
        }
        if a.fly {
            let hits = state.path.check(&state.r.world);
            println!("Fly path: {:.0} m loop, {:.0} s at {} m/s", state.path.length, state.path.length / a.fly_speed, a.fly_speed);
            for (d, p) in hits.iter().take(20) {
                println!("  passes through a crown at {:.0} m along the path ({:.1}, {:.1}, {:.1})", d, p.x, p.y, p.z);
            }
            if hits.is_empty() {
                println!("  path is clear of tree crowns");
            }
        }
        state.r.taa_enabled = !a.no_taa;
        state.r.sharpness = a.sharpen;
        if a.fullscreen {
            state.window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        state.window.set_title("SmartImposter");
        self.state = Some(state);
        Ok(())
    }

    fn frame(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let shot = self.args.shot.clone();
        let shot_frames = self.args.frames.max(1);
        let views = self.args.views.clone();
        let Some(s) = self.state.as_mut() else { return Ok(()) };
        // Multi-view capture: `frames` frames per view, then jump to the next camera.
        let view = (s.rendered / shot_frames) as usize;
        if !views.is_empty() && s.rendered % shot_frames == 0 && view < views.len() {
            let v = views[view];
            match (s.geo.as_mut(), &s.r.globe) {
                // globe: lat, lon, height above the ground, heading, pitch
                (Some(geo), Some(scene)) => {
                    geo.ecef = geo_place(&scene.terrain, v[0] as f64, v[1] as f64, v[2] as f64);
                    geo.frame = globe::scene::frame_at(geo.ecef);
                    geo.prev = None;
                    s.cam.pos = Vec3::ZERO;
                }
                _ => s.cam.pos = Vec3::new(v[0], s.r.world.surface_at(v[0], v[2]) + v[1], v[2]),
            }
            s.cam.yaw = v[3].to_radians();
            s.cam.pitch = v[4].to_radians();
            s.yaw_target = s.cam.yaw;
            s.pitch_target = s.cam.pitch;
            if v[5] >= 0.0 {
                s.lod = v[5];
            }
            if v[6] >= 0.0 {
                s.debug = v[6] as u32;
            }
            s.prev_cam = None;
            s.r.reset_taa();
            s.frame_times.clear();
            s.r.profile = Default::default();
        }
        if s.r.need_resize {
            let sz = s.window.inner_size();
            s.r.resize(sz.width, sz.height)?;
            if s.r.need_resize {
                return Ok(());
            }
        }
        let now = Instant::now();
        let raw_dt = (now - s.last).as_secs_f32();
        let dt = raw_dt.min(0.1);
        s.last = now;
        if s.rendered >= 5 {
            s.frame_times.push(raw_dt * 1000.0);
        }
        if s.rendered == 5 && views.is_empty() {
            s.r.profile = Default::default(); // exclude warm-up frames
        }

        // ---- manual fly-cam input (any input takes over from the automatic path)
        let k = |c: KeyCode| s.keys.contains(&c);
        let mut input = Vec3::ZERO; // x right, y up, z forward
        if k(KeyCode::KeyW) { input.z += 1.0; }
        if k(KeyCode::KeyS) { input.z -= 1.0; }
        if k(KeyCode::KeyD) { input.x += 1.0; }
        if k(KeyCode::KeyA) { input.x -= 1.0; }
        if k(KeyCode::KeyE) || k(KeyCode::Space) { input.y += 1.0; }
        if k(KeyCode::KeyQ) { input.y -= 1.0; }
        let mut speed = s.base_speed;
        if let (Some(geo), Some(scene)) = (&s.geo, &s.r.globe) {
            // globe: fly speed follows the height above the ground
            let alt = geo.ecef.length() - globe::tiles::ground_radius(&scene.terrain, geo.ecef);
            speed *= (alt as f32 / 20.0).max(1.0);
        }
        if k(KeyCode::ShiftLeft) { speed *= 4.0; }
        if k(KeyCode::ControlLeft) { speed *= 0.25; }
        if s.fly_on && input != Vec3::ZERO {
            s.fly_on = false;
            s.yaw_target = s.cam.yaw;
            s.pitch_target = s.cam.pitch;
            s.vel = Vec3::ZERO;
        }

        if s.fly_on {
            s.fly_dist += self.args.fly_speed * dt;
            let (pos, yaw, pitch) = s.path.camera(&s.r.world, s.fly_dist);
            s.cam.pos = pos;
            s.cam.yaw = yaw;
            s.cam.pitch = pitch;
            s.yaw_target = yaw;
            s.pitch_target = pitch;
        } else {
            // Smoothed look and velocity: exponential approach to the targets. Cinematic mode
            // uses much slower response for recording.
            let (look_rate, accel) = if s.cinematic { (5.0, 1.8) } else { (30.0, 9.0) };
            let a = 1.0 - (-look_rate * dt).exp();
            s.cam.yaw += (s.yaw_target - s.cam.yaw) * a;
            s.cam.pitch += (s.pitch_target - s.cam.pitch) * a;
            // Snap when settled so the image can accumulate once the camera is at rest.
            if (s.yaw_target - s.cam.yaw).abs() < 1e-4 {
                s.cam.yaw = s.yaw_target;
            }
            if (s.pitch_target - s.cam.pitch).abs() < 1e-4 {
                s.cam.pitch = s.pitch_target;
            }
            let (fwd, right, _) = s.cam.basis();
            let wish = if input != Vec3::ZERO {
                (right * input.x + Vec3::Y * input.y + fwd * input.z).normalize() * speed
            } else {
                Vec3::ZERO
            };
            s.vel += (wish - s.vel) * (1.0 - (-accel * dt).exp());
            if input == Vec3::ZERO && s.vel.length() < 0.02 {
                s.vel = Vec3::ZERO;
            }
            s.cam.pos += s.vel * dt;
            if let (Some(geo), Some(scene)) = (s.geo.as_mut(), &s.r.globe) {
                // globe: the move goes onto the Earth position; the frame follows the camera
                geo.ecef += geo.frame.transpose() * s.cam.pos.as_dvec3();
                s.cam.pos = Vec3::ZERO;
                let ground = globe::tiles::ground_radius(&scene.terrain, geo.ecef) + 0.6;
                if geo.ecef.length() < ground {
                    geo.ecef = geo.ecef.normalize() * ground;
                    s.vel.y = s.vel.y.max(0.0);
                }
                // the frame follows; the velocity is in the frame, so it carries over unchanged
                geo.frame = globe::scene::transport(geo.frame, geo.ecef);
            } else {
                let ground = s.r.world.surface_at(s.cam.pos.x, s.cam.pos.z) + 0.6;
                if s.cam.pos.y < ground {
                    s.cam.pos.y = ground;
                    s.vel.y = s.vel.y.max(0.0);
                }
            }
        }
        let (fwd, right, up) = s.cam.basis();
        if let (true, Some(geo), Some(scene)) = (self.args.bench_move != 0.0, s.geo.as_mut(), &s.r.globe) {
            // Globe fly-through benchmark: along the heading at a constant height above the ground.
            let ground = globe::tiles::ground_radius(&scene.terrain, geo.ecef);
            let above = geo.ecef.length() - ground;
            let dir = geo.frame.transpose() * Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero().as_dvec3();
            let p = geo.ecef + dir * self.args.bench_move as f64;
            geo.ecef = p.normalize() * (globe::tiles::ground_radius(&scene.terrain, p) + above);
            geo.frame = globe::scene::transport(geo.frame, geo.ecef);
        } else if self.args.bench_move != 0.0 {
            // Keep the initial height above the terrain while moving (walk / fly benchmarks).
            let above = s.cam.pos.y - s.r.world.surface_at(s.cam.pos.x, s.cam.pos.z);
            s.cam.pos += Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero() * self.args.bench_move;
            s.cam.pos.y = s.r.world.surface_at(s.cam.pos.x, s.cam.pos.z) + above;
        }

        let (w, h) = s.r.extent();
        let (rw, rh) = s.r.render_extent();
        let fp: Vec<u32> = [
            s.cam.pos.x, s.cam.pos.y, s.cam.pos.z, s.cam.yaw, s.cam.pitch, s.lod, s.lod2, s.sun_az, s.sun_el, s.exposure,
            s.debug as f32, s.ao_rays as f32, w as f32, h as f32, rw as f32, rh as f32,
        ]
        .iter()
        .map(|v| v.to_bits())
        .collect();
        if fp != s.fingerprint {
            s.fingerprint = fp;
            s.accum = 0;
        } else {
            s.accum = (s.accum + 1).min(1 << 20);
        }

        let tan_half = (s.cam.fov_y * 0.5).tan();
        let aspect = w as f32 / h as f32;
        let (az, el_) = (s.sun_az.to_radians(), s.sun_el.to_radians());
        let sun = Vec3::new(el_.cos() * az.sin(), el_.sin(), -el_.cos() * az.cos()).normalize();
        let cam_now = [
            [s.cam.pos.x, s.cam.pos.y, s.cam.pos.z, s.lod],
            [right.x, right.y, right.z, tan_half * aspect],
            [up.x, up.y, up.z, tan_half],
            // Ray spread from the supersampled resolution: finer impostor mips when supersampling.
            [fwd.x, fwd.y, fwd.z, 2.0 * tan_half / rh as f32],
        ];
        let mut prev = s.prev_cam.unwrap_or(cam_now);
        s.prev_cam = Some(cam_now);
        let mut sun = sun;
        let mut planet = [0.0f32; 4];
        let mut earth = [[0.0f32; 4]; 3];
        let (mut sun_w, mut moon, mut sun_true) = (4.6f32, [0.0f32; 4], [0.0f32, 0.0, 0.0, 1.0]);
        if let Some(geo) = s.geo.as_mut() {
            // globe: previous camera and the sun in this frame's coordinates, planet centre
            let rot = geo.frame;
            let rt = rot.transpose();
            for (k, row) in earth.iter_mut().enumerate() {
                let r = rt.row(k);
                *row = [r.x as f32, r.y as f32, r.z as f32, 0.0];
            }
            let now_e = [geo.ecef, rt * right.as_dvec3(), rt * up.as_dvec3(), rt * fwd.as_dvec3()];
            let p = geo.prev.unwrap_or(now_e);
            let v = |d: DVec3| (rot * d).as_vec3();
            let pp = v(p[0] - geo.ecef);
            let (pr, pu, pf) = (v(p[1]), v(p[2]), v(p[3]));
            prev = [[pp.x, pp.y, pp.z, s.lod], [pr.x, pr.y, pr.z, prev[1][3]], [pu.x, pu.y, pu.z, prev[2][3]], [pf.x, pf.y, pf.z, prev[3][3]]];
            geo.prev = Some(now_e);
            // the clock, the sun and the moon (Earth-fixed), the light: the sun by day, the moon
            // by night (once the sun is 8 degrees down and the sky has gone dark)
            geo.time += dt as f64 * geo.rate;
            let sun_e = globe::sky::sun(geo.time);
            let moon_v = globe::sky::moon(geo.time) - geo.ecef / 1000.0; // km, from the camera
            let (moon_e, moon_km) = (moon_v.normalize(), moon_v.length());
            let up_e = geo.ecef.normalize();
            let alpha = sun_e.dot(-moon_e).clamp(-1.0, 1.0).acos(); // phase angle (0 = full)
            let moon_i = 4.6 * globe::sky::moonlight(alpha, moon_km);
            let night = sun_e.dot(up_e) < (-8f64).to_radians().sin();
            let (light, light_i) = if night { (moon_e, moon_i) } else { (sun_e, 4.6) };
            sun = v(light).normalize();
            sun_w = light_i as f32;
            let ms = v(moon_e).normalize();
            moon = [ms.x, ms.y, ms.z, (globe::sky::MOON_RADIUS_KM / moon_km).atan() as f32];
            let ss = v(sun_e).normalize();
            // eye adaptation from the light on the ground (daylight fading through twilight)
            let day = 4.6 * smoothstep(-0.14, 0.2, sun_e.dot(up_e));
            let moon_up = moon_i * smoothstep(-0.05, 0.1, moon_e.dot(up_e));
            // (a half moon gives ~2.4e-6 of daylight; ^0.85 leaves a moonlit night ~3 stops darker
            // than day, as a dark-adapted eye sees it; airglow and starlight keep a floor)
            let target = ((4.6 / (day + moon_up + 2e-8)).powf(0.85)).clamp(1.0, 1e6) as f32;
            geo.adapt = if geo.adapt == 0.0 || geo.rate == 0.0 { target } else { (geo.adapt.ln() + (target.ln() - geo.adapt.ln()) * (1.0 - (-3.0 * dt).exp())).exp() };
            // applied in main.rgen before the 16-bit targets (night radiance would flush to zero)
            sun_true = [ss.x, ss.y, ss.z, geo.adapt];
            let pc = v(-geo.ecef);
            planet = [pc.x, pc.y, pc.z, 1.0];
            let gg = globe::scene::GlobeGlobals { planet: [pc.x, pc.y, pc.z, globe::scene::RADIUS], ..Default::default() };
            s.r.set_globe_view(geo.ecef, rot, gg);
        }
        let g = Globals {
            cam_pos: cam_now[0],
            cam_right: cam_now[1],
            cam_up: cam_now[2],
            cam_fwd: cam_now[3],
            sun_dir: [sun.x, sun.y, sun.z, sun_w],
            params: [now.elapsed().as_secs_f32(), 0.0, s.exposure, 4.0],
            iparams: [s.accum, s.debug, s.frame, s.ao_rays],
            lod_params: [self.args.lod_band, s.lod2, self.args.lod2_band, 0.0],
            prev_pos: prev[0],
            prev_right: prev[1],
            prev_up: prev[2],
            prev_fwd: prev[3],
            extra: [s.haze, s.r.world.forest_radius, world::site().map(|s| s.alt_shift()).unwrap_or(0.0), (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64() % 3600.0).unwrap_or(0.0)) as f32],
            planet,
            moon,
            sun_true,
            earth,
            // z: globe tree thinning (1 smooth, 0 stepped; impostor.rint, globe.rchit)
            ocean: [[0.0; 4]; 3],
            cloud: [self.args.cloud_cover, if s.clouds && s.geo.is_some() { 1.0 } else { 0.0 }, if s.smooth_thin { 1.0 } else { 0.0 }, 0.0],
        };
        let last_view = views.is_empty() || view + 1 >= views.len();
        let shot_due = shot.is_some() && if views.is_empty() { s.rendered + 1 >= shot_frames } else { (s.rendered + 1) % shot_frames == 0 };
        let capture = shot_due || s.capture_next;
        // globe: a view is only captured once its terrain and trees have streamed in
        let settling = self.args.bench_move == 0.0 && s.r.globe.as_ref().map(|g| g.busy()).unwrap_or(false) && s.settle < 2000;
        let capture = capture && !settling;
        let (stats, img) = s.r.render(&g, s.cam.pos, s.lod, capture)?;
        s.frame = s.frame.wrapping_add(1);
        if settling && shot_due {
            s.settle += 1; // hold this view
            if s.settle == 400 {
                if let Some(gs) = s.r.globe.as_ref() {
                    println!("still streaming after 400 held frames: {}", gs.busy_detail());
                }
            }
            s.frame_times.clear();
        } else {
            s.settle = 0;
            s.rendered += 1;
        }
        if let Some(img) = img {
            let path = if shot_due && !views.is_empty() {
                let p = shot.clone().unwrap();
                format!("{}_{view}.png", p.trim_end_matches(".png"))
            } else if shot_due {
                shot.clone().unwrap()
            } else {
                s.shot_counter += 1;
                format!("shots/shot_{:03}.png", s.shot_counter)
            };
            save_png(&path, w, h, &img)?;
            println!("Saved {path}");
            s.capture_next = false;
            if shot_due {
                if !s.frame_times.is_empty() {
                    let mut t = s.frame_times.clone();
                    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let avg = t.iter().sum::<f32>() / t.len() as f32;
                    println!(
                        "timing over {} frames: avg {:.2} ms ({:.0} fps), median {:.2} ms, p95 {:.2} ms, p99 {:.2} ms, max {:.2} ms, frames > 33 ms: {}",
                        t.len(),
                        avg,
                        1000.0 / avg,
                        t[t.len() / 2],
                        t[(t.len() * 95 / 100).min(t.len() - 1)],
                        t[(t.len() * 99 / 100).min(t.len() - 1)],
                        t[t.len() - 1],
                        t.iter().filter(|&&x| x > 33.3).count()
                    );
                }
                println!("{}", s.r.profile.line());
                if std::env::var("SI_MEM").is_ok() {
                    print!("{}", s.r.gpu.memory_report());
                }
                if let Some(scene) = &s.r.globe {
                    println!("{}", scene.stats_line());
                    if let Some(geo) = &s.geo {
                        let (lat, lon) = globe::tiles::lat_lon(geo.ecef);
                        let h = geo.ecef.length() - globe::tiles::ground_radius(&scene.terrain, geo.ecef);
                        let hd = globe::scene::heading(geo.frame, s.cam.basis().0.as_dvec3());
                        println!("camera --at {lat:.5},{lon:.5},{h:.1},{hd:.1},{:.1}", s.cam.pitch.to_degrees());
                    }
                }
                println!(
                    "stats: near trees {}, TLAS instances {}, camera --cam {:.2},{:.2},{:.2},{:.1},{:.1}",
                    stats.near_trees,
                    stats.instances,
                    s.cam.pos.x,
                    s.cam.pos.y,
                    s.cam.pos.z,
                    s.cam.yaw.to_degrees(),
                    s.cam.pitch.to_degrees()
                );
                if last_view {
                    el.exit();
                }
            }
        }

        s.fps_frames += 1;
        let el_t = s.fps_time.elapsed().as_secs_f32();
        if el_t > 0.5 {
            let fps = s.fps_frames as f32 / el_t;
            let mode = ["lit", "LOD view", "normals", "albedo", "AO", "sun visibility"][s.debug.min(5) as usize];
            let cam_mode = if let Some(geo) = &s.geo {
                let rate = match geo.rate {
                    r if r == 0.0 => "stopped".to_string(),
                    r if r == 1.0 => "real time".to_string(),
                    r if r < 3600.0 => format!("{:.0} min/s", r / 60.0),
                    r if r < 86_400.0 => format!("{:.0} h/s", r / 3600.0),
                    r => format!("{:.0} day/s", r / 86_400.0),
                };
                format!("{} ({rate}; , . time, / speed) | {:.0} m/s", globe::sky::format_utc(geo.time), s.base_speed)
            } else if s.fly_on {
                "AUTO fly (F to stop)".to_string()
            } else {
                format!("manual {:.0} m/s{}{}", s.base_speed, if s.cinematic { ", cinematic" } else { "" }, if s.captured { "" } else { ", click to look" })
            };
            s.window.set_title(&format!(
                "SmartImposter | {:.0} fps ({:.2} ms) | {} | sharpen {:.2} (H) | {} | {} near plants (mesh) / {} total | {} TLAS instances | mesh < {:.0} m ([ ]) | {}",
                fps,
                1000.0 / fps,
                match s.r.dlss_mode() {
                    Some(m) => format!("DLSS-RR {} (L off, J mode)", dlss::MODES[m].0),
                    None => format!(
                        "SS {:.1}x (K) | TAA {} (T){}",
                        s.r.render_scale(),
                        if s.r.taa_enabled { "on" } else { "off" },
                        if s.r.dlss_available() { " | DLSS off (L)" } else { "" }
                    ),
                },
                s.r.sharpness,
                cam_mode,
                stats.near_trees,
                if s.geo.is_some() { s.r.globe_trees } else { s.r.world.trees.len() },
                stats.instances,
                s.lod,
                mode,
            ));
            s.fps_time = Instant::now();
            s.fps_frames = 0;
        }
        Ok(())
    }

    fn key(&mut self, code: KeyCode, pressed: bool, repeat: bool, el: &ActiveEventLoop) {
        let Some(s) = self.state.as_mut() else { return };
        if pressed {
            s.keys.insert(code);
        } else {
            s.keys.remove(&code);
        }
        if !pressed || repeat {
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
            KeyCode::KeyC => s.cinematic = !s.cinematic,
            KeyCode::KeyT => s.r.taa_enabled = !s.r.taa_enabled,
            KeyCode::KeyU => {
                s.clouds = !s.clouds;
                println!("clouds: {}", if s.clouds { "on" } else { "off" });
            }
            KeyCode::KeyN => {
                s.smooth_thin = !s.smooth_thin;
                println!("tree thinning: {}", if s.smooth_thin { "smooth" } else { "stepped (per tile)" });
            }
            KeyCode::KeyH => s.r.sharpness = if s.r.sharpness >= 0.99 { 0.0 } else { (s.r.sharpness + 0.25).min(1.0) },
            KeyCode::KeyG => s.haze = if s.haze > 0.0 { 0.0 } else { 1.0 },
            KeyCode::KeyL => {
                let next = if s.r.dlss_mode().is_some() { None } else { Some(s.dlss_last) };
                if let Err(e) = s.r.set_dlss(next) {
                    eprintln!("DLSS switch failed: {e:?}");
                }
                s.prev_cam = None;
            }
            KeyCode::KeyJ => {
                if let Some(m) = s.r.dlss_mode() {
                    s.dlss_last = (m + 1) % dlss::MODES.len();
                    if let Err(e) = s.r.set_dlss(Some(s.dlss_last)) {
                        eprintln!("DLSS mode change failed: {e:?}");
                    }
                    s.prev_cam = None;
                }
            }
            KeyCode::KeyK => {
                let next = match s.r.render_scale() {
                    x if x < 1.25 => 1.5,
                    x if x < 1.75 => 2.0,
                    _ => 1.0,
                };
                if let Err(e) = s.r.set_render_scale(next) {
                    eprintln!("render scale change failed: {e:?}");
                }
            }
            KeyCode::Digit0 => s.debug = 0,
            KeyCode::Digit1 => s.debug = 1,
            KeyCode::Digit2 => s.debug = 2,
            KeyCode::Digit3 => s.debug = 3,
            KeyCode::Digit4 => s.debug = 4,
            KeyCode::Digit5 => s.debug = 5,
            KeyCode::BracketLeft => s.lod = (s.lod - 10.0).max(0.0),
            KeyCode::BracketRight => s.lod = (s.lod + 10.0).min(300.0),
            KeyCode::Semicolon => s.lod2 = (s.lod2 - 10.0).max(0.0),
            KeyCode::Quote => s.lod2 = (s.lod2 + 10.0).min(600.0),
            KeyCode::KeyO => s.ao_rays = (s.ao_rays + 1) % 3,
            // globe: the clock (, . or left / right: an hour, Shift: a day; up / down: a day; / its speed)
            KeyCode::Comma | KeyCode::Period | KeyCode::ArrowLeft | KeyCode::ArrowRight | KeyCode::ArrowUp | KeyCode::ArrowDown | KeyCode::Slash
                if s.geo.is_some() =>
            {
                let shift = s.keys.contains(&KeyCode::ShiftLeft) || s.keys.contains(&KeyCode::ShiftRight);
                let geo = s.geo.as_mut().unwrap();
                let step = if shift { 86_400.0 } else { 3600.0 };
                match code {
                    KeyCode::Comma | KeyCode::ArrowLeft => geo.time -= step,
                    KeyCode::Period | KeyCode::ArrowRight => geo.time += step,
                    KeyCode::ArrowUp => geo.time += 86_400.0,
                    KeyCode::ArrowDown => geo.time -= 86_400.0,
                    _ => {
                        const RATES: [f64; 5] = [1.0, 60.0, 3600.0, 86_400.0, 0.0];
                        let i = RATES.iter().position(|r| *r == geo.rate).map(|i| (i + 1) % RATES.len()).unwrap_or(0);
                        geo.rate = RATES[i];
                    }
                }
            }
            KeyCode::ArrowLeft => s.sun_az -= 10.0,
            KeyCode::ArrowRight => s.sun_az += 10.0,
            KeyCode::ArrowUp => s.sun_el = (s.sun_el + 5.0).min(89.0),
            KeyCode::ArrowDown => s.sun_el = (s.sun_el - 5.0).max(3.0),
            KeyCode::Minus => s.exposure /= 1.15,
            KeyCode::Equal => s.exposure *= 1.15,
            KeyCode::F12 => s.capture_next = true,
            KeyCode::KeyM => {
                // back to the globe, above this place
                if let (Some(site), Ok(exe)) = (world::site(), std::env::current_exe()) {
                    let (lat, lon) = site.lat_lon(s.cam.pos.x, s.cam.pos.z);
                    let alt = (s.cam.pos.y - s.r.world.surface_at(s.cam.pos.x, s.cam.pos.z)).max(2.0);
                    let size = s.window.inner_size();
                    let at = format!("{lat:.6},{lon:.6},{alt:.1},{:.1},{:.1}", s.cam.yaw.to_degrees(), s.cam.pitch.to_degrees());
                    let _ = std::process::Command::new(exe)
                        .args(["--globe", "--at", &at, "--size", &format!("{}x{}", size.width, size.height)])
                        .spawn();
                    el.exit();
                }
            }
            KeyCode::KeyF => {
                s.fly_on = !s.fly_on;
                s.vel = Vec3::ZERO;
                s.yaw_target = s.cam.yaw;
                s.pitch_target = s.cam.pitch;
            }
            KeyCode::F11 => {
                let fs = s.window.fullscreen().is_some();
                s.window.set_fullscreen(if fs { None } else { Some(winit::window::Fullscreen::Borderless(None)) });
            }
            KeyCode::KeyP => match (&s.geo, &s.r.globe) {
                (Some(geo), Some(scene)) => {
                    let (lat, lon) = globe::tiles::lat_lon(geo.ecef);
                    let h = geo.ecef.length() - globe::tiles::ground_radius(&scene.terrain, geo.ecef);
                    let (fwd, _, _) = s.cam.basis();
                    let hd = globe::scene::heading(geo.frame, fwd.as_dvec3());
                    println!("--at {lat:.5},{lon:.5},{h:.1},{hd:.1},{:.1}", s.cam.pitch.to_degrees());
                }
                _ => println!(
                    "--cam {:.2},{:.2},{:.2},{:.1},{:.1}",
                    s.cam.pos.x,
                    s.cam.pos.y,
                    s.cam.pos.z,
                    s.cam.yaw.to_degrees(),
                    s.cam.pitch.to_degrees()
                ),
            },
            _ => {}
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.state.is_none() {
            if let Err(e) = self.init(el) {
                eprintln!("init failed: {e:?}");
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
                    self.key(code, event.state == ElementState::Pressed, event.repeat, el);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(s) = self.state.as_mut() {
                    match button {
                        // Left click: capture the mouse for free look (Esc releases).
                        MouseButton::Left if state == ElementState::Pressed && !s.captured => {
                            s.captured = true;
                            if s.window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                                let _ = s.window.set_cursor_grab(CursorGrabMode::Confined);
                            }
                            s.window.set_cursor_visible(false);
                        }
                        // Right button: hold to look without capturing.
                        MouseButton::Right if !s.captured => {
                            s.looking = state == ElementState::Pressed;
                            let _ = s
                                .window
                                .set_cursor_grab(if s.looking { CursorGrabMode::Confined } else { CursorGrabMode::None });
                            s.window.set_cursor_visible(!s.looking);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(s) = self.state.as_mut() {
                    let steps = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                    };
                    s.base_speed = (s.base_speed * 1.2f32.powf(steps)).clamp(0.5, 300.0);
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
                    if s.fly_on {
                        // Mouse look takes over from the automatic path.
                        s.fly_on = false;
                        s.vel = Vec3::ZERO;
                        s.yaw_target = s.cam.yaw;
                        s.pitch_target = s.cam.pitch;
                    }
                    s.yaw_target += delta.0 as f32 * 0.0022;
                    s.pitch_target = (s.pitch_target - delta.1 as f32 * 0.0022).clamp(-1.55, 1.55);
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

fn main() -> Result<()> {
    // `--convert-dem a.tif ...`: GLO-30 GeoTIFFs -> compact .dem tiles (tools/fetch_earth.py)
    let raw: Vec<String> = std::env::args().collect();
    if raw.get(1).map(String::as_str) == Some("--convert-dem") {
        return globe::dem::convert_cli(&raw[2..]);
    }
    let args = parse_args();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { args, state: None };
    event_loop.run_app(&mut app)?;
    if let Some(s) = app.state.as_ref() {
        unsafe {
            let _ = s.r.gpu.device.device_wait_idle();
        }
    }
    // GPU resources are reclaimed by the driver at process exit.
    std::process::exit(0);
}
