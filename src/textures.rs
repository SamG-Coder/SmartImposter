//! Material textures exported from the Unreal Megaplant (tools/ue_export.py): decode, upload,
//! generate mips on the GPU. Also the per-species sets and the ground layers.

use crate::gpu::{color_range, Gpu, Image};
use anyhow::{bail, Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::path::{Path, PathBuf};

/// Species texture roles in binding order, matched by file-name suffix, with their formats.
const ROLES: [(&str, vk::Format); 4] = [
    ("_CA.png", vk::Format::R8G8B8A8_SRGB),   // foliage colour + alpha
    ("_NT.png", vk::Format::R8G8B8A8_UNORM),  // foliage normal + translucency
    ("_C.png", vk::Format::R8G8B8A8_SRGB),    // bark colour
    ("_NAH.png", vk::Format::R8G8B8A8_UNORM), // bark normal + AO + height
];

pub fn find(dir: &Path, suffix: &str, foliage: bool) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
        let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        n.ends_with(suffix) && n.contains("Foliage") == foliage && (n.contains("Foliage") || n.contains("Bark"))
    })
}

fn decode_rgba(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
    let mut dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;
    let px = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => px.to_vec(),
        png::ColorType::Rgb => px.chunks(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => px.chunks(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        png::ColorType::Grayscale => px.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        other => bail!("{}: unsupported PNG colour type {other:?}", path.display()),
    };
    Ok((info.width, info.height, rgba))
}

pub fn upload_with_mips(gpu: &Gpu, w: u32, h: u32, rgba: &[u8], format: vk::Format, name: &str) -> Result<Image> {
    let mips = 32 - w.max(h).leading_zeros();
    let img = gpu.create_image(
        w,
        h,
        mips,
        format,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
        name,
    )?;
    let mut staging = gpu.create_buffer(rgba.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "tex staging")?;
    staging.mapped()[..rgba.len()].copy_from_slice(rgba);
    let d = &gpu.device;
    let barrier = |cb: vk::CommandBuffer, mip: u32, count: u32, old, new, sa, da, ss, ds| {
        let b = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(ss)
            .src_access_mask(sa)
            .dst_stage_mask(ds)
            .dst_access_mask(da)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img.raw)
            .subresource_range(color_range(mip, count))];
        unsafe { d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
    };
    let t = vk::PipelineStageFlags2::TRANSFER;
    gpu.one_time(|cb| unsafe {
        barrier(cb, 0, mips, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags2::NONE, vk::AccessFlags2::TRANSFER_WRITE, vk::PipelineStageFlags2::NONE, t);
        let layers = |mip| vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: mip, base_array_layer: 0, layer_count: 1 };
        d.cmd_copy_buffer_to_image(
            cb,
            staging.raw,
            img.raw,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy { image_subresource: layers(0), image_extent: vk::Extent3D { width: w, height: h, depth: 1 }, ..Default::default() }],
        );
        for m in 1..mips {
            barrier(cb, m - 1, 1, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::AccessFlags2::TRANSFER_WRITE, vk::AccessFlags2::TRANSFER_READ, t, t);
            let (sw, sh) = ((w >> (m - 1)).max(1) as i32, (h >> (m - 1)).max(1) as i32);
            let (dw, dh) = ((w >> m).max(1) as i32, (h >> m).max(1) as i32);
            d.cmd_blit_image(
                cb,
                img.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                img.raw,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit {
                    src_subresource: layers(m - 1),
                    src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: sw, y: sh, z: 1 }],
                    dst_subresource: layers(m),
                    dst_offsets: [vk::Offset3D::default(), vk::Offset3D { x: dw, y: dh, z: 1 }],
                }],
                vk::Filter::LINEAR,
            );
        }
        barrier(cb, 0, mips - 1, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags2::TRANSFER_READ, vk::AccessFlags2::SHADER_READ, t, vk::PipelineStageFlags2::ALL_COMMANDS);
        barrier(cb, mips - 1, 1, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags2::TRANSFER_WRITE, vk::AccessFlags2::SHADER_READ, t, vk::PipelineStageFlags2::ALL_COMMANDS);
    })?;
    gpu.destroy_buffer(&mut staging);
    Ok(img)
}

/// Halve an RGBA8 image with a 2x2 box filter until it fits in `max` texels per side.
fn shrink(mut w: u32, mut h: u32, mut px: Vec<u8>, max: u32) -> (u32, u32, Vec<u8>) {
    while w.max(h) > max && w >= 2 && h >= 2 {
        let (nw, nh) = (w / 2, h / 2);
        let mut out = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh as usize {
            for x in 0..nw as usize {
                for c in 0..4 {
                    let at = |xx: usize, yy: usize| px[(yy * w as usize + xx) * 4 + c] as u32;
                    let s = at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1);
                    out[(y * nw as usize + x) * 4 + c] = ((s + 2) / 4) as u8;
                }
            }
        }
        (w, h, px) = (nw, nh, out);
    }
    (w, h, px)
}

/// One texture to load: path (None = 1x1 placeholder of `fill`) and format.
struct Job {
    path: Option<PathBuf>,
    format: vk::Format,
    fill: [u8; 4],
}

/// Mean linear luminance of an sRGB RGBA8 image over its opaque texels (all texels when the
/// alpha channel is empty).
fn mean_luminance(px: &[u8]) -> f32 {
    stats(px, px).0
}

/// (mean linear luminance of `px`, mean blue channel of `px`) over the texels that are opaque in
/// `mask` (all texels when the mask has no alpha).
fn stats(px: &[u8], mask: &[u8]) -> (f32, f32) {
    let lin = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let same_size = mask.len() == px.len();
    let has_alpha = same_size && mask.chunks(4).any(|p| p[3] > 128);
    let (mut sum, mut sum_b, mut n) = (0.0f64, 0.0f64, 0u64);
    for (i, p) in px.chunks(4).enumerate().step_by(7) {
        if has_alpha && mask[i * 4 + 3] <= 128 {
            continue;
        }
        sum += (0.2126 * lin(p[0]) + 0.7152 * lin(p[1]) + 0.0722 * lin(p[2])) as f64;
        sum_b += p[2] as f64 / 255.0;
        n += 1;
    }
    if n == 0 { (0.0, 0.0) } else { ((sum / n as f64) as f32, (sum_b / n as f64) as f32) }
}

/// Decode all PNGs on worker threads (capped at `max` texels per side), then upload with mips.
/// Also returns each texture's mean linear luminance (0 for placeholders).
fn load_jobs(gpu: &Gpu, jobs: &[Job], max: u32) -> Result<(Vec<Image>, Vec<(f32, f32)>)> {
    let t = std::time::Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let decoded: Vec<std::sync::Mutex<Option<Result<(u32, u32, Vec<u8>)>>>> = jobs.iter().map(|_| std::sync::Mutex::new(None)).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                if let Some(p) = &jobs[i].path {
                    let r = decode_rgba(p).with_context(|| format!("decoding {}", p.display())).map(|(w, h, px)| shrink(w, h, px, max));
                    *decoded[i].lock().unwrap() = Some(r);
                }
            });
        }
    });
    let mut out = Vec::new();
    let mut lums = Vec::new();
    let mut bytes = 0u64;
    let mut prev: Vec<u8> = Vec::new(); // foliage NT follows foliage CA: use its alpha as the mask
    for (job, slot) in jobs.iter().zip(decoded) {
        match slot.into_inner().unwrap() {
            Some(r) => {
                let (w, h, px) = r?;
                let st = if job.format == vk::Format::R8G8B8A8_UNORM { stats(&px, &prev) } else { (mean_luminance(&px), 0.0) };
                lums.push(st);
                bytes += w as u64 * h as u64 * 4 * 4 / 3;
                let name = job.path.as_ref().unwrap().file_name().unwrap().to_string_lossy().into_owned();
                out.push(upload_with_mips(gpu, w, h, &px, job.format, &name)?);
                prev = px;
            }
            None => {
                lums.push((0.0, 0.0));
                prev.clear();
                out.push(upload_with_mips(gpu, 1, 1, &job.fill, job.format, "placeholder")?);
            }
        }
    }
    let loaded = jobs.iter().filter(|j| j.path.is_some()).count();
    if loaded > 0 {
        println!("Textures: {loaded} loaded ({:.0} MB on GPU) in {:.2}s", bytes as f64 / 1e6, t.elapsed().as_secs_f32());
    }
    Ok((out, lums))
}

fn tex_max() -> u32 {
    std::env::var("SI_TEX_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048)
}

/// Foliage brightness every species is normalised to (mean linear luminance of the leaf
/// texture). Megaplant colour maps are authored very differently (Unreal's material darkens the
/// pale ones), so the raw textures range from ~0.04 to ~0.2.
const FOLIAGE_LUMINANCE: f32 = 0.09; // Black Alder's, which the lighting was tuned on

/// 4 textures per species (foliage CA, foliage NT, bark C, bark NAH); missing ones get neutral
/// placeholders. Returns the textures, per species (foliage brightness scale, translucency mask
/// mean), and false when no species has foliage textures (procedural leaf colours).
pub fn load_species(gpu: &Gpu, cat: &crate::catalog::Catalog) -> Result<(Vec<Image>, Vec<(f32, f32)>, bool)> {
    let fills = [[40, 70, 20, 255], [128, 128, 85, 255], [70, 58, 46, 255], [128, 128, 255, 255]];
    let mut jobs = Vec::new();
    for s in &cat.species {
        for (i, (_, format)) in ROLES.iter().enumerate() {
            let path = s.textures[i].clone().filter(|p| p.exists());
            jobs.push(Job { path, format: *format, fill: fills[i] });
        }
    }
    if jobs.is_empty() {
        for (i, (_, format)) in ROLES.iter().enumerate() {
            jobs.push(Job { path: None, format: *format, fill: fills[i] });
        }
    }
    let has = cat.species.iter().any(|s| s.textures[0].as_ref().is_some_and(|p| p.exists()));
    if !has {
        println!("Material textures not found - using procedural leaf colours");
    }
    let (images, lums) = load_jobs(gpu, &jobs, tex_max())?;
    let scales: Vec<(f32, f32)> = cat
        .species
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let l = lums.get(i * 4).map(|x| x.0).unwrap_or(0.0);
            let tb = lums.get(i * 4 + 1).map(|x| x.1).unwrap_or(0.0);
            // Grass is matched to the ground textures it stands on (the scans are 2-7x brighter).
            let target = if s.kind == crate::catalog::Kind::Grass { 0.085 } else { FOLIAGE_LUMINANCE }; // ~ the ground scans
            let k = if l > 0.0 { (target / l).clamp(0.12, 1.6) } else { 1.0 };
            println!("  {:<24} foliage luminance {:.3} -> x{:.2}, translucency mean {:.2}", s.name, l, k, tb);
            (k, if tb > 0.0 { tb } else { 0.33 })
        })
        .collect();
    Ok((images, scales, has))
}

/// Ground layers: albedo (sRGB), normal, ORM per layer, in catalog order.
pub fn load_ground(gpu: &Gpu, ground: &[crate::catalog::GroundMaterial]) -> Result<Vec<Image>> {
    let mut jobs = Vec::new();
    for g in ground {
        jobs.push(Job { path: Some(g.albedo.clone()), format: vk::Format::R8G8B8A8_SRGB, fill: [0; 4] });
        jobs.push(Job { path: Some(g.normal.clone()), format: vk::Format::R8G8B8A8_UNORM, fill: [0; 4] });
        jobs.push(Job { path: Some(g.orm.clone()), format: vk::Format::R8G8B8A8_UNORM, fill: [0; 4] });
    }
    if jobs.is_empty() {
        jobs.push(Job { path: None, format: vk::Format::R8G8B8A8_SRGB, fill: [60, 60, 40, 255] });
    } else {
        println!("Ground layers: {}", ground.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", "));
    }
    Ok(load_jobs(gpu, &jobs, tex_max())?.0)
}
