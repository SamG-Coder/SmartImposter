//! Material textures exported from the Unreal Megaplant (tools/ue_export.py): decode, upload,
//! generate mips on the GPU. Also the per-species sets and the ground layers.

use crate::gpu::{color_range, Gpu, Image};
use anyhow::{bail, Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::path::{Path, PathBuf};

/// How a texture is stored on the GPU (all block-compressed, visually lossless: >= 35 dB):
/// colour maps as BC7, normal x / y as BC5, a single data channel (translucency, occlusion) as
/// BC4. A normal map with a third channel in use becomes two textures (BC5 + BC4): one BC7
/// block cannot hold three unrelated channels well.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Enc {
    Colour,
    Rg,
    Ch(usize),
}

/// Species texture slots in binding order (SPECIES_SLOTS per species; tri.rchit): source (index
/// into the catalog's CA / NT / C / NAH paths), format, encoding, placeholder.
pub const SPECIES_SLOTS: usize = 6;
const SLOTS: [(usize, vk::Format, Enc, [u8; 4]); SPECIES_SLOTS] = [
    (0, vk::Format::R8G8B8A8_SRGB, Enc::Colour, [40, 70, 20, 255]),     // foliage colour + alpha
    (1, vk::Format::R8G8B8A8_UNORM, Enc::Rg, [128, 128, 85, 255]),      // foliage normal x / y
    (1, vk::Format::R8G8B8A8_UNORM, Enc::Ch(2), [85, 85, 85, 255]),     // foliage translucency
    (2, vk::Format::R8G8B8A8_SRGB, Enc::Colour, [70, 58, 46, 255]),     // bark colour
    (3, vk::Format::R8G8B8A8_UNORM, Enc::Rg, [128, 128, 255, 255]),     // bark normal x / y
    (3, vk::Format::R8G8B8A8_UNORM, Enc::Ch(2), [255, 255, 255, 255]),  // bark occlusion
];

pub fn find(dir: &Path, suffix: &str, foliage: bool) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
        let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        n.ends_with(suffix) && n.contains("Foliage") == foliage && (n.contains("Foliage") || n.contains("Bark"))
    })
}

pub fn decode_rgba(path: &Path) -> Result<(u32, u32, Vec<u8>)> {
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
        if mips > 1 {
            barrier(cb, 0, mips - 1, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags2::TRANSFER_READ, vk::AccessFlags2::SHADER_READ, t, vk::PipelineStageFlags2::ALL_COMMANDS);
        }
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

/// BC7-compressed textures (bc7.rs): a quarter of RGBA8 in memory, visually lossless. Mips are
/// made on the CPU (sRGB averaged in linear light, as the GPU blit did) and every level encoded;
/// the result is cached in assets/cache/tex keyed by the source file, so later runs skip the PNG
/// decode too.
const BC7_CACHE: &str = "assets/cache/tex";
const BC7_VERSION: u32 = 1;

struct Bc7Tex {
    w: u32,
    h: u32,
    /// encoded blocks of every mip level, largest first
    levels: Vec<Vec<u8>>,
    /// PSNR of the largest level against the source (dB); 0 when read from the cache
    psnr: f64,
    /// the source pixels (for the luminance statistics) when freshly encoded
    px: Option<Vec<u8>>,
    /// statistics from the cache
    stats: Option<(f32, f32)>,
}

fn gpu_format(f: vk::Format, enc: Enc) -> vk::Format {
    match (enc, f) {
        (Enc::Colour, vk::Format::R8G8B8A8_SRGB) => vk::Format::BC7_SRGB_BLOCK,
        (Enc::Colour, _) => vk::Format::BC7_UNORM_BLOCK,
        (Enc::Rg, _) => vk::Format::BC5_UNORM_BLOCK,
        (Enc::Ch(_), _) => vk::Format::BC4_UNORM_BLOCK,
    }
}

/// Bytes of one level's blocks.
fn level_bytes(w: u32, h: u32, enc: Enc) -> usize {
    let blocks = (w.div_ceil(4) * h.div_ceil(4)) as usize;
    blocks * if let Enc::Ch(_) = enc { 8 } else { 16 }
}

fn cache_path(src: &Path, max: u32, format: vk::Format, enc: Enc) -> Option<PathBuf> {
    use std::hash::{Hash, Hasher};
    let meta = std::fs::metadata(src).ok()?;
    let mtime = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (src.to_string_lossy().as_ref(), meta.len(), mtime, max, format.as_raw(), enc, BC7_VERSION).hash(&mut h);
    Some(Path::new(BC7_CACHE).join(format!("{:016x}.bc7", h.finish())))
}

fn read_cache(path: &Path, enc: Enc) -> Option<Bc7Tex> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 28 || &b[..8] != b"SIBC7\0\0\x01" {
        return None;
    }
    let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (w, h, mips) = (u(8), u(12), u(16));
    let stats = (f(20), f(24));
    let mut levels = Vec::new();
    let mut o = 28;
    for m in 0..mips {
        let (mw, mh) = ((w >> m).max(1), (h >> m).max(1));
        let n = level_bytes(mw, mh, enc);
        levels.push(b.get(o..o + n)?.to_vec());
        o += n;
    }
    Some(Bc7Tex { w, h, levels, psnr: 0.0, px: None, stats: Some(stats) })
}

fn write_cache(path: &Path, t: &Bc7Tex, stats: (f32, f32)) {
    let mut out = Vec::new();
    out.extend_from_slice(b"SIBC7\0\0\x01");
    for v in [t.w, t.h, t.levels.len() as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&stats.0.to_le_bytes());
    out.extend_from_slice(&stats.1.to_le_bytes());
    for l in &t.levels {
        out.extend_from_slice(l);
    }
    let _ = std::fs::create_dir_all(BC7_CACHE);
    let tmp = path.with_extension("part");
    if std::fs::write(&tmp, out).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// The mip chain of an RGBA8 image (2x2 box filter; sRGB colour averaged in linear light).
fn mip_chain(w: u32, h: u32, px: &[u8], srgb: bool) -> Vec<(u32, u32, Vec<u8>)> {
    let to_lin: Vec<f32> = (0..256)
        .map(|c| {
            let c = c as f32 / 255.0;
            if !srgb {
                c
            } else if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let to_8 = |v: f32| -> u8 {
        let v = v.clamp(0.0, 1.0);
        let e = if !srgb { v } else if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (e * 255.0 + 0.5) as u8
    };
    let mut out = vec![(w, h, px.to_vec())];
    let (mut cw, mut ch) = (w, h);
    while cw > 1 || ch > 1 {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let src = &out.last().unwrap().2;
        let mut dst = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let xs = [(2 * x).min(cw - 1), (2 * x + 1).min(cw - 1)];
                let ys = [(2 * y).min(ch - 1), (2 * y + 1).min(ch - 1)];
                for c in 0..4 {
                    let mut sum = 0.0;
                    for yy in ys {
                        for xx in xs {
                            let v = src[((yy * cw + xx) * 4 + c) as usize];
                            sum += if c == 3 { v as f32 / 255.0 } else { to_lin[v as usize] };
                        }
                    }
                    let a = sum / 4.0;
                    dst[((y * nw + x) * 4 + c) as usize] = if c == 3 { (a * 255.0 + 0.5) as u8 } else { to_8(a) };
                }
            }
        }
        out.push((nw, nh, dst));
        (cw, ch) = (nw, nh);
    }
    out
}

/// Encode an image and its mips (BC7, BC5 or BC4).
fn encode_bc7(w: u32, h: u32, px: Vec<u8>, srgb: bool, enc: Enc) -> Bc7Tex {
    let chain = mip_chain(w, h, &px, srgb);
    let chans: Vec<usize> = match enc {
        Enc::Colour => Vec::new(),
        Enc::Rg => vec![0, 1],
        Enc::Ch(c) => vec![c],
    };
    let code = |mw: u32, mh: u32, p: &[u8]| if chans.is_empty() { crate::bc7::encode_image(mw, mh, p) } else { crate::bc7::encode_channels(mw, mh, p, &chans) };
    let levels: Vec<Vec<u8>> = chain.iter().map(|(mw, mh, p)| code(*mw, *mh, p)).collect();
    let psnr = if chans.is_empty() { crate::bc7::psnr(w, h, &px, &levels[0]) } else { crate::bc7::psnr_channels(w, h, &px, &levels[0], &chans) };
    Bc7Tex { w, h, levels, psnr, px: Some(px), stats: None }
}

/// Upload BC7 levels to a sampled image.
fn upload_bc7(gpu: &Gpu, t: &Bc7Tex, format: vk::Format, name: &str) -> Result<Image> {
    let mips = t.levels.len() as u32;
    let img = gpu.create_image(t.w, t.h, mips, format, vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST, name)?;
    let total: usize = t.levels.iter().map(|l| l.len()).sum();
    let mut staging = gpu.create_buffer(total as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "tex staging")?;
    let mut regions = Vec::new();
    let mut o = 0usize;
    for (m, l) in t.levels.iter().enumerate() {
        staging.mapped()[o..o + l.len()].copy_from_slice(l);
        regions.push(vk::BufferImageCopy {
            buffer_offset: o as u64,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: m as u32, base_array_layer: 0, layer_count: 1 },
            image_extent: vk::Extent3D { width: (t.w >> m).max(1), height: (t.h >> m).max(1), depth: 1 },
            ..Default::default()
        });
        o += l.len();
    }
    let d = &gpu.device;
    let barrier = |cb: vk::CommandBuffer, old, new, sa, da, ss, ds| {
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
            .subresource_range(color_range(0, mips))];
        unsafe { d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
    };
    let t2 = vk::PipelineStageFlags2::TRANSFER;
    gpu.one_time(|cb| unsafe {
        barrier(cb, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::AccessFlags2::NONE, vk::AccessFlags2::TRANSFER_WRITE, vk::PipelineStageFlags2::NONE, t2);
        d.cmd_copy_buffer_to_image(cb, staging.raw, img.raw, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);
        barrier(cb, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::AccessFlags2::TRANSFER_WRITE, vk::AccessFlags2::SHADER_READ, t2, vk::PipelineStageFlags2::ALL_COMMANDS);
    })?;
    gpu.destroy_buffer(&mut staging);
    Ok(img)
}

/// One texture to load: path (None = 1x1 placeholder of `fill`) and format.
struct Job {
    path: Option<PathBuf>,
    format: vk::Format,
    fill: [u8; 4],
    enc: Enc,
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

/// Load all textures on worker threads (capped at `max` texels per side): from the BC7 cache, or
/// decoded, mip-mapped and BC7-encoded, then uploaded. Also returns each texture's statistics
/// (mean linear luminance, mean blue; 0 for placeholders).
/// Textures and per-texture statistics (mean linear luminance, mean blue).
type Loaded = (Vec<Image>, Vec<(f32, f32)>);

fn load_jobs(gpu: &Gpu, jobs: &[Job], max: u32) -> Result<Loaded> {
    let t = std::time::Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done: Vec<std::sync::Mutex<Option<Result<Bc7Tex>>>> = jobs.iter().map(|_| std::sync::Mutex::new(None)).collect();
    // every core: encoding a texture the first time is the slow part
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let job = &jobs[i];
                let Some(p) = &job.path else { continue };
                let cache = cache_path(p, max, job.format, job.enc);
                if let Some(c) = cache.as_deref().and_then(|c| read_cache(c, job.enc)) {
                    *done[i].lock().unwrap() = Some(Ok(c));
                    continue;
                }
                let srgb = job.format == vk::Format::R8G8B8A8_SRGB;
                let r = decode_rgba(p).with_context(|| format!("decoding {}", p.display())).map(|(w, h, px)| {
                    let (w, h, px) = shrink(w, h, px, max);
                    encode_bc7(w, h, px, srgb, job.enc)
                });
                *done[i].lock().unwrap() = Some(r);
            });
        }
    });
    let mut out = Vec::new();
    let mut lums = Vec::new();
    let mut bytes = 0u64;
    let (mut encoded, mut psnr_min, mut psnr_sum) = (0usize, f64::MAX, 0.0);
    let mut prev: Vec<u8> = Vec::new(); // a normal map follows its colour map: the colour's alpha is the mask
    for (job, slot) in jobs.iter().zip(done) {
        match slot.into_inner().unwrap() {
            Some(r) => {
                let tex = r?;
                let st = match (&tex.px, tex.stats) {
                    (Some(px), _) => {
                        let st = if job.format == vk::Format::R8G8B8A8_UNORM { stats(px, &prev) } else { (mean_luminance(px), 0.0) };
                        if let Some(c) = cache_path(job.path.as_ref().unwrap(), max, job.format, job.enc) {
                            write_cache(&c, &tex, st);
                        }
                        encoded += 1;
                        psnr_min = psnr_min.min(tex.psnr);
                        psnr_sum += tex.psnr;
                        st
                    }
                    (None, Some(st)) => st,
                    _ => (0.0, 0.0),
                };
                lums.push(st);
                let name = job.path.as_ref().unwrap().file_name().unwrap().to_string_lossy().into_owned();
                bytes += tex.levels.iter().map(|l| l.len() as u64).sum::<u64>();
                out.push(upload_bc7(gpu, &tex, gpu_format(job.format, job.enc), &name)?);
                if job.enc == Enc::Colour {
                    prev = tex.px.unwrap_or_default();
                }
            }
            None => {
                lums.push((0.0, 0.0));
                if job.enc == Enc::Colour {
                    prev.clear();
                }
                out.push(upload_with_mips(gpu, 1, 1, &job.fill, job.format, "placeholder")?);
            }
        }
    }
    let loaded = jobs.iter().filter(|j| j.path.is_some()).count();
    if loaded > 0 {
        println!("Textures: {loaded} loaded ({:.0} MB on GPU, BC7 / BC5 / BC4) in {:.2}s", bytes as f64 / 1e6, t.elapsed().as_secs_f32());
        if encoded > 0 {
            println!("  {encoded} encoded and cached, PSNR mean {:.1} dB, lowest {:.1} dB", psnr_sum / encoded as f64, psnr_min);
        }
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
/// The species textures, per species (foliage brightness scale, translucency mask mean), and
/// whether any species has foliage textures.
type SpeciesTextures = (Vec<Image>, Vec<(f32, f32)>, bool);

/// `used`: the species the scene shows; the others keep their slots as 1x1 placeholders.
pub fn load_species(gpu: &Gpu, cat: &crate::catalog::Catalog, used: &[bool]) -> Result<SpeciesTextures> {
    if let Some([leaf,bark]) = cat.tree_palette {
        let mut maps = Vec::new();
        for (slot, &(_,format,_,fill)) in SLOTS.iter().enumerate() {
            if slot != 0 && slot != 3 {
                maps.push(upload_with_mips(gpu,1,1,&fill,format,"tree material constant")?);
                continue;
            }
            let mut pixels = Vec::with_capacity(64*64*4);
            for y in 0..64 {
                for x in 0..64 {
                    let (u,v) = ((x as f32+0.5)/64.0,(y as f32+0.5)/64.0);
                    let (rgb,gain) = if slot == 0 {
                        let rib = (1.0-(u-0.5).abs()*42.0).max(0.0);
                        let vein = ((v*7.0-(u-0.5).abs()*2.5)*std::f32::consts::TAU).cos().max(0.0).powi(9);
                        (leaf,0.97+0.18*rib+0.09*vein)
                    } else {
                        let phase = std::f32::consts::TAU;
                        let grain = (phase*(u*13.0+0.17*(phase*v).sin()+0.12*(phase*(3.0*u+v)).cos())).sin();
                        (bark,1.09-0.32*grain.max(0.0).powi(3))
                    };
                    pixels.extend(rgb.map(|c| (c as f32*gain).round() as u8));
                    pixels.push(255);
                }
            }
            maps.push(upload_with_mips(gpu,64,64,&pixels,format,"grown tree colour")?);
        }
        return Ok((maps,vec![(1.0,0.33)],true));
    }
    let mut jobs = Vec::new();
    for (i, s) in cat.species.iter().enumerate() {
        for &(src, format, enc, fill) in &SLOTS {
            let path = s.textures[src].clone().filter(|p| used[i] && p.exists());
            jobs.push(Job { path, format, fill, enc });
        }
    }
    if jobs.is_empty() {
        for &(_, format, enc, fill) in &SLOTS {
            jobs.push(Job { path: None, format, fill, enc });
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
            let l = lums.get(i * SPECIES_SLOTS).map(|x| x.0).unwrap_or(0.0);
            let tb = lums.get(i * SPECIES_SLOTS + 1).map(|x| x.1).unwrap_or(0.0);
            // Grass is matched to the ground textures it stands on (the scans are 2-7x brighter).
            let target = if s.kind == crate::catalog::Kind::Grass { 0.085 } else { FOLIAGE_LUMINANCE }; // ~ the ground scans
            let k = if l > 0.0 { (target / l).clamp(0.12, 1.6) } else { 1.0 };
            if used[i] {
                println!("  {:<24} foliage luminance {:.3} -> x{:.2}, translucency mean {:.2}", s.name, l, k, tb);
            }
            (k, if tb > 0.0 { tb } else { 0.33 })
        })
        .collect();
    Ok((images, scales, has))
}

/// Ground layers: albedo (sRGB), normal, ORM per layer, in catalog order.
/// `used`: the scans the scene shows (the others are not loaded).
pub fn load_ground(gpu: &Gpu, ground: &[Option<crate::catalog::GroundMaterial>], used: &[bool]) -> Result<Vec<Image>> {
    let mut jobs = Vec::new();
    for (i, g) in ground.iter().enumerate() {
        // Missing layers keep their slots (1x1 placeholders): the shader indexes layers by position.
        let p = |f: fn(&crate::catalog::GroundMaterial) -> &PathBuf| g.as_ref().filter(|_| used[i]).map(|g| f(g).clone());
        // colour BC7, normal x / y BC5, occlusion (the only ORM channel read) BC4
        jobs.push(Job { path: p(|g| &g.albedo), format: vk::Format::R8G8B8A8_SRGB, fill: [100, 100, 100, 255], enc: Enc::Colour });
        jobs.push(Job { path: p(|g| &g.normal), format: vk::Format::R8G8B8A8_UNORM, fill: [128, 128, 255, 255], enc: Enc::Rg });
        jobs.push(Job { path: p(|g| &g.orm), format: vk::Format::R8G8B8A8_UNORM, fill: [255, 200, 0, 255], enc: Enc::Ch(0) });
    }
    if jobs.is_empty() {
        jobs.push(Job { path: None, format: vk::Format::R8G8B8A8_SRGB, fill: [60, 60, 40, 255], enc: Enc::Colour });
    } else {
        println!(
            "Ground layers: {}",
            crate::catalog::GROUND_LAYERS
                .iter()
                .zip(ground)
                .map(|(n, g)| if g.is_some() { n.to_string() } else { format!("({n} missing: procedural)") })
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // 26 layers x 3 maps at 2048 texels: ~0.45 GB in BC7 (SI_GROUND_TEX_MAX to change it).
    let max = std::env::var("SI_GROUND_TEX_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(2048).min(tex_max());
    Ok(load_jobs(gpu, &jobs, max)?.0)
}

#[cfg(test)]
mod quality {
    /// BC7 fidelity per texture kind (run with --ignored --nocapture): the species textures and
    /// the ground scans, at 512 texels for speed.
    #[test]
    #[ignore]
    fn bc7_report() {
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>, depth: u32) {
            if depth > 6 {
                return;
            }
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        walk(&p, out, depth + 1);
                    } else if p.extension().is_some_and(|x| x == "png") {
                        out.push(p);
                    }
                }
            }
        }
        walk(std::path::Path::new("assets/textures"), &mut files, 0);
        for g in crate::catalog::find_ground().into_iter().flatten().take(12) {
            files.extend([g.albedo, g.normal, g.orm]);
        }
        let mut by: std::collections::BTreeMap<String, Vec<(f64, String)>> = Default::default();
        for f in files.iter() {
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            let kind = name.rsplit('_').next().unwrap_or("").to_string();
            let Ok((w, h, px)) = super::decode_rgba(f) else { continue };
            let (w, h, px) = super::shrink(w, h, px, 512);
            // colour: BC7; normals: x / y as BC5 and the third channel as BC4; ORM: occlusion BC4
            let q = match kind.as_str() {
                "N.png" | "NT.png" | "NAH.png" => {
                    let rg = crate::bc7::encode_channels(w, h, &px, &[0, 1]);
                    let b = crate::bc7::encode_channels(w, h, &px, &[2]);
                    crate::bc7::psnr_channels(w, h, &px, &rg, &[0, 1]).min(if kind == "N.png" { 99.0 } else { crate::bc7::psnr_channels(w, h, &px, &b, &[2]) })
                }
                "ORM.png" => crate::bc7::psnr_channels(w, h, &px, &crate::bc7::encode_channels(w, h, &px, &[0]), &[0]),
                _ => crate::bc7::psnr(w, h, &px, &crate::bc7::encode_image(w, h, &px)),
            };
            by.entry(kind).or_default().push((q, name));
        }
        for (k, v) in &by {
            let mean = v.iter().map(|x| x.0).sum::<f64>() / v.len() as f64;
            let worst = v.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
            println!("{k:>10}: {} textures, mean {mean:.1} dB, worst {:.1} dB ({})", v.len(), worst.0, worst.1);
        }
    }
}
