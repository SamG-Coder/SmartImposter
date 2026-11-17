//! Material textures exported from the Unreal Megaplant (tools/ue_export.py): decode, upload,
//! generate mips on the GPU.

use crate::gpu::{color_range, Gpu, Image};
use anyhow::{bail, Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use std::path::{Path, PathBuf};

/// Texture roles in binding order, matched by file-name suffix, with their formats.
const ROLES: [(&str, vk::Format); 4] = [
    ("_CA.png", vk::Format::R8G8B8A8_SRGB),   // foliage colour + alpha
    ("_NT.png", vk::Format::R8G8B8A8_UNORM),  // foliage normal + translucency
    ("_C.png", vk::Format::R8G8B8A8_SRGB),    // bark colour
    ("_NAH.png", vk::Format::R8G8B8A8_UNORM), // bark normal + AO + height
];

fn find(dir: &Path, suffix: &str, foliage: bool) -> Option<PathBuf> {
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

fn upload_with_mips(gpu: &Gpu, w: u32, h: u32, rgba: &[u8], format: vk::Format, name: &str) -> Result<Image> {
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

/// Load the 4 material textures from `dir`. Returns 1x1 neutral placeholders (and false) when
/// they are not there, so the renderer falls back to procedural leaf colours.
pub fn load(gpu: &Gpu, dir: &str) -> Result<(Vec<Image>, bool)> {
    let dir = Path::new(dir);
    let paths: Vec<Option<PathBuf>> =
        ROLES.iter().enumerate().map(|(i, (suffix, _))| find(dir, suffix, i < 2)).collect();
    let have_all = paths.iter().all(|p| p.is_some());
    let mut out = Vec::new();
    for (i, (_, format)) in ROLES.iter().enumerate() {
        if have_all {
            let path = paths[i].as_ref().unwrap();
            let t = std::time::Instant::now();
            let (w, h, rgba) = decode_rgba(path).with_context(|| format!("decoding {}", path.display()))?;
            out.push(upload_with_mips(gpu, w, h, &rgba, *format, &path.display().to_string())?);
            println!("Texture {} {}x{} ({:.2}s)", path.file_name().unwrap().to_string_lossy(), w, h, t.elapsed().as_secs_f32());
        } else {
            out.push(upload_with_mips(gpu, 1, 1, &[128, 128, 255, 255], *format, "placeholder")?);
        }
    }
    if !have_all {
        println!("Material textures not found in {} - using procedural leaf colours", dir.display());
    }
    Ok((out, have_all))
}
