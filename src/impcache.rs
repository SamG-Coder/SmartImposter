//! The baked impostor atlases (BC-compressed blocks of every mip of the four atlases per plant
//! variant) on disk, so the globe (--globe) draws exactly the trees the forest scene baked without
//! baking them itself. Written by the forest renderer after its bake when missing or stale.

use crate::renderer::ImpostorInfo;
use anyhow::{bail, Context, Result};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const CACHE_PATH: &str = "assets/cache/impostors_v1.bin";
const MAGIC: &[u8; 8] = b"SIIMPv1\0";

pub struct Variant {
    pub name: String,
    pub species: String,
    /// catalog::Kind as u32: 0 tree, 1 shrub, 2 plant, 3 grass
    pub kind: u32,
    pub height: f32,
    pub info: ImpostorInfo,
    /// [atlas (0 albedo, 1 normal, 2 depth, 3 visibility), mip level, size (texels), byte offset]
    pub jobs: Vec<[u32; 4]>,
    pub blocks: Vec<u8>,
}

/// Identifies a bake: variant names and their impostor layouts.
pub fn key(names: &[String], infos: &[ImpostorInfo]) -> String {
    let mut k = String::new();
    for (n, i) in names.iter().zip(infos) {
        k += &format!("{n}:{}x{}x{};", i.grid, i.res, i.mips);
    }
    k
}

pub fn needs_export(path: &Path, key: &str) -> bool {
    match read_key(path) {
        Ok(k) => k != key,
        Err(_) => true,
    }
}

fn read_key(path: &Path) -> Result<String> {
    let mut f = BufReader::new(std::fs::File::open(path)?);
    let mut m = [0u8; 8];
    f.read_exact(&mut m)?;
    if &m != MAGIC {
        bail!("not an impostor cache");
    }
    read_str(&mut f)
}

fn write_str(w: &mut impl Write, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u32).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_str(r: &mut impl Read) -> Result<String> {
    let n = read_u32(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8(b)?)
}

/// Returns the bytes written.
pub fn write(path: &Path, key: &str, vars: &[Variant]) -> Result<u64> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    let mut w = BufWriter::new(std::fs::File::create(&tmp)?);
    w.write_all(MAGIC)?;
    write_str(&mut w, key)?;
    w.write_all(&(vars.len() as u32).to_le_bytes())?;
    let mut total = 0u64;
    for v in vars {
        write_str(&mut w, &v.name)?;
        write_str(&mut w, &v.species)?;
        w.write_all(&v.kind.to_le_bytes())?;
        w.write_all(&v.height.to_le_bytes())?;
        w.write_all(bytemuck::bytes_of(&v.info))?;
        w.write_all(&(v.jobs.len() as u32).to_le_bytes())?;
        w.write_all(bytemuck::cast_slice(&v.jobs))?;
        w.write_all(&(v.blocks.len() as u64).to_le_bytes())?;
        w.write_all(&v.blocks)?;
        total += v.blocks.len() as u64;
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(total)
}

pub fn read(path: &Path) -> Result<Vec<Variant>> {
    let mut f = BufReader::new(std::fs::File::open(path).with_context(|| {
        format!("{} missing: run the forest scene once (it writes the baked impostors there)", path.display())
    })?);
    let mut m = [0u8; 8];
    f.read_exact(&mut m)?;
    if &m != MAGIC {
        bail!("{} is not an impostor cache", path.display());
    }
    let _key = read_str(&mut f)?;
    let n = read_u32(&mut f)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let name = read_str(&mut f)?;
        let species = read_str(&mut f)?;
        let kind = read_u32(&mut f)?;
        let height = f32::from_le_bytes(read_u32(&mut f)?.to_le_bytes());
        let mut ib = vec![0u8; std::mem::size_of::<ImpostorInfo>()];
        f.read_exact(&mut ib)?;
        let info: ImpostorInfo = bytemuck::pod_read_unaligned(&ib);
        let nj = read_u32(&mut f)? as usize;
        let mut jb = vec![0u8; nj * 16];
        f.read_exact(&mut jb)?;
        let jobs: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(&jb);
        let mut lb = [0u8; 8];
        f.read_exact(&mut lb)?;
        let mut blocks = vec![0u8; u64::from_le_bytes(lb) as usize];
        f.read_exact(&mut blocks)?;
        out.push(Variant { name, species, kind, height, info, jobs, blocks });
    }
    Ok(out)
}
