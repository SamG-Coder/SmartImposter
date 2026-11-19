//! Frond: a lossless codec for SmartImposter's parallax impostor atlases. Research tool.
//!
//!   frond enc <atlas.bc> <out.frond>      compress a cache file
//!   frond dec <in.frond> <out.bc>         decompress it again (bit for bit the cache file)
//!   frond stats <atlas.bc>...             where the bits go
//!   frond bench [--baselines] [--threads N] <atlas.bc | dir>...
//!                                         sizes and speeds, round trip checked; with --baselines
//!                                         also zstd, brotli and lz4 on the same streams
//!   frond sizes <atlas.bc | dir>...       sizes only, many files at once: Frond, zstd -3 / -19
//!                                         and brotli -9 over a whole cache

mod bc;
mod codec;
mod rans;

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::time::Instant;

fn files(args: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for a in args {
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut inside: Vec<PathBuf> = std::fs::read_dir(&p).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "bc")).collect();
            inside.sort();
            out.extend(inside);
        } else {
            out.push(p);
        }
    }
    out
}

fn name(p: &std::path::Path) -> String {
    let n = p.file_stem().unwrap_or_default().to_string_lossy();
    // (drop the cache key)
    n.rsplit_once('-').map_or(n.to_string(), |(a, _)| a.to_string())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let threads_all = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    match args.first().map(|s| s.as_str()) {
        Some("enc") if args.len() == 3 => {
            let file = std::fs::read(&args[1]).with_context(|| format!("reading {}", args[1]))?;
            let a = bc::parse(&file)?;
            let out = codec::encode(a.key, &a.shape, a.stream, threads_all, None);
            std::fs::write(&args[2], &out)?;
            println!("{} -> {}: {} -> {} bytes ({:.1}%)", args[1], args[2], file.len(), out.len(), 100.0 * out.len() as f64 / file.len() as f64);
        }
        Some("dec") if args.len() == 3 => {
            let file = std::fs::read(&args[1]).with_context(|| format!("reading {}", args[1]))?;
            let (key, shape, stream) = codec::decode(&file, threads_all)?;
            std::fs::write(&args[2], bc::unparse(key, shape, &stream))?;
            println!("{} -> {}: {} -> {} bytes", args[1], args[2], file.len(), stream.len() + bc::SIIMP_HEADER);
        }
        Some("stats") if args.len() >= 2 => {
            for p in files(&args[1..]) {
                let file = std::fs::read(&p)?;
                let a = bc::parse(&file)?;
                let mut bits = vec![0.0; codec::CATEGORIES.len()];
                let out = codec::encode(a.key, &a.shape, a.stream, threads_all, Some(&mut bits));
                let s = a.shape;
                println!("{}: {}x{} frames of {}x{}, {} levels; {:.2} MB -> {:.2} MB ({:.1}%)", name(&p), s.grid, s.grid, s.rx, s.ry, s.mips, file.len() as f64 / 1e6, out.len() as f64 / 1e6, 100.0 * out.len() as f64 / file.len() as f64);
                let total: f64 = bits.iter().sum();
                for (c, b) in codec::CATEGORIES.iter().zip(&bits) {
                    println!("  {c:24} {:9.3} MB  {:5.1}%", b / 8e6, 100.0 * b / total);
                }
            }
        }
        Some("bench") if args.len() >= 2 => bench(&args[1..], threads_all)?,
        Some("sizes") if args.len() >= 2 => sizes(&args[1..], threads_all)?,
        _ => bail!("usage: frond enc <atlas.bc> <out.frond> | dec <in.frond> <out.bc> | stats <atlas.bc>... | bench [--baselines] [--threads N] <atlas.bc | dir>..."),
    }
    Ok(())
}

/// Sizes over many files, one file per thread: [raw, frond, zstd -3, zstd -19, brotli -9] in total
/// and per kind of plant (the first word of the name).
fn sizes(args: &[String], threads: usize) -> Result<()> {
    let list = files(args);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let rows: std::sync::Mutex<Vec<(String, [u64; 5])>> = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(p) = list.get(i) else { break };
                let file = std::fs::read(p).expect("reading");
                let a = bc::parse(&file).expect("an atlas");
                let f = codec::encode(a.key, &a.shape, a.stream, 1, None);
                let (_, _, back) = codec::decode(&f, 1).expect("decoding");
                assert!(back == a.stream, "{}: round trip differs", p.display());
                let z3 = zstd::bulk::compress(a.stream, 3).unwrap().len();
                let z19 = zstd::bulk::compress(a.stream, 19).unwrap().len();
                let mut br = Vec::new();
                let params = brotli::enc::BrotliEncoderParams { quality: 9, lgwin: 24, ..Default::default() };
                brotli::BrotliCompress(&mut &a.stream[..], &mut br, &params).unwrap();
                rows.lock().unwrap().push((name(p), [a.stream.len() as u64, f.len() as u64, z3 as u64, z19 as u64, br.len() as u64]));
            });
        }
    });
    let rows = rows.into_inner().unwrap();
    let mut groups: std::collections::BTreeMap<String, (usize, [u64; 5])> = Default::default();
    for (n, r) in &rows {
        for key in [n.split('_').next().unwrap_or("").to_string(), "all".to_string()] {
            let g = groups.entry(key).or_default();
            g.0 += 1;
            (0..5).for_each(|k| g.1[k] += r[k]);
        }
    }
    println!("{:10} {:>5} {:>10} {:>16} {:>16} {:>16} {:>16} {:>22}", "kind", "files", "raw MB", "frond", "zstd -3", "zstd -19", "brotli -9", "frond vs zstd -19");
    for (k, (n, t)) in &groups {
        let c = |i: usize| format!("{:8.1} ({:4.1}%)", t[i] as f64 / 1e6, 100.0 * t[i] as f64 / t[0] as f64);
        println!("{k:10} {n:5} {:10.1} {} {} {} {} {:>21.1}%", t[0] as f64 / 1e6, c(1), c(2), c(3), c(4), -100.0 * (1.0 - t[1] as f64 / t[3] as f64));
    }
    Ok(())
}

fn bench(args: &[String], threads_all: usize) -> Result<()> {
    let mut baselines = false;
    let mut threads = threads_all;
    let mut rest = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--baselines" => baselines = true,
            "--threads" => threads = it.next().and_then(|v| v.parse().ok()).context("--threads N")?,
            _ => rest.push(a.clone()),
        }
    }
    let list = files(&rest);
    println!("{} files, {threads} threads", list.len());
    let secs = |t: Instant| t.elapsed().as_secs_f64();
    // totals: raw, frond, then the baselines
    let mut tot = [0u64; 8];
    let mut time = [0.0f64; 4]; // encode, decode, decode on one thread, raw bytes timed
    let labels = ["zstd -3", "zstd -19", "brotli -9", "lz4", "zstd -19 --long", "brotli -11"];
    println!("{:34} {:>8} {:>8} {:>6} {:>9} {:>9} {:>9}", "atlas", "raw MB", "frond", "%", "enc MB/s", "dec MB/s", "dec 1 thr");
    for p in &list {
        let file = std::fs::read(p)?;
        let a = bc::parse(&file).with_context(|| p.display().to_string())?;
        let t = Instant::now();
        let out = codec::encode(a.key, &a.shape, a.stream, threads, None);
        let t_enc = secs(t);
        let t = Instant::now();
        let (key, shape, stream) = codec::decode(&out, threads)?;
        let t_dec = secs(t);
        if key != a.key || shape != a.shape || stream != a.stream {
            bail!("{}: round trip differs", p.display());
        }
        let t = Instant::now();
        let h = codec::header(&out)?;
        let mut again = vec![0u8; stream.len()];
        codec::decode_into(&out, &h, &mut again, 0, 1)?;
        let t_one = secs(t);
        let mb = a.stream.len() as f64 / 1e6;
        println!("{:34} {:8.2} {:8.2} {:6.1} {:9.0} {:9.0} {:9.0}", name(p), mb, out.len() as f64 / 1e6, 100.0 * out.len() as f64 / a.stream.len() as f64, mb / t_enc, mb / t_dec, mb / t_one);
        tot[0] += a.stream.len() as u64;
        tot[1] += out.len() as u64;
        time[0] += t_enc;
        time[1] += t_dec;
        time[2] += t_one;
        time[3] += mb;
        if baselines {
            let raw = a.stream;
            let mut row = String::new();
            let mut run = |k: usize, f: &dyn Fn() -> (usize, f64)| {
                let (n, dec_s) = f();
                tot[2 + k] += n as u64;
                row += &format!("  {} {:.2} MB ({:.1}%, decodes {:.0} MB/s)", labels[k], n as f64 / 1e6, 100.0 * n as f64 / raw.len() as f64, mb / dec_s);
            };
            let zstd_at = |level: i32, long: bool| {
                move || {
                    let mut e = zstd::bulk::Compressor::new(level).unwrap();
                    if long {
                        e.set_parameter(zstd::zstd_safe::CParameter::EnableLongDistanceMatching(true)).unwrap();
                        e.set_parameter(zstd::zstd_safe::CParameter::WindowLog(27)).unwrap();
                    }
                    let c = e.compress(raw).unwrap();
                    let t = Instant::now();
                    let mut d = zstd::bulk::Decompressor::new().unwrap();
                    d.set_parameter(zstd::zstd_safe::DParameter::WindowLogMax(27)).unwrap();
                    let back = d.decompress(&c, raw.len()).unwrap();
                    assert!(back == raw);
                    (c.len(), t.elapsed().as_secs_f64())
                }
            };
            let brotli_at = |q: u32| {
                move || {
                    let mut c = Vec::new();
                    let params = brotli::enc::BrotliEncoderParams { quality: q as i32, lgwin: 24, ..Default::default() };
                    brotli::BrotliCompress(&mut &raw[..], &mut c, &params).unwrap();
                    let t = Instant::now();
                    let mut back = Vec::with_capacity(raw.len());
                    brotli::BrotliDecompress(&mut &c[..], &mut back).unwrap();
                    assert!(back == raw);
                    (c.len(), t.elapsed().as_secs_f64())
                }
            };
            run(0, &zstd_at(3, false));
            run(1, &zstd_at(19, false));
            run(2, &brotli_at(9));
            run(3, &|| {
                let c = lz4_flex::compress(raw);
                let t = Instant::now();
                let back = lz4_flex::decompress(&c, raw.len()).unwrap();
                assert!(back == raw);
                (c.len(), t.elapsed().as_secs_f64())
            });
            run(4, &zstd_at(19, true));
            if std::env::var("FROND_BROTLI11").is_ok() {
                run(5, &brotli_at(11));
            }
            println!("{row}");
        }
    }
    let pct = |n: u64| 100.0 * n as f64 / tot[0].max(1) as f64;
    println!("\nall: {:.1} MB -> frond {:.1} MB ({:.1}%); encode {:.0} MB/s, decode {:.0} MB/s, {:.0} MB/s on one thread", tot[0] as f64 / 1e6, tot[1] as f64 / 1e6, pct(tot[1]), time[3] / time[0], time[3] / time[1], time[3] / time[2]);
    if baselines {
        for (k, l) in labels.iter().enumerate() {
            if tot[2 + k] > 0 {
                println!("     {l}: {:.1} MB ({:.1}%); frond is {:.1}% smaller", tot[2 + k] as f64 / 1e6, pct(tot[2 + k]), 100.0 * (1.0 - tot[1] as f64 / tot[2 + k] as f64));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{bc, codec};

    /// A small atlas of noise, empty blocks and well-formed blocks: every class of block.
    fn synthetic() -> (bc::Shape, Vec<u8>) {
        let shape = bc::Shape { grid: 2, rx: 16, ry: 32, mips: 2 };
        let mut s = vec![0u8; shape.stream_bytes()];
        let mut seed = 99u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 16) as u8
        };
        for (k, b) in s.chunks_exact_mut(16).enumerate() {
            match k % 5 {
                0 | 1 => {} // empty
                2 => b.iter_mut().for_each(|v| *v = next()), // noise: mostly literal
                _ => {
                    b.iter_mut().for_each(|v| *v = next());
                    // well-formed BC4 halves: high endpoint above low
                    for h in [0, 8] {
                        if b[h] <= b[h + 1] {
                            b[h] = b[h + 1].saturating_add(1);
                            b[h + 1] = b[h] - 1;
                        }
                    }
                }
            }
        }
        (shape, s)
    }

    #[test]
    fn every_block_class_round_trips() {
        let (shape, s) = synthetic();
        for threads in [1, 4] {
            let f = codec::encode(7, &shape, &s, threads, None);
            let (key, sh, back) = codec::decode(&f, threads).unwrap();
            assert_eq!((key, sh), (7, shape));
            assert!(back == s);
        }
    }

    #[test]
    fn the_coarse_levels_decode_alone() {
        let (shape, s) = synthetic();
        let f = codec::encode(7, &shape, &s, 2, None);
        let h = codec::header(&f).unwrap();
        let mut out = vec![0xAAu8; s.len()];
        codec::decode_into(&f, &h, &mut out, 1, 2).unwrap();
        // level 1's images and the visibility image are there; level 0's bytes were not touched
        let l1 = shape.region(0, 1).offset;
        assert!(out[l1..] == s[l1..]);
        assert!(out[..l1].iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn a_damaged_file_is_an_error_not_a_crash() {
        let (shape, s) = synthetic();
        let mut f = codec::encode(7, &shape, &s, 1, None);
        let n = f.len();
        for k in (120..n).step_by(7) {
            f[k] ^= 0x5A;
        }
        assert!(codec::decode(&f, 1).is_err());
        assert!(codec::decode(&f[..n / 2], 1).is_err());
    }
}
