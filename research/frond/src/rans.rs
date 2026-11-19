//! The entropy coder: range asymmetric numeral system (rANS; Duda 2013) with adaptive models.
//!
//! * A symbol is coded by its slice [start, start + freq) of a 15-bit total (32768).
//! * The coder state is 32 bits and is renormalised 16 bits at a time: the compressed data is a
//!   sequence of little-endian u16 words after the 4-byte final state.
//! * rANS decodes symbols in the reverse of the order they were encoded in. The models adapt in
//!   the forward direction, so the encoder runs the models forward first, remembering each
//!   symbol's slice, and then codes the slices backwards (`Encoder::finish`). The decoder reads
//!   forward and adapts as it goes.
//! * A model is a cumulative distribution over 2..=64 symbols that moves towards the symbols it
//!   sees (the update dav1d / AV1 use): every boundary below or at the symbol moves down, every
//!   boundary above moves up, by 1 / 2^rate of the distance to its limit. A symbol never drops
//!   below a frequency of 1. A model learns fast at first and settles: its rate starts at 3 and
//!   rises by one every 8 symbols it has seen, up to the rate its use asks for.

pub const SCALE: u32 = 15;
pub const TOTAL: u32 = 1 << SCALE;
/// Lower bound of the normalised state interval.
const LOW: u32 = 1 << 16;

/// An adaptive distribution over M - 1 symbols: cdf[s]..cdf[s + 1] is symbol s's slice.
#[derive(Clone, Copy)]
pub struct Model<const M: usize> {
    cdf: [u16; M],
}

impl<const M: usize> Default for Model<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const M: usize> Model<M> {
    /// Every symbol equally likely.
    pub fn new() -> Self {
        let n = M - 1;
        let mut cdf = [0u16; M];
        for (i, c) in cdf.iter_mut().enumerate().take(n) {
            *c = (i as u32 * TOTAL / n as u32) as u16;
        }
        // (cdf[n] would be TOTAL, which does not fit 16 bits: the slot counts the symbols seen)
        Model { cdf }
    }

    #[inline]
    fn slice(&self, s: usize) -> (u32, u32) {
        let start = self.cdf[s] as u32;
        let end = if s + 2 == M { TOTAL } else { self.cdf[s + 1] as u32 };
        (start, end - start)
    }

    /// The symbol whose slice holds `v` (0 <= v < TOTAL).
    #[inline]
    fn find(&self, v: u32) -> usize {
        let n = M - 1;
        let mut s = 0;
        while s + 1 < n && self.cdf[s + 1] as u32 <= v {
            s += 1;
        }
        s
    }

    #[inline]
    fn update(&mut self, s: usize, rate: u32) {
        let n = M - 1;
        let seen = self.cdf[n] as u32;
        let rate = rate.min(3 + seen / 8);
        self.cdf[n] = (seen + 1).min(255) as u16;
        for i in 1..n {
            let c = self.cdf[i] as u32;
            self.cdf[i] = if i <= s {
                // towards its floor i (a frequency of 1 for every symbol below)
                c - ((c - i as u32) >> rate)
            } else {
                // towards its ceiling TOTAL - (n - i)
                c + ((TOTAL - (n - i) as u32 - c) >> rate)
            } as u16;
        }
    }
}

/// What the block coder is written against: the encoder passes the symbol through and records
/// it, the decoder ignores the argument and returns the symbol it reads. One body of modelling
/// code then serves both directions and cannot disagree with itself.
pub trait Coder {
    fn sym<const M: usize>(&mut self, m: &mut Model<M>, rate: u32, s: usize) -> usize;
    /// Accounting: the bits spent since the last call go to `what` (encoder with stats only).
    fn tally(&mut self, _what: usize) {}
}

#[derive(Default)]
pub struct Encoder {
    /// (start, freq) of every symbol, in coding order
    slices: Vec<(u16, u16)>,
    /// with `stats`: bits per category, and the bits not yet attributed
    pub stats: Option<Vec<f64>>,
    pending: f64,
}

impl Encoder {
    pub fn with_stats(categories: usize) -> Self {
        Encoder { slices: Vec::new(), stats: Some(vec![0.0; categories]), pending: 0.0 }
    }

    /// The compressed bytes: final state (u32 LE), then the words the decoder reads in order.
    pub fn finish(&self) -> Vec<u8> {
        let mut words: Vec<u16> = Vec::with_capacity(self.slices.len() / 3);
        let mut x: u32 = LOW;
        for &(start, freq) in self.slices.iter().rev() {
            let (start, freq) = (start as u32, freq as u32);
            // keep x * TOTAL / freq inside 32 bits: shift 16 bits out first
            if x >= ((LOW >> SCALE) << 16) * freq {
                words.push(x as u16);
                x >>= 16;
            }
            x = ((x / freq) << SCALE) + (x % freq) + start;
        }
        let mut out = Vec::with_capacity(4 + words.len() * 2);
        out.extend_from_slice(&x.to_le_bytes());
        for w in words.iter().rev() {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }
}

impl Coder for Encoder {
    #[inline]
    fn sym<const M: usize>(&mut self, m: &mut Model<M>, rate: u32, s: usize) -> usize {
        let (start, freq) = m.slice(s);
        self.slices.push((start as u16, freq as u16));
        if self.stats.is_some() {
            self.pending += (TOTAL as f64 / freq as f64).log2();
        }
        m.update(s, rate);
        s
    }
    fn tally(&mut self, what: usize) {
        if let Some(st) = &mut self.stats {
            st[what] += self.pending;
            self.pending = 0.0;
        }
    }
}

pub struct Decoder<'a> {
    x: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let x = if data.len() >= 4 { u32::from_le_bytes([data[0], data[1], data[2], data[3]]) } else { LOW };
        Decoder { x, data, pos: 4 }
    }
}

impl Coder for Decoder<'_> {
    #[inline]
    fn sym<const M: usize>(&mut self, m: &mut Model<M>, rate: u32, _s: usize) -> usize {
        let v = self.x & (TOTAL - 1);
        let s = m.find(v);
        let (start, freq) = m.slice(s);
        self.x = freq * (self.x >> SCALE) + v - start;
        if self.x < LOW {
            // (a truncated stream reads zeros: the output is wrong, but nothing is read out of bounds)
            let w = if self.pos + 2 <= self.data.len() { u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]) } else { 0 };
            self.pos += 2;
            self.x = (self.x << 16) | w as u32;
        }
        m.update(s, rate);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_round_trip_through_adapting_models() {
        // a skewed source over 8 symbols with two contexts, and a binary one
        let mut seed = 12345u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        let src: Vec<(usize, usize, usize)> = (0..50_000)
            .map(|_| {
                let ctx = (next() % 2) as usize;
                let r = next() % 100;
                let s = if ctx == 0 { (r / 40) as usize } else { 7 - (r / 30).min(7) as usize };
                (ctx, s, (next() % 16 == 0) as usize)
            })
            .collect();
        let mut enc = Encoder::default();
        let (mut m, mut b) = ([Model::<9>::new(); 2], Model::<3>::new());
        for &(ctx, s, f) in &src {
            enc.sym(&mut m[ctx], 5, s);
            enc.sym(&mut b, 5, f);
        }
        let bytes = enc.finish();
        // (the source has about 2.1 bits per pair; raw it takes 4)
        assert!(bytes.len() * 8 < src.len() * 5 / 2, "{} bytes", bytes.len());
        let mut dec = Decoder::new(&bytes);
        let (mut m, mut b) = ([Model::<9>::new(); 2], Model::<3>::new());
        for &(ctx, s, f) in &src {
            assert_eq!(dec.sym(&mut m[ctx], 5, 0), s);
            assert_eq!(dec.sym(&mut b, 5, 0), f);
        }
    }

    #[test]
    fn a_model_never_loses_a_symbol() {
        let mut m = Model::<17>::new();
        for _ in 0..10_000 {
            m.update(3, 4);
        }
        for s in 0..16 {
            assert!(m.slice(s).1 >= 1, "symbol {s}");
        }
        assert!(m.slice(3).1 > TOTAL - 64);
    }
}
