# Frond, version 1: the format

Frond is a lossless container for one impostor atlas of SmartImposter: the block-compressed
(BC3 / BC5) albedo, normal, depth and visibility images of a plant seen from a grid of
directions, with all their mip levels. A Frond file decodes to exactly the bytes of the
renderer's cache file (`assets/cache/impostors/*.bc`), which are the bytes the GPU samples.

This document is the specification: what is in a file, in which order, and how every symbol is
modelled. `src/codec.rs`, `src/rans.rs` and `src/bc.rs` are the reference implementation; where
the two disagree, the code is right and this document has a bug.

All integers are little-endian. "Level" is a mip level; level 0 is the finest.

## 1. What is being stored

An atlas has a shape: `grid` (frames per side, 12), `rx` x `ry` (texels per frame) and `mips`
(levels). The uncompressed stream is a sequence of 16-byte BC blocks:

```text
for level in 0..mips:   albedo, normal, depth     each (grid rx >> level) x (grid ry >> level) texels
for level in 0..mips-1: visibility                half resolution: the size of the others' level + 1
```

Blocks of an image are in raster order. Each 16-byte block is two 8-byte halves:

| image      | bytes 0..8                      | bytes 8..16                                  |
|------------|---------------------------------|----------------------------------------------|
| albedo     | BC4 **occlusion**               | BC1 **albedo colour**                        |
| normal     | BC4 **normal x**                | BC4 **normal y**                             |
| depth      | BC4 **depth**                   | BC4 **coverage**                             |
| visibility | BC4 **visibility**              | BC1 **visibility colour**                    |

So a block position of a level holds six BC4 *channels* and two BC1 *colours* (the visibility
image of level L - 1 has the size of the other images' level L and is coded with them; level 0
has no visibility image at its size).

A BC4 half is `hi, lo` (two bytes) and sixteen 3-bit indices. The atlas encoder only writes the
8-value mode (`hi > lo`). Frond works with the *rank* of an index, 0 for the value `lo` up to 7
for `hi`:

```text
index: 0 1 2 3 4 5 6 7        value(rank) = lo + ((hi - lo) rank + 3) / 7      (integer division)
rank:  7 0 6 5 4 3 2 1
```

A BC1 half is two RGB565 endpoints `c0, c1` and sixteen 2-bit indices. Frond works with the
*position* along the endpoint line, 0 at `c0` to 3 at `c1`, and with the endpoints as
components (r: 5 bits, g: 6, b: 5):

```text
index:    0 1 2 3             colour(position)[k] = (c0[k] (3 - position) + c1[k] position + 1) / 3
position: 0 3 1 2
```

## 2. File layout

```text
offset  size  field
0       8     magic "FROND\0\0\x01"
8       8     key        (the renderer's cache key of the bake; carried through)
16      4     grid
20      4     rx
24      4     ry
28      4     mips
32      8     checksum of the uncompressed stream (section 7)
40      64    the four images' empty blocks, 16 bytes each (albedo, normal, depth, visibility)
104     4     chunk count n
108     4 n   chunk lengths in bytes
...           the chunks, in order
```

### Chunks

A **chunk** is a run of whole block rows of one level, with all the images of that level: albedo,
normal, depth, and (for level >= 1) the visibility image of level - 1.

The chunk list follows from the shape alone and is not stored. Levels go from the coarsest
(`mips - 1`) to the finest (0). A level is one chunk per row of frames (`grid` chunks, each
`(ry >> level) / 4` block rows) if such a chunk has at least 4096 blocks, and otherwise a single
chunk.

A chunk is:

```text
4     length m of the literal blocks
m     literal blocks (section 3)
...   the rANS stream (section 5)
```

### What depends on what

* A chunk needs the complete level above it (level + 1) and nothing else of its own level.
* So the chunks of one level can be decoded in parallel, and a reader that only wants levels
  >= L (the renderer's base atlas) reads the file up to the last chunk of level L and stops.
* The models (section 6) a chunk starts from are the models as they were at the end of the
  *last* chunk of the level above; the coarsest level starts from the initial models.

## 3. Blocks

Blocks of a chunk are coded in raster order. Every block position has a **class**:

| class | meaning |
|---|---|
| 0 empty   | all its images' blocks equal the header's empty blocks |
| 1 coded   | every BC4 channel has `hi > lo`: the block is coded as fields (section 4) |
| 2 literal | anything else: the blocks are stored as they are in the chunk's literal bytes, 16 bytes per image, in image order |

The class is a 3-symbol model chosen by the classes of the left and the upper block
(`3 left + up`; a missing neighbour counts as empty).

The empty blocks in the header are whatever the atlas encoder writes where there is no plant
(the encoder takes the most common block among each image's first 4096 at level 0).

For prediction (section 4), every texel of a level has a *coverage flag*, a *value* per BC4
channel and a *colour* per BC1 colour. Texels of empty and literal blocks have flag 0, value 0
and colour 0.

## 4. A coded block

The fields are coded in this order: the six BC4 channels **coverage, depth, occlusion, normal x,
normal y, visibility** (visibility only when the level has it), then the two BC1 colours
**albedo, visibility colour**. Within a channel: endpoints, then the sixteen texels in raster
order.

Coverage comes first because everything else is conditioned on it. After the coverage channel is
decoded, a texel is **covered** unless its rank is 0 and the block's `lo` is 0 (its coverage value
is 0). Its **weight** is its coverage value (at least 1), or 0 if it is not covered.

`above` is the level above; texel (x, y) of this level lies under texel (x / 2, y / 2) of it.
The 2 x 2 texels under one texel of the level above are **siblings**; they always share a block.

### 4.1 BC4 endpoints

For every channel but coverage, if there is a level above and at least one of the four texels
of it over this block is covered, let `mn`, `mx` be the lowest and highest of their values, and
`bucket` be 0 if they are equal and `1 + floor(log2(mx - mn))` otherwise. Then two bytes are
coded, each as a high nibble (model by `bucket`) and a low nibble (model by the high nibble):

```text
hi = mx + first byte          (mod 256)
lo = mn - second byte         (mod 256)
```

The texels above are means of the block's texels, so `hi` is at or above the highest and `lo` at
or below the lowest of them, but for rounding: both bytes are small non-negative numbers.

Otherwise (coverage, the coarsest level, or nothing above), with `p` the endpoints of the left
block if it is coded, else of the upper block if that is coded:

```text
hi = p.hi + first byte                      (mod 256; first byte's model by: none / left / up)
lo = hi - 1 - second byte                   (model by 0, or 1 + floor(log2(p.hi - p.lo)))
```

### 4.2 BC4 texels

**A texel that is not covered** (any channel but coverage) holds what the atlas encoder made of
the blank value its bake leaves there: depth 0, occlusion 1, normal 0.5, visibility 1. The
encoder's rank for a value `v` is

```text
rank = floor(clamp((v - lo / 255) / (hi / 255 - lo / 255), 0, 1) * 7 + 0.5)      (32-bit floats)
```

One binary symbol says whether the texel's rank is that; if not, the rank follows (8-symbol
model). In the atlases this is a hit 99.9% of the time.

**A covered texel** (and every texel of the coverage channel) is an 8-symbol rank whose model is
chosen by four things, `((q 10 + a) 10 + l) 10 + u`:

* `q`, 0..3: which of the four siblings it is (x & 1, + 2 if y & 1);
* `l`, `u`: the state of the texel to its left and above it (also across block borders, within
  the chunk): 8 if there is none, 9 if it is not covered (never for the coverage channel), else
  where its value falls on *this* block's scale of ranks,
  `rank_of(v) = 0 if v <= lo, else min(7, ((v - lo) 7 + (hi - lo) / 2) / (hi - lo))`;
* `a`: what the level above expects. 8 if there is no level above, 9 if the texel above is not
  covered (never for coverage), else `rank_of(clamp(rest / left, 0, 255))`, where

  ```text
  rest = value above * (sum of the four siblings' weights) - sum over the siblings already coded of weight * value
  left = sum of the weights of this texel and the siblings not yet coded      (at least 1)
  ```

  with weight 1 for all four in the coverage channel. The texel above is the weighted mean of
  its four siblings, so `rest / left` is what the texels not yet coded must average to: a loose
  guess for the first sibling and a sharp one for the last.

### 4.3 BC1 endpoints

Per component k (r, g, b), with `mask` 31 or 63. If there is a level above with a covered texel
over this block, `mn[k]`, `mx[k]` are the lowest and highest component of their colours and
`bucket` as in 4.1:

```text
c0[k] = mx[k] + first symbol      (mod mask + 1; model by bucket)
c1[k] = mn[k] - second symbol     (mod mask + 1; model by bucket)
```

Otherwise, with `p` the endpoint 0 of the left or upper coded block (else 0):

```text
c0[k] = p[k] + first symbol       (model by: none / left / up)
c1[k] = c0[k] - second symbol
```

### 4.4 BC1 texels

A texel that is not covered: one binary symbol says whether its position is the atlas encoder's
for the blank colour (albedo: black; visibility colour: mid grey), which is the nearest of the
four palette colours `c0, c1, (2 c0 + c1) / 3, (c0 + 2 c1) / 3` (as floats, components / 31 and
/ 63), the first of equals. If not, the position follows.

A covered texel is a 4-symbol position, model `((q 6 + a) 6 + l) 6 + u`: `q` as above; `l`, `u`
the position of the left and upper texel (4: none, 5: not covered); `a` 4 if there is no level
above, 5 if the texel above is not covered, else the position nearest to the colour the level
above expects (`rest / left` per component as in 4.2, weights from coverage), measured along the
endpoint line with red and blue counted double:

```text
dot  = sum over k of (v[k] - c0[k]) (c1[k] - c0[k]) s[k]^2          s = (2, 1, 2)
len2 = sum over k of ((c1[k] - c0[k]) s[k])^2
a    = 0 if len2 = 0, else clamp((6 dot + len2) / (2 len2), 0, 3)    (integer division towards zero)
```

## 5. The entropy coder

Symbols are coded with rANS (range asymmetric numeral systems): 32-bit state, 15-bit
probabilities (total 32768), renormalised 16 bits at a time. A chunk's rANS stream is the final
encoder state (4 bytes) followed by 16-bit words in the order the decoder reads them.

Decoding a symbol from a model with cumulative frequencies `cdf`:

```text
v = x & 32767
s = the symbol with cdf[s] <= v < cdf[s + 1]
x = (cdf[s + 1] - cdf[s]) * (x >> 15) + v - cdf[s]
if x < 65536:  x = (x << 16) | next word
```

rANS decodes in the reverse of the encoding order, and the models adapt in decoding order. The
encoder therefore runs the models forward over the chunk, recording each symbol's
(start, frequency), and then pushes those through the coder backwards.

## 6. Models

A model over n symbols is n - 1 movable boundaries `cdf[1..n]` (`cdf[0]` is 0, `cdf[n]` 32768),
starting evenly spaced: `cdf[i] = i * 32768 / n`. After coding symbol s, every boundary moves:

```text
for i in 1..n:
    if i <= s:  cdf[i] -= (cdf[i] - i) >> rate
    else:       cdf[i] += (32768 - (n - i) - cdf[i]) >> rate
```

`rate` is `min(limit, 3 + seen / 8)`, where `seen` counts the symbols the model has coded
(stopping at 255) and `limit` is 5 for the class models and 6 for everything else: a model
learns fast and then settles. No symbol's frequency falls below 1.

The models of a chunk (count x symbols):

| models | count | symbols |
|---|---|---|
| class | 9 | 3 |
| BC4 endpoints, from above: high nibble | 6 channels x 2 endpoints x 9 buckets | 16 |
| BC4 endpoints, from above: low nibble | 6 x 2 x 16 | 16 |
| BC4 `hi` against a neighbour: high / low nibble | 6 x 3, 6 x 16 | 16 |
| BC4 `hi - lo - 1`: high / low nibble | 6 x 9, 6 x 16 | 16 |
| BC4 rank of a covered texel | 6 x 4000 | 8 |
| BC4 blank hit; rank on a miss | 6; 6 | 2; 8 |
| BC1 endpoints, from above | 2 colours x 2 endpoints x (r, b: 7 buckets; g: 8) | 32; 64 |
| BC1 endpoint 0 against a neighbour | 2 x 3 per component | 32; 64 |
| BC1 endpoint 1 against endpoint 0 | 2 per component | 32; 64 |
| BC1 position of a covered texel | 2 x 864 | 4 |
| BC1 blank hit; position on a miss | 2; 2 | 2; 4 |

## 7. Checksum

Over the uncompressed stream, 8 bytes at a time as 64-bit words `w`:

```text
h = 0x9E3779B97F4A7C15 ^ length
for each w:              h = (h ^ w) * 0x9E3779B97F4A7C15;  h ^= h >> 32
for each trailing byte:  h = (h ^ byte) * 0x100000001B3
```

A decoder must reject a file whose decoded stream does not hash to the stored value.

## 8. Notes for implementers

* Everything in sections 4 to 6 is integer arithmetic except the two "what the atlas encoder
  would have written" predictions, which use IEEE 754 single precision with no fused
  multiply-add. A decoder on another platform must keep them bit-exact, or code all blank texels
  through the miss path.
* The format is lossless for *any* BC3 / BC5 block data of an atlas's shape: whatever does not
  fit the model becomes a literal block. Only the compression ratio depends on the data being an
  impostor atlas.
* A damaged file cannot make a conforming decoder read or write out of bounds: every index is
  derived from the shape, and the rANS reader returns zeros past the end. It decodes to wrong
  bytes and fails the checksum.
