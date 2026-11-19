# Frond: a compression format for parallax impostor atlases

Research, kept apart from the renderer: its own crate, not part of the renderer's build, and the
renderer does not read Frond files. This directory holds the codec (`src/`), the format as a
specification ([FORMAT.md](FORMAT.md)) and, here, what was measured and why the format looks the
way it does.

**What it is.** A lossless codec for the impostor atlas cache files of SmartImposter
(`assets/cache/impostors/*.bc`): the BC3 / BC5 blocks the GPU samples, for one plant, 144 views,
four images, all mip levels. A Frond file decodes to exactly those bytes.

**Result.** All 207 atlases of the renderer (5.49 GB) take 650 MB: 11.8% of the raw size, and
31.8% smaller than zstd -19 (954 MB) or brotli -9 (963 MB). Every one decodes bit for bit.
It decodes at 40-110 MB/s on one thread and about 240 MB/s on sixteen, far slower than zstd
(2-3 GB/s). It does not change GPU memory: the GPU still needs the BC blocks.

```
cargo run --release -- enc   atlas.bc out.frond
cargo run --release -- dec   out.frond atlas.bc
cargo run --release -- stats atlas.bc                 # where the bits go
cargo run --release -- bench --baselines atlas.bc...  # sizes and speeds next to zstd / brotli / lz4
cargo run --release -- sizes ../../assets/cache/impostors
cargo test --release
```

## 1. What is in an atlas

An atlas file is a 36-byte header and a stream of 16-byte BC blocks (`src/bc.rs`,
[FORMAT.md](FORMAT.md) section 1): per mip level an albedo (BC3: occlusion + colour), a normal
(BC5) and a depth image (BC5: depth + coverage), then a half-resolution visibility image (BC3).
A block is two 8-byte halves, each two endpoints and sixteen small indices.

Measured on the cache (three examples; the rest look alike):

| | poplar (192x448 frames) | beech (224x352) | wild grass (128x128) |
|---|---|---|---|
| raw | 53.6 MB | 49.0 MB | 10.2 MB |
| blocks that are empty | 77.5% | 64.6% | 81.6% |
| blocks fully covered | 3.7% | 0.6% | 0.0% |
| blocks partly covered | 18.8% | 34.7% | 18.4% |
| texels without coverage inside non-empty blocks | 12.9% | 18.9% | 23.8% |
| zlib -9 | 21.2% | 32.8% | 17.0% |
| xz -6 | 19.6% | 30.6% | 15.5% |
| zstd -19 | 20.3% | 31.7% | 16.2% |
| "skip empty blocks, order-0 code the fields of the rest" | 19.5% | 31.0% | 15.8% |

Three things follow.

1. **Most of an atlas is nothing**, even after the frames were cropped to the plant. Every
   general compressor finds that, which is why they all land in the same place.
2. **They all land where a trivial model does** (last row). zlib, xz, zstd and brotli remove the
   empty blocks and little else: to them the rest is noise.
3. **The rest really is close to noise.** Almost every non-empty block is a silhouette block
   (foliage is all edges), and the 3-bit indices of its covered texels have 2.96-2.99 bits of
   entropy on their own. Indices are 80% of the bits of a block, so this is where a codec wins
   or does not.

## 2. What a general compressor cannot see

**The four images describe one silhouette.** Where coverage is zero, the other images hold no
data, only what the atlas encoder (`shaders/bcenc.comp`) made of the blank value the bake leaves
there, and that follows from the block's endpoints alone. Measured: predicting the index of an
uncovered texel from the coverage image and the block's endpoints is right for 100.00% of depth
and occlusion texels and 99.86% of normal texels (the rest are ties in the GPU's rounding).
13-24% of the texels of non-empty blocks cost almost nothing this way.

**The images are mip chains.** A texel of one level is the coverage-weighted mean of the four
texels under it (`shaders/mip.comp`). Coding coarse to fine, a texel is known up to what its
siblings do. What each context tells about a covered texel's index (bits per index, static
estimate, beech level 0; 3.00 is no information):

| context | depth | occlusion | normal x | coverage |
|---|---|---|---|---|
| none | 2.98 | 2.96 | 2.99 | 2.68 |
| left and upper neighbour | 2.66 | 2.89 | 2.91 | 2.24 |
| + upper-left neighbour | 2.64 | 2.89 | 2.90 | 2.21 |
| + the same texel of another channel | 2.63 | 2.87-2.88 | 2.89-2.90 | |
| the texel of the level above | 2.54 | 2.67 | 2.71 | 2.02 |
| level above + neighbours | 2.43 | 2.65 | 2.69 | 1.95 |
| **what the level above leaves for the siblings not yet coded** + neighbours | **2.29** | **2.43** | **2.56** | **1.74** |
| ... for the last of the four siblings alone | 2.14 | 1.94 | 2.34 | 1.36 |

The level above is worth more than the neighbours, and other channels are worth nothing: depth,
occlusion and normal of a leaf texel are unrelated at the scale of BC quantisation. The last row
is the sharpest tool found: once three siblings are decoded, the fourth is nearly determined.

**A block's range brackets the level above.** The four coarser texels over a block are means of
its texels, so the block's high endpoint lies at or above the highest of them and its low
endpoint at or below the lowest. Coding the endpoints as those two (small, one-sided) distances
instead of against a neighbouring block takes depth endpoints from 11.0 to 8.6 bits per block,
normal endpoints from 14.4 to 12.3 and albedo colour endpoints from 15.0 to 8.2. (Coverage
endpoints are the exception, 0 or 1 nearly everywhere: they stay with the neighbour.)

## 3. The format in one paragraph

Blocks are coded per level from the coarsest to the finest, a level in bands of one row of
frames that decode in parallel. A block is *empty* (no bits), *coded*, or *literal* (anything the
model does not hold, stored raw: the format is lossless for any input). A coded block is its
coverage half first, then depth, occlusion, the normals, visibility and the two colours, each as
endpoints and sixteen texels. Uncovered texels are a one-bit "as the encoder would have written"
flag; covered texels are symbols under a model chosen by the sibling position, what the level
above expects, and the left and upper neighbours. Everything goes through adaptive models into
a rANS coder (32-bit state, 15-bit probabilities). A reader that only wants the coarse levels
stops early. The full specification is [FORMAT.md](FORMAT.md).

## 4. How it got there

Sizes of the three example atlases together (112.8 MB raw; zstd -19: 28.0 MB):

| step | size | vs zstd -19 |
|---|---|---|
| empty / coded / literal classes, coverage first, neighbour contexts, endpoints against the neighbour block | 23.5 MB | -16% |
| levels coarse to fine; rank context from what the level above leaves for the remaining siblings; a chunk starts from the models the level above ended with | 21.5 MB | -23% |
| BC4 endpoints as distances from the level above's texels; context split by sibling position | 20.9 MB | -25% |
| models that learn fast and then settle (rate 3 rising to 6) | 20.5 MB | -27% |
| the colour blocks (endpoints and positions) from the level above as well | 19.8 MB | -29% |

Tried and dropped:

| idea | why not |
|---|---|
| the same texel's rank in another channel as context | 0.01-0.03 bits: the channels are unrelated at this scale |
| the neighbour block's endpoint distance as extra context for endpoints | no measurable gain |
| endpoints of coverage from the level above | worse (0.29 -> 0.51 MB on the beech): they are nearly constant, the neighbour predicts them |
| upper-left neighbour as a third neighbour context | 0.01-0.03 bits for 10x the contexts |
| faster or slower fixed adaptation rates (4, 5, 7) | all within 1% of each other; the settling rate beat every fixed one |

## 5. Results

Whole cache, sizes (`frond sizes`, every file round-trip checked):

| kind | files | raw | Frond | zstd -3 | zstd -19 | brotli -9 | Frond vs zstd -19 |
|---|---|---|---|---|---|---|---|
| trees | 69 | 3659 MB | 445 MB (12.2%) | 671 MB | 645 MB | 650 MB | -31.0% |
| shrubs | 53 | 878 MB | 99 MB (11.3%) | 155 MB | 148 MB | 150 MB | -33.0% |
| rocks, logs, cacti | 22 | 308 MB | 32 MB (10.4%) | 51 MB | 48 MB | 49 MB | -33.0% |
| ferns and small plants | 24 | 132 MB | 14 MB (10.4%) | 23 MB | 21 MB | 22 MB | -35.7% |
| grasses | 39 | 516 MB | 61 MB (11.7%) | 96 MB | 92 MB | 93 MB | -34.0% |
| **all** | **207** | **5494 MB** | **650 MB (11.8%)** | **995 MB (18.1%)** | **954 MB (17.4%)** | **963 MB (17.5%)** | **-31.8%** |

Against xz -6 on the three example atlases: 27% smaller on each.

Speed on eight atlases (323 MB raw; a 16-thread desktop CPU; MB of output per second; the
baselines' encode speeds were not measured):

| | size | decode, one thread | decode, 16 threads | encode, 16 threads |
|---|---|---|---|---|
| Frond | 48.2 MB (14.9%) | 42-108 MB/s (60 overall) | 110-370 MB/s (243 overall) | 234 MB/s |
| zstd -19 | 69.3 MB (21.4%) | 2150-2950 MB/s | | |
| zstd -3 | 72.1 MB (22.3%) | 2300-3300 MB/s | | |
| brotli -9 | 69.9 MB (21.6%) | 470-780 MB/s | | |
| lz4 | 79.5 MB (24.6%) | 4800-5850 MB/s | | |

Where Frond's bits go (beech, 11.0 MB):

| | share |
|---|---|
| indices of the five BC4 channels + visibility | 63.2% |
| BC4 endpoints | 21.1% |
| albedo and visibility colour indices | 10.5% |
| colour endpoints | 5.1% |
| block classes | 0.2% |

## 6. What this is good for, and what it is not

* **Good for:** shipping or archiving a baked cache (5.5 GB becomes 0.65 GB, a third less than
  the best general compressor), and encoding is as fast as decoding, so writing it at bake time
  costs little.
* **Not a free replacement for the runtime cache.** The renderer loads the forest scene's 1.8 GB
  of atlases in about a second from the raw cache; at the decode speed measured above, Frond
  would take about 7 s on sixteen threads (an estimate: the renderer does not read Frond). A
  streamed full atlas on the globe (50-125 MB) would take 0.2-0.35 s to decode. zstd -3 makes
  the cache 995 MB instead of Frond's 650 MB and decodes forty times faster.
* **Not a GPU memory saving.** The GPU samples BC blocks; Frond is how they sit on disk.
* **The floor is the data.** After everything above, a covered texel's index still costs 2.3 to
  2.6 of its 3 bits. The atlas encoder quantises noisy foliage to eight levels per block and
  what is left is mostly that noise; no lossless model removes it.

## 7. Not done

* **The renderer does not use it.** Using it would mean a Frond reader in `src/atlas.rs` (the
  loader thread decoding into the upload buffer, coarse levels for the base atlas) and a choice
  about the load-time cost above. Nothing in `src/` was changed for this research.
* **Views are not predicted from each other.** The 144 frames show the same surface, and the
  depth image says where each texel is in space, so one frame could in principle be predicted by
  reprojecting its neighbours. Nothing here measures how much that would give; it is the obvious
  next experiment and it would cost decode speed.
* **The decoder is not tuned.** One rANS state, a linear search per symbol, bounds-checked
  context lookups, at most twelve bands in parallel. Interleaved states and table lookups are
  known ways to go several times faster; none were tried.
* **Encoding for compressibility.** Choosing BC endpoints and indices at bake time so that they
  compress better (rate-distortion optimisation) would shrink the files further, but it changes
  the texels the GPU samples, which is outside "lossless".
* Float determinism of the two "as the encoder would have written" predictions was only tested
  on this machine (x86-64); see FORMAT.md section 8.
