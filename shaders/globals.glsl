// Per-frame constants (uniform buffer), shared by the ray tracing stages and the TAA pass.
#ifndef GLOBALS_GLSL
#define GLOBALS_GLSL

struct Globals {
    vec4 camPos;    // xyz, w = lod distance
    vec4 camRight;  // xyz, w = tan(fov/2) * aspect
    vec4 camUp;     // xyz, w = tan(fov/2)
    vec4 camFwd;    // xyz, w = pixel spread angle (rad)
    vec4 sunDir;    // xyz, w = sun intensity
    vec4 params;    // x profiling switch (SI_PROF_SKIP), y material textures loaded (renderer), z exposure, w ao ray range
    uvec4 iparams;  // x frames the camera has been still, y debug mode, z frame counter, w ao rays
    vec4 lodParams; // x mesh<->impostor band width, y hi->lo impostor distance, z its band width, w ground layers loaded (bit mask)
    vec4 prevPos;   // previous frame camera (TAA reprojection), same layout as camPos..camFwd
    vec4 prevRight;
    vec4 prevUp;
    vec4 prevFwd;
    vec4 extra;     // x aerial perspective strength (1 = clear day, 0 = off), y grown forest radius (m)
    vec4 planet;    // globe mode: xyz planet centre relative to the camera (camera frame), w = 1; else 0
    vec4 moon;      // globe mode: xyz direction to the moon (camera frame), w = angular radius (rad)
    vec4 sunTrue;   // globe mode: xyz direction to the sun; w = pre-exposure (eye adaptation, 1 by day; 1 outside globe mode)
    vec4 earthX;    // globe mode: rows of the camera frame -> Earth axes (ECEF) rotation (clouds.glsl)
    vec4 earthY;
    vec4 earthZ;
    vec4 cloud;     // globe mode: x cloud cover (0..1), y clouds on (1) / off (0), z smooth tree thinning (1) / stepped (0)
    vec4 ocean0;    // globe sea (ocean.rs): x wind m/s, y residual slope variance, z largest wave height m, w chop
    vec4 ocean1;    //   xy wind direction (patch coordinates), z whitecap coverage, w 1 once a frame is in
    vec4 ocean2;    //   slope variance of each cascade
    uvec4 biome;    // forest scene: ground slot -> scanned surface, one byte per slot (biome.rs)
};

// Sub-pixel jitter for TAA / DLSS: Halton(2,3), 32-frame cycle, in [0,1)^2 (mirrored in dlss.rs).
float halton(uint i, uint b) {
    float f = 1.0, r = 0.0;
    while (i > 0u) {
        f /= float(b);
        r += f * float(i % b);
        i /= b;
    }
    return r;
}
vec2 taaJitter(uint frame) { uint i = (frame % 32u) + 1u; return vec2(halton(i, 2u), halton(i, 3u)); }

#endif
