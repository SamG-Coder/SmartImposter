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
    vec4 cloud;     // x cloud cover (0..1), y volumetric clouds on (1) / off (0), z globe: smooth tree thinning (1) / stepped (0), w seconds of animation
    vec4 ocean0;    // globe sea (ocean.rs): x wind m/s, y residual slope variance, z largest wave height m, w chop
    vec4 ocean1;    //   xy wind direction (patch coordinates), z whitecap coverage, w 1 once a frame is in + enclosure 0..1
    vec4 ocean2;    //   slope variance of each cascade
    uvec4 biome;    // forest scene: ground slot -> scanned surface, one byte per slot (biome.rs)
    vec4 lawn;      // forest scene lawn (lawn.rs): x blade height m (0 = no lawn), y tile m, zw views (azimuths, slopes)
    vec4 lawn2;     //   x texels per view, y gutter texels, z atlas levels, w atlas slot
    vec4 lawn3;     //   xy slope (cot elevation) of the steepest and flattest views, z 1 = the park (--park: watered, no dried grass), w real blades around the camera to this distance (m)
    vec4 lawn4;     //   x segments of a blade, y auto mode: the short grass as a part of the baked height (0 = not auto), z the low end of the height patches, w the distance (m) by which the layer has faded into the ground texture
    vec4 acInv0;    // --test-flight, the aircraft: the renderer's frame -> its model frame (rows, w the translation)
    vec4 acInv1;
    vec4 acInv2;
    vec4 acPrev0;   //   a point on the aircraft this frame -> where it was last frame (rows): motion vectors
    vec4 acPrev1;
    vec4 acPrev2;
    vec4 acState;   //   x 1 = an aircraft is flying, y afterburner 0..1, z Mach number, w vapour cone 0..1
    vec4 acState2;  //   x vapour over the wings 0..1, y points in the wake, z air density / sea level's, w seconds flown
    vec4 acVel;     //   xyz which way the air goes past the aircraft (model frame, unit), w engine power 0..1
    vec4 wake[32];  //   where the aircraft has been, newest first (the renderer's frame); w age (s)
    vec4 wakeB[32]; //   x condensation trail left there 0..1, y how much the air was stirred there 0..1
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
