// Per-frame constants (uniform buffer), shared by the ray tracing stages and the TAA pass.
#ifndef GLOBALS_GLSL
#define GLOBALS_GLSL

struct Globals {
    vec4 camPos;    // xyz, w = lod distance
    vec4 camRight;  // xyz, w = tan(fov/2) * aspect
    vec4 camUp;     // xyz, w = tan(fov/2)
    vec4 camFwd;    // xyz, w = pixel spread angle (rad)
    vec4 sunDir;    // xyz, w = sun intensity
    vec4 params;    // x time, y unused, z exposure, w ao ray range
    uvec4 iparams;  // x frames the camera has been still, y debug mode, z frame counter, w ao rays
    vec4 lodParams; // x mesh<->impostor band width, y hi->lo impostor distance, z its band width, w variant count
    vec4 prevPos;   // previous frame camera (TAA reprojection), same layout as camPos..camFwd
    vec4 prevRight;
    vec4 prevUp;
    vec4 prevFwd;
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
