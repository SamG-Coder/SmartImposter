// Globe (1:1 Earth) shared declarations: globals, payload, atmosphere.
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#include "rt.glsl"

struct GlobeGlobals {
    vec4 camRight;
    vec4 camUp;
    vec4 camFwd;  // w = tan(fov_y / 2)
    vec4 sun;     // xyz direction to the sun, w = illuminance scale
    vec4 planet;  // xyz planet centre relative to the camera, w = radius
    vec4 params;  // x exposure, y frame, z debug view, w ground scans present (bit mask)
    vec4 misc;    // x pixel angle (rad); trees (forest.rs): y every tree within, z SPLIT x finest tile edge, w range (horizon)
    uvec4 addr;   // xy shared index buffer address, z smooth tree thinning (1) or stepped (0)
    // the city nearest the camera (globe/city.rs shader_city): its plane's origin relative to the
    // camera, its east and north axes; x block pitch east-west, y north-south, z radius (0 =
    // none), w number of cores; the cores: x, y in the plane, strength, radius
    vec4 cityO;
    vec4 cityE;
    vec4 cityN;
    vec4 cityP;
    vec4 cityCores[8];
};


#include "atmosphere.glsl"

vec3 tonemap(vec3 c) {
    // ACES fit (Narkowicz)
    c = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
    return pow(c, vec3(1.0 / 2.2));
}
