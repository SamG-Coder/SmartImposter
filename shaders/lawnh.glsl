// How much of its height the lawn stands at around a place (lawn.rs): patches a few metres across,
// and with --lawn auto the world's grass map (long where grass is left to grow, short elsewhere).
// One continuous field for the terrain's layer (tri.rchit squashes its blades by it) and the real
// blades near the camera (tri.rahit trims each to it; lawn.rs patch_height is the same on the
// CPU). Included after the globals and the ground textures.
float lawnStands(vec2 xz) {
    float t = mix(g.lawn4.z, 1.0, smoothstep(0.2, 0.8, 0.6 * vnoise(xz * 0.23 + 11.3) + 0.4 * vnoise(xz * 0.9)));
    if (g.lawn4.y > 0.0) t *= mix(g.lawn4.y, 1.0, textureLod(groundTex[MAX_GROUND_TEX - 3], (xz + NEAR_HALF) / (2.0 * NEAR_HALF), 0.0).r);
    return t;
}
