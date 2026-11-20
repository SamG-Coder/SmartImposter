// The smoke and fire of `--battle` (battle.rs): rocket trails, afterburner flames, tracers, flares,
// fireballs and the smoke they leave. Each is a tube of Gaussian section between two points (or
// a ball, where the two are the same), and each is an instance of one unit box in the scene's
// acceleration structure, with a mask bit of its own: only the ray that main.rgen sends to
// gather them meets them. fx.rint works out how much of a tube the ray passes through, fx.rahit
// adds that to the ray's payload and lets the ray go on to the next: no order is needed, the
// sums are the same whichever comes first.
#ifndef FX_GLSL
#define FX_GLSL

struct FxPrim {
    vec4 a; // one end (the renderer's frame), w its radius there (m)
    vec4 b; // the other end, its radius
    vec4 c; // rgb: a smoke's colour, a fire's light; w density (1 / m)
    vec4 d; // x 0 smoke, 1 fire; y a number of its own (0..1); z where along its trail a is (m); w how ragged (0..1)
};

struct FxPayload {
    vec3 emit;  // light added
    float od;   // optical depth of smoke
    vec3 scat;  // the smoke's colour, weighted by its depth
    float wsum; // weights, and distances weighted: where along the ray it all is
    float tsum;
};

#endif
