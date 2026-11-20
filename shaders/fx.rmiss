#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require
#include "fx.glsl"

// The ray that gathers the smoke and fire of --battle always ends here (fx.rahit stops at
// nothing): its sums are already in the payload.
layout(location = 2) rayPayloadInEXT FxPayload fp;

void main() {}
