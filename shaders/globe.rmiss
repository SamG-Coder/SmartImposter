#version 460
#extension GL_GOOGLE_include_directive : require
#include "globe.glsl"

layout(location = 0) rayPayloadInEXT GlobePayload pl;

void main() { pl.t = -1.0; }
