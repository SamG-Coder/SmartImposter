#version 460
#extension GL_GOOGLE_include_directive : require
#include "rt.glsl"

layout(location = 0) rayPayloadInEXT HitPayload pl;

void main() { pl.t = -1.0; }
