#version 460
#extension GL_GOOGLE_include_directive : require
#include "rt.glsl"

layout(location = 1) rayPayloadInEXT uint vis;

void main() { vis = 1u; }
