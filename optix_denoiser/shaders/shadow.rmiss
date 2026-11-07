#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#include "payload.glsl"

layout(location = 1) rayPayloadInEXT ShadowPayload shadowPayload;

void main() {
    // If the ray hits the sky/nothing, the light is NOT occluded
    shadowPayload.isOccluded = false; 
}