/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.glsl"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/func.h"
#include "nvvkhl/shaders/dh_hdr.h"

layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 2, binding = eHdr) uniform sampler2D hdrTexture;


void main()
{// When point light is active, env contribution is disabled — no MIS needed
  if (frameInfo.pointLightColorEnabled.w > 0.5)
  {
    payload.contrib = vec3(0.0);
    payload.hitT    = INFINITE;
    return;
  }



  // Adding HDR lookup
  // Apply per-axis rotation: envRotation is a vec3 in degrees, convert to radians
  vec3 rot = radians(frameInfo.envRotation);
  vec3 dir = gl_WorldRayDirectionEXT;
  // Miss rotates the incoming ray by the inverse of the environment rotation
  //vec3 dir        = rotate(gl_WorldRayDirectionEXT, vec3(0, 1, 0), -frameInfo.envRotation);
  dir = rotate(dir, vec3(1.0, 0.0, 0.0), -rot.x);
  dir = rotate(dir, vec3(0.0, 1.0, 0.0), -rot.y);
  dir = rotate(dir, vec3(0.0, 0.0, 1.0), -rot.z);
  vec2 uv         = getSphericalUv(dir);  // See sampling.glsl
  vec3 env        = texture(hdrTexture, uv).rgb;
  
  // --- MIS: BSDF-side weight (power heuristic β=2) ---
  // env sampling PDF is stored in hdrTexture.w (precomputed by NVVK HdrEnv)
  // payload.bsdfPdf = 0 → primary ray or first hit: no competing NEE → weight = 1
  float mis_weight = 1.0;
  float envPdf     = texture(hdrTexture, uv).w;
  if (payload.bsdfPdf > 0.0 && envPdf > 1e-10)
  {
    float a2    = payload.bsdfPdf * payload.bsdfPdf;
    float b2    = envPdf * envPdf;
    mis_weight  = a2 / (a2 + b2 + 1e-10);
    // Complement of the NEE weight applied in rchit: powerHeuristic(lightPdf, bsdfPdf)
    // Together they partition energy without double-counting or energy loss
  }
  
  //payload.contrib = env * frameInfo.clearColor.xyz * frameInfo.envIntensity;
  //payload.contrib = env * frameInfo.clearColor.xyz;
  // NOTE: BSDF-side MIS weight intentionally disabled.
  // The env importance sampling PDF from hdrTexture.w is in sr^-1 (can be >> 1000 for
  // bright spots) while bsdfPdf is ~0.1-0.3, causing powerHeuristic to return ~0 and
  // making everything black. The NEE power heuristic in pathtrace.rchit already reduces
  // double-counting on the light-sampling side without suppressing indirect paths.
  // TODO: implement proper BSDF-side MIS using EnvAccel + an explicit environmentPdf()
  //       function that normalises to the same domain as sampleData.pdf.
  payload.contrib = env * frameInfo.clearColor.xyz * frameInfo.envIntensity;
  payload.hitT    = INFINITE;
}
