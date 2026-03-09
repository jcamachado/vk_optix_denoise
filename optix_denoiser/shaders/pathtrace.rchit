/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.glsl"

#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/ggx.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/dh_hdr.h"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/random.h"
#include "nvvkhl/shaders/bsdf_functions.h"
#include "nvvkhl/shaders/vertex_accessor.h"


hitAttributeEXT vec2 attribs;

#include "get_hit.glsl"

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(buffer_reference, scalar) readonly buffer Materials { GltfShadeMaterial m[]; };


layout(set = 0, binding = eTlas ) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures
layout(set = 2, binding = eImpSamples,  scalar)	buffer _EnvAccel { EnvAccel envSamplingData[]; };
layout(set = 2, binding = eHdr) uniform sampler2D hdrTexture;

layout(push_constant) uniform RtxPushConstant_ { PushConstant pc; };
// clang-format on

// Includes depending on layout description
#include "nvvkhl/shaders/pbr_mat_eval.h"  // texturesMap
#include "nvvkhl/shaders/hdr_env_sampling.h"


void stopPath()
{
  payload.hitT = INFINITE;
}

struct ShadingResult
{
  vec3 weight;
  vec3 radiance;
  vec3 rayOrigin;
  vec3 rayDirection;
};
vec2 dirToEnvUV(vec3 d)
{
  float phi = atan(d.z, d.x); // -PI..PI
  float u = (phi + M_PI) * (0.5 * M_1_PI); // (phi + PI) / (2*PI)
  float theta = acos(clamp(d.y, -1.0, 1.0)); // 0..PI
  float v = theta * M_1_OVER_PI; // theta / PI
  return vec2(u, v);
}
float envPdfDir(vec3 dir)
{
  vec2 uv = dirToEnvUV(dir);
  vec4 val = texture(hdrTexture, uv); // requires .w = pdf per solid angle OR modify to use alias table
  return val.w;
}

vec3 envRadianceDir(vec3 dir, out float pdf)
{
  vec2 uv = dirToEnvUV(dir);
  vec4 val = texture(hdrTexture, uv);
  pdf = val.w;
  return val.xyz;
}

// --------------------------------------------------------------------
// Sampling the Sun or the HDR
// - Returns
//      The contribution divided by PDF
//      The direction to the light source
//      The PDF
//
  // helper: Rodrigues rotate a vector 'v' around axis 'a' by angle 'ang'
vec3 rodrigues_rotate(vec3 v, vec3 a, float ang)
{
    float c = cos(ang);
    float s = sin(ang);
    return v * c + cross(a, v) * s + a * dot(a, v) * (1.0 - c);
}



vec3 sampleLights(in HitState state, inout uint seed, out vec3 dirToLight, out float lightPdf)
{
  // Fast-out: HDR is disabled, nothing to sample.
  if (frameInfo.pointLightColorEnabled.w > 0.5)
  {
    dirToLight = vec3(0.0);
    lightPdf   = 0.0;
    return vec3(0.0);
  }
  vec3 rand_val     = vec3(rand(seed), rand(seed), rand(seed));
  vec4 radiance_pdf = environmentSample(hdrTexture, rand_val, dirToLight);
  vec3 radiance     = radiance_pdf.xyz;
  lightPdf          = radiance_pdf.w;

  // Apply rotation and environment intensity
  //dirToLight = rotate(dirToLight, vec3(0, 1, 0), frameInfo.envRotation);
  //radiance *= frameInfo.clearColor.xyz * frameInfo.envIntensity;
  vec3 rotDeg = frameInfo.envRotation;
  vec3 rot = radians(rotDeg); // convert to radians


  // Apply rotations: X (pitch), Y (yaw), Z (roll)
  dirToLight = rodrigues_rotate(dirToLight, vec3(1.0, 0.0, 0.0), rot.x);
  dirToLight = rodrigues_rotate(dirToLight, vec3(0.0, 1.0, 0.0), rot.y);
  dirToLight = rodrigues_rotate(dirToLight, vec3(0.0, 0.0, 1.0), rot.z);
  radiance *= frameInfo.clearColor.xyz * frameInfo.envIntensity;
  // radiance *= frameInfo.clearColor.xyz;

  // Return radiance over pdf
  return radiance / lightPdf;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
ShadingResult shading(in PbrMaterial pbrMat, in HitState hit)
{
  ShadingResult result;

  vec3 to_eye = -gl_WorldRayDirectionEXT;

  result.radiance = pbrMat.emissive;  // Emissive material


  // Light contribution; can be environment or punctual lights
  vec3  contribution         = vec3(0);
  vec3  dirToLight           = vec3(0);
  float lightPdf             = 0.F;
  vec3  lightRadianceOverPdf = sampleLights(hit, payload.seed, dirToLight, lightPdf);

// --- Point light next-event estimation (delta light, MIS weight = 1) ---
  if (frameInfo.pointLightColorEnabled.w > 0.5)
  {
    vec3  toLightPL = vec3(frameInfo.pointLightPos) - hit.pos;
    float dist2     = dot(toLightPL, toLightPL);
    if (dist2 > 1e-6)
    {
      float dist    = sqrt(dist2);
      vec3  dirToPL = toLightPL / dist;

      if (dot(dirToPL, hit.geonrm) > 0.0)
      {
        BsdfEvaluateData evalPL;
        evalPL.k1 = -gl_WorldRayDirectionEXT;
        evalPL.k2 = dirToPL;
        evalPL.xi = vec3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
        bsdfEvaluate(evalPL, pbrMat);

        // Shadow ray: use offsetRay from the hit point — result.rayOrigin is not yet set here.
        vec3 shadowOrigin = offsetRay(hit.pos, hit.geonrm);
        uint ray_flag = gl_RayFlagsTerminateOnFirstHitEXT
                      | gl_RayFlagsSkipClosestHitShaderEXT
                      | gl_RayFlagsCullBackFacingTrianglesEXT;
        payload.hitT = 0.0;
        traceRayEXT(topLevelAS, ray_flag, 0xFF, 0, 0, 0, shadowOrigin, 0.001, dirToPL, dist - 0.001, 0);
        bool visible = (payload.hitT == INFINITE);
        payload.hitT = gl_HitTEXT;

        if (visible)
        {
          const float inv4pi = 0.07957747154;
          vec3 radiancePL = vec3(frameInfo.pointLightColorEnabled) * (inv4pi / dist2) * frameInfo.envIntensity;

          // Write directly to result.radiance — point light has its own shadow ray,
          // it must NOT depend on nextEventValid (which is false when HDR is off).
          result.radiance += radiancePL * evalPL.bsdf_diffuse;
          result.radiance += radiancePL * evalPL.bsdf_glossy;
        }
      }
    }
  }

  const bool nextEventValid = (dot(dirToLight, hit.geonrm) > 0.0f) && lightPdf != 0.0f;

  // Evaluate BSDF
  if(nextEventValid)
  {
    BsdfEvaluateData evalData;
    evalData.k1   = -gl_WorldRayDirectionEXT;
    evalData.k2   = dirToLight;
    evalData.xi = vec3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
    bsdfEvaluate(evalData, pbrMat);

    if(evalData.pdf > 0.0)
    {
      const float mis_weight = lightPdf / (lightPdf + evalData.pdf);

      // sample weight
      const vec3 w = lightRadianceOverPdf * mis_weight;
      contribution += w * evalData.bsdf_diffuse;
      contribution += w * evalData.bsdf_glossy;
    }
  }

  // Sample BSDF
  {
    BsdfSampleData sampleData;
    sampleData.k1   = -gl_WorldRayDirectionEXT;  // outgoing direction
    sampleData.xi   = vec3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
    bsdfSample(sampleData, pbrMat);

    if(sampleData.event_type == BSDF_EVENT_ABSORB)
    {
      stopPath();
      return result;  // Need to add the contribution ?
    }

    result.weight       = sampleData.bsdf_over_pdf;
    result.rayDirection = sampleData.k2;
    vec3 offsetDir      = dot(result.rayDirection, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
    result.rayOrigin    = offsetRay(hit.pos, offsetDir);
  }
  /*
  if(nextEventValid)
  {
    // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
    uint ray_flag = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
    payload.hitT = 0.0F;
    traceRayEXT(topLevelAS, ray_flag, 0xFF, 0, 0, 0, result.rayOrigin, 0.001, dirToLight, INFINITE, 0);
    // If hitting nothing, add light contribution
    if(payload.hitT == INFINITE)
      result.radiance += contribution;
    payload.hitT = gl_HitTEXT;
  }*/
  // (This preserves all existing logic; it just skips the shadow trace when frameInfo.envThroughWalls == 1.)
  if(nextEventValid)
  {
    // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
    uint ray_flag = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
    payload.hitT = 0.0F;
    traceRayEXT(topLevelAS, ray_flag, 0xFF, 0, 0, 0, result.rayOrigin, 0.001, dirToLight, INFINITE, 0);
    // If hitting nothing, add light contribution
    if(payload.hitT == INFINITE)
      result.radiance += contribution;
    payload.hitT = gl_HitTEXT;
  }

  return result;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  RenderNode renderNode = RenderNodeBuf(sceneDesc.renderNodeAddress)._[gl_InstanceID];
  RenderPrimitive rprim = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[gl_InstanceCustomIndexEXT];

  HitState hit = getHitState(rprim);

  // Scene materials
  uint      matIndex  = max(0, renderNode.materialID);  // material of primitive mesh
  Materials materials = Materials(sceneDesc.materialAddress);

  // Material of the object and evaluated material (includes textures)
  GltfShadeMaterial mat    = materials.m[matIndex];
  MeshState         mesh   = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, vec2[2](hit.uv, hit.uv), false/*isInside*/);
  PbrMaterial       pbrMat = evaluateMaterial(mat, mesh);

  payload.hitT         = gl_HitTEXT;
  ShadingResult result = shading(pbrMat, hit);

  payload.weight       = result.weight;
  payload.contrib      = result.radiance;
  payload.rayOrigin    = result.rayOrigin;
  payload.rayDirection = result.rayDirection;

  // -- Debug --
  //  payload.contrib = hit.nrm * .5 + .5;
  //  payload.contrib = matEval.albedo.xyz;
  //  payload.contrib = mat.pbrBaseColorFactor.xyz;
  //  payload.contrib = matEval.tangent * .5 + .5;
  //  payload.contrib = vec3(matEval.metallic);
  //  payload.contrib = vec3(matEval.roughness);
  //  stopRay();
}
