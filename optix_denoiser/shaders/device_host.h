#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#ifdef __cplusplus
using mat4 = glm::mat4;
using vec4 = glm::vec4;
using vec3 = glm::vec3;
#endif // __cplusplus

#include "nvvkhl/shaders/dh_lighting.h"

struct PushConstant
{
  int frame;      // For RTX
  int maxDepth;   // For RTX
  int maxSamples; // For RTX
  int materialId; // For raster
  int instanceId;
};

#define MAX_NB_LIGHTS 1
#define GRID_SIZE 16

struct FrameInfo
{
  mat4 proj;
  mat4 proj2;
  mat4 view;
  mat4 view2;
  vec4 clearColor;
  vec3 camPos;
  vec3 camPos2;
  float envRotation;
  float maxLuminance;
};

#endif // HOST_DEVICE_H
