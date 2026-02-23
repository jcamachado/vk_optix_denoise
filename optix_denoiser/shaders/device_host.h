#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#ifdef __cplusplus
using mat4 = glm::mat4;
using vec4 = glm::vec4;
using vec3 = glm::vec3;
#endif // __cplusplus

#include "nvvkhl/shaders/dh_lighting.h"

struct PushConstant // total 32 bytes
{
  int frame;      // For RTX 4 bytes
  int maxDepth;   // For RTX 4 bytes
  int maxSamples; // For RTX 4 bytes
  int materialId; // For raster 4 bytes
  int instanceId; //  4 bytes
  int passId;    //  4 bytes
  float middleRadius; // optional override for reprojection radius (0 = auto) 4 bytes 
  float eyeSeparation; // 4 bytes
  float fovDegrees; // per eye horizontal fov in degrees 4 bytes
  int mode; // 0 = right dominant, 1 = left dominant (default) 4 bytes 
};

#define MAX_NB_LIGHTS 1
#define GRID_SIZE 16

// std140-friendly: use vec4 instead of vec3 to match GPU alignment
//struct FrameInfo // Total size 320 bytes (matches std140)
//{
//	mat4 proj;          // 64 bytes (offset 0)
//	mat4 proj2;         // 64 bytes (offset 64)
//	mat4 view;          // 64 bytes (offset 128)
//	mat4 view2;         // 64 bytes (offset 192)
//	vec4 clearColor;    // 16 bytes (offset 256)
//	vec4 camPos;        // 16 bytes (offset 272) - was vec3, padded to vec4 for std140
//	vec4 camPos2;       // 16 bytes (offset 288) - was vec3, padded to vec4 for std140
//	float envRotation;  // 4 bytes  (offset 304)
//	float maxLuminance; // 4 bytes  (offset 308)
//	float _pad0;        // 4 bytes  (offset 312) - pad to 16-byte alignment
//	float _pad1;        // 4 bytes  (offset 316)
//};

struct AreaLight // Total size 112 bytes
{
	vec3 position;   // Center of the rectangle
	vec3 u;          // Vector along one side (width)
	vec3 v;          // Vector along the other side (height)
	vec3 emission;   // RGB intensity (W/m^2)
	float area;      // Precomputed area
};


struct FrameInfo // Total size 688 byte, 16*43
{
	mat4 proj;          // 64 bytes (offset 0)
	mat4 proj2;         // 64 bytes (offset 64)
	mat4 view;          // 64 bytes (offset 128)
	mat4 view2;         // 64 bytes (offset 192)
	mat4 projInv;       // 64 bytes (offset 256) - NEW
	mat4 proj2Inv;      // 64 bytes (offset 320) - NEW
	mat4 viewInv;       // 64 bytes (offset 384) - NEW
	mat4 view2Inv;      // 64 bytes (offset 448) - NEW
	vec4 clearColor;    // 16 bytes (offset 512)
	vec4 camPos;        // 16 bytes (offset 528)
	vec4 camPos2;       // 16 bytes (offset 544)
	float envRotation;  // 4 bytes  (offset 560)
	float maxLuminance; // 4 bytes  (offset 564)
	float _pad0;        // 4 bytes  (offset 568) - pad to 16-byte alignment
	float _pad1;        // 4 bytes  (offset 572)
	//AreaLight areaLight;	   // 112 bytes (offset 568) = > total 680 bytes divided by 16
	//float _pad0;        // 4 bytes  (offset 680) - pad to 16-byte alignment
	//float _pad1;        // 4 bytes  (offset 684)
};


#endif // HOST_DEVICE_H
