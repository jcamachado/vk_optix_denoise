#ifndef PAYLOAD_H
#define PAYLOAD_H

precision highp float;

#define MISS_DEPTH 1000


// (NEE = Next Event Estimation, MIS = Multiple Importance Sampling)
// Pdf = Probability Density Function
struct HitPayload
{
  uint  seed;
  float hitT;
  float bsdfPdf; // 0 = no competing NEE -> MIS weight = 1
  vec3  contrib;
  vec3  weight;
  vec3  rayOrigin;
  vec3  rayDirection;
};
HitPayload initPayload()
{
  HitPayload p;
  p.seed         = 0U;
  p.hitT         = 0.F;
  p.bsdfPdf      = 0.F;
  p.contrib      = vec3(0.F);
  p.weight       = vec3(1.F);
  p.rayOrigin    = vec3(0.F);
  p.rayDirection = vec3(0.F, 0.F, -1.F);
  return p;
}

// #OPTIX_D
struct GbufferPayload
{
  uint packAlbedo;
  uint packNormal;
  uint packDepth;
};

#endif  // PAYLOAD_H