#include "minimum_lbvh.h"
#include "helper_math.h"
#include "camera.h"
using namespace minimum_lbvh;

extern "C" __global__ void render(float4 *pixels, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles )
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if (imageSize.x <= xi || imageSize.y < yi)
    {
        return;
    }

    int pixel = xi + yi * imageSize.x;

    float3 ro, rd;
    rayGenerator.shoot(&ro, &rd, (float)xi / imageSize.x, (float)yi / imageSize.y);

    Hit hit;
    intersect(&hit, internals, triangles, *rootNode, ro, rd, invRd(rd));
    if (hit.t != MINIMUM_LBVH_FLT_MAX)
    {
        float3 n = normalize(hit.ng);
        //if (smooth)
        //{
        //    TriangleAttrib attrib = triangleAttribs[hit.triangleIndex];
        //    n = attrib.shadingNormals[0] +
        //        (attrib.shadingNormals[1] - attrib.shadingNormals[0]) * hit.uv.x +
        //        (attrib.shadingNormals[2] - attrib.shadingNormals[0]) * hit.uv.y;
        //    n = normalize(n);
        //}
        float3 color = (n + make_float3(1.0f)) * 0.5f;
        pixels[pixel] = { color.x, color.y, color.z, 1.0f };
    }
    else
    {
        pixels[pixel] = { 0, 0, 0, 1 };
    }
}