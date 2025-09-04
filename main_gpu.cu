#include "minimum_lbvh.h"
#include "helper_math.h"
#include "camera.h"
using namespace minimum_lbvh;

__device__ uint32_t packRGBA( float4 color )
{
    int4 i4 = make_int4(color * 255.0f + make_float4(0.5f));
    i4 = clamp(i4, 0, 255);
    return (i4.z << 16) | (i4.y << 8) | i4.x;
}

extern "C" __global__ void render(uint32_t *pixels, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, NodeIndex* stackBuffer )
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    NodeIndex* stack = allocStack(stackBuffer);

    if (xi < imageSize.x && yi < imageSize.y)
    {
        int pixel = xi + yi * imageSize.x;

        float3 ro, rd;
        rayGenerator.shoot(&ro, &rd, (float)xi / imageSize.x, (float)yi / imageSize.y);

        Hit hit;
        intersect(&hit, internals, triangles, *rootNode, ro, rd, invRd(rd));
        // intersect_stackfull(&hit, internals, triangles, *rootNode, ro, rd, invRd(rd), stack);
        if (hit.t != MINIMUM_LBVH_FLT_MAX)
        {
            float3 n = normalize(hit.ng);
            float3 color = (n + make_float3(1.0f)) * 0.5f;
            pixels[pixel] = packRGBA({ color.x, color.y, color.z, 1.0f });
        }
        else
        {
            pixels[pixel] = packRGBA({ 0, 0, 0, 1 });
        }
    }

    freeStack(stack);
}