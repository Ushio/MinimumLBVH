#include "minimum_lbvh.h"
#include "helper_math.h"
#include "camera.h"
#include "sobol.h"

using namespace minimum_lbvh;

constexpr float PI = 3.14159265358979323846f;

__device__ uint32_t packRGBA( float4 color )
{
    int4 i4 = make_int4(color * 255.0f + make_float4(0.5f));
    i4 = clamp(i4, 0, 255);
    return (i4.z << 16) | (i4.y << 8) | i4.x;
}

__device__ float3 sampleHemisphereCosWeighted(float xi_0, float xi_1)
{
    float phi = xi_0 * 2.0f * PI;
    float r = sqrtf(xi_1);

    // uniform in a circle
    float x = cosf(phi) * r;
    float z = sinf(phi) * r;

    // project to hemisphere
    float y = sqrtf(fmax(1.0f - r * r, 0.0f));
    return { x, y, z };
}
__device__ void GetOrthonormalBasis(float3 zaxis, float3* xaxis, float3* yaxis) {
    const float sign = copysignf(1.0f, zaxis.z);
    const float a = -1.0f / (sign + zaxis.z);
    const float b = zaxis.x * zaxis.y * a;
    *xaxis = { 1.0f + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x };
    *yaxis = { b, sign + zaxis.y * zaxis.y * a, -zaxis.y };
}

/*
 * PCG random number generator from
 * https://www.pcg-random.org/download.html#minimal-c-implementation
 */
struct PCG
{
    __device__ PCG(uint64_t seed, uint64_t sequence)
    {
        state = 0U;
        inc = (sequence << 1u) | 1u;

        uniform();
        state += seed;
        uniform();
    }

    __device__ uint32_t uniform()
    {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    __device__ float uniformf()
    {
        uint32_t bits = (uniform() >> 9) | 0x3f800000;
        float value;
        memcpy(&value, &bits, sizeof(float));
        return value - 1.0f;
    }

    uint64_t state;  // RNG state.  All values are possible.
    uint64_t inc;    // Controls which RNG sequence(stream) is selected. Must *always* be odd.
};

extern "C" __global__ void normal(uint32_t *pixels, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles )
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if (xi < imageSize.x && yi < imageSize.y)
    {
        int pixel = xi + yi * imageSize.x;

        float3 ro, rd;
        rayGenerator.shoot(&ro, &rd, (float)xi / imageSize.x, (float)yi / imageSize.y);

        Hit hit;
        intersect(&hit, internals, triangles, *rootNode, ro, rd, invRd(rd));
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
}

extern "C" __global__ void ao(uint32_t* pixels, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, int useSobol )
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if (imageSize.x <= xi  && imageSize.y <= yi)
    {
        return;
    }

    int pixel = xi + yi * imageSize.x;

    float3 ro, rd;
    rayGenerator.shoot(&ro, &rd, (float)xi / imageSize.x, (float)yi / imageSize.y);

    Hit hit;
    intersect(&hit, internals, triangles, *rootNode, ro, rd, invRd(rd));
    if (hit.t == MINIMUM_LBVH_FLT_MAX)
    {
        pixels[pixel] = packRGBA({ 0, 0, 0, 1 });
        return;
    }

    float3 n = normalize(hit.ng);
    if ( 0.0f < dot(n, rd))
    {
        n = -n;
    }

    float3 xaxis;
    float3 zaxis;
    GetOrthonormalBasis(n, &xaxis, &zaxis);

    float3 ao_origin = ro + rd * hit.t + n * 0.001f;

    PCG rng(hashPCG(pixel), 0);

    int nSample = 16;
    float ao = 0.0f;
    for (int i = 0; i < nSample; i++)
    {
        float2 random;
        if (useSobol)
        {
            sobol::scrambled_sobol_2d(&random.x, &random.y, i, xi, yi);
        }
        else
        {
            random = make_float2(rng.uniformf(), rng.uniformf());
        }

        float3 dirLocal = sampleHemisphereCosWeighted(random.x, random.y);
        float3 dir = xaxis * dirLocal.x + zaxis * dirLocal.z + n * dirLocal.y;

        float length = 1.0f;
        Hit hit;
        hit.t = length;
        intersect(&hit, internals, triangles, *rootNode, ao_origin, dir, invRd(dir), RAY_QUERY_ANY);
        if (hit.t != length)
        {
            ao += 1.0f / nSample;
        }
    }
    float brightness = powf(1.0f - ao, 1.0f / 2.2f);
    float3 color = { brightness, brightness , brightness };
    pixels[pixel] = packRGBA({ color.x, color.y, color.z, 1.0f });
}

extern "C" __global__ void pt(uint32_t* pixels, float4 *accumulators, int2 imageSize, RayGenerator rayGenerator, const NodeIndex* rootNode, const InternalNode* internals, const Triangle* triangles, int useSobol, int maxSPP)
{
    int xi = threadIdx.x + blockDim.x * blockIdx.x;
    int yi = threadIdx.y + blockDim.y * blockIdx.y;

    if (imageSize.x <= xi && imageSize.y <= yi)
    {
        return;
    }

    int pixel = xi + yi * imageSize.x;

    float4 acc = accumulators[pixel];
    int iteration = acc.w;

    PCG rng(hashPCG(hashPCG(pixel) + iteration), 0);

    float3 ro, rd;
    rayGenerator.shoot(&ro, &rd, (float)xi / imageSize.x, (float)yi / imageSize.y);

    float3 color = {};
    float3 throughput = { 1.0f, 1.0f, 1.0f };

    Hit hit;
    intersect(&hit, internals, triangles, *rootNode, ro, rd, invRd(rd));

    if (hit.t != MINIMUM_LBVH_FLT_MAX)
    for (int depth = 0; depth < 4; depth++)
    {
        float2 random;
        if (useSobol)
        {
            sobol::scrambled_sobol_2d(&random.x, &random.y, iteration, pixel, depth);
        }
        else
        {
            random = make_float2(rng.uniformf(), rng.uniformf());
        }

        float3 hit_p = ro + rd * hit.t;
        float3 hit_n = normalize(hit.ng);
        if (0.0f < dot(hit_n, rd))
        {
            hit_n = -hit_n;
        }

        float3 xaxis;
        float3 zaxis;
        GetOrthonormalBasis(hit_n, &xaxis, &zaxis);
        float3 dirLocal = sampleHemisphereCosWeighted(random.x, random.y);
        float3 next_ro = hit_p + hit_n * 0.0001f;
        float3 next_rd = xaxis * dirLocal.x + zaxis * dirLocal.z + hit_n * dirLocal.y;

        throughput *= make_float3(0.7f, 0.65f, 0.67f);

        hit = Hit();
        intersect(&hit, internals, triangles, *rootNode, next_ro, next_rd, invRd(next_rd));
        if (hit.t == MINIMUM_LBVH_FLT_MAX )
        {
            color += throughput * make_float3(1.0f);
            break;
        }

        ro = next_ro;
        rd = next_rd;
    }

    if (iteration < maxSPP)
    {
        acc += make_float4(color.x, color.y, color.z, 1);
        accumulators[pixel] = acc;
    }

    pixels[pixel] = packRGBA({ 
        powf(acc.x / acc.w, 1.0f / 2.2f), 
        powf(acc.y / acc.w, 1.0f / 2.2f), 
        powf(acc.z / acc.w, 1.0f / 2.2f), 
        1.0f });
}