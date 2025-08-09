#pragma once

#include "helper_math.h"

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define MINIMUM_LBVH_KERNELCC
#endif

#include <stdint.h>
#include <intrin.h>

#define MORTON_MAX_VALUE_3D 0x1FFFFF

namespace minimum_lbvh
{
	inline float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax)
	{
		return (value - inputMin) * ((outputMax - outputMin) / (inputMax - inputMin)) + outputMin;
	}
	inline float compMin(float3 v)
	{
		return fminf(fminf(v.x, v.y), v.z);
	}
	inline float compMax(float3 v)
	{
		return fmaxf(fmaxf(v.x, v.y), v.z);
	}

	inline float2 slabs(float3 ro, float3 one_over_rd, float3 lower, float3 upper)
	{
		float3 t0 = (lower - ro) * one_over_rd;
		float3 t1 = (upper - ro) * one_over_rd;

		float3 tmin = fminf(t0, t1);
		float3 tmax = fmaxf(t0, t1);
		float region_min = compMax(tmin);
		float region_max = compMin(tmax);

		region_min = fmaxf(region_min, 0.0f);

		return { region_min, region_max };
	}

	struct Triangle
	{
		float3 vs[3];
	};

	struct AABB
	{
		float3 lower;
		float3 upper;

		void setEmpty()
		{
			lower = make_float3(+FLT_MAX);
			upper = make_float3(-FLT_MAX);
		}
		void extend(const float3& p)
		{
			lower = fminf(lower, p);
			upper = fmaxf(upper, p);
		}

		void extend(const AABB& b)
		{
			lower = fminf(lower, b.lower);
			upper = fmaxf(upper, b.upper);
		}
		float surface_area() const
		{
			float3 size = upper - lower;
			return (size.x * size.y + size.y * size.z + size.z * size.x) * 2.0f;
		}

		//uint64_t encodeMortonCode(float3 p) const
		//{
		//	uint32_t coord[3];
		//	for (int i = 0; i < 3; i++)
		//	{
		//		int v = (int)remap(p[i], lower[i], upper[i], 0, MORTON_MAX_VALUE_3D + 1);
		//		coord[i] = (uint32_t)clamp(v, 0, MORTON_MAX_VALUE_3D);
		//	}
		//}
	};

	struct NodeIndex
	{
		NodeIndex() :m_index(0), m_isLeaf(0) {}
		NodeIndex(uint32_t index, bool isLeaf) :m_index(index), m_isLeaf(isLeaf) {}
		uint32_t m_index : 31;
		uint32_t m_isLeaf : 1;
	};

	struct LBVHNode
	{
		NodeIndex parent;
		NodeIndex children[2];
	};

	// Return the number of consecutive high-order zero bits in a 32-bit integer
	inline int clz(uint32_t x)
	{
#if !defined( MINIMUM_LBVH_KERNELCC )
		unsigned long scan;
		if (_BitScanReverse(&scan, x) == 0)
		{
			return 32;
		}
		return 31 - scan;
#else
		return __clz(x);
#endif
	}
	inline int clz64(uint64_t x)
	{
#if !defined( MINIMUM_LBVH_KERNELCC )
		unsigned long scan;
		if (_BitScanReverse64(&scan, x) == 0)
		{
			return 64;
		}
		return 63 - scan;
#else
		return __clzll(x);
#endif
	}

	// Find the position of the least significant bit set to 1
	inline int ffs(uint32_t x) {
#if !defined( MINIMUM_LBVH_KERNELCC )
		unsigned long scan;
		if (_BitScanForward(&scan, x) == 0)
		{
			return 0;
		}
		return scan + 1;
#else
		return __ffs(x);
#endif
	}
	inline int ffs64(uint64_t x) {
#if !defined( MINIMUM_LBVH_KERNELCC )
		unsigned long scan;
		if (_BitScanForward64(&scan, x) == 0)
		{
			return 0;
		}
		return scan + 1;
#else
		return __ffsll(x);
#endif
	}

	inline int delta(uint32_t a, uint32_t b)
	{
		return 32 - clz(a ^ b);
	}
	inline int delta(uint64_t a, uint64_t b)
	{
		return 64 - clz64(a ^ b);
	}

	inline uint64_t encodeMortonCode_Naive(uint32_t x, uint32_t y, uint32_t z)
	{
		uint64_t code = 0;
		for (uint64_t i = 0; i < 64 / 3; ++i)
		{
			code |=
				((uint64_t)(x & (1u << i)) << (2 * i + 0)) |
				((uint64_t)(y & (1u << i)) << (2 * i + 1)) |
				((uint64_t)(z & (1u << i)) << (2 * i + 2));
		}
		return code;
	}

	inline uint32_t compact3(uint64_t m)
	{
		uint64_t x = m & 0x1249249249249249;
		x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
		x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
		x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
		x = (x ^ (x >> 16)) & 0x1f00000000ffff;
		x = (x ^ (x >> 32)) & 0x1fffff;
		return static_cast<uint32_t>(x);
	}

	inline uint64_t splat3(uint32_t a)
	{
		uint64_t x = a & 0x1fffff;
		x = (x | x << 32) & 0x1f00000000ffff;
		x = (x | x << 16) & 0x1f0000ff0000ff;
		x = (x | x << 8) & 0x100f00f00f00f00f;
		x = (x | x << 4) & 0x10c30c30c30c30c3;
		x = (x | x << 2) & 0x1249249249249249;
		return x;
	}

	inline void decodeMortonCode(uint64_t morton, uint32_t* x, uint32_t* y, uint32_t* z)
	{
		*x = compact3(morton);
		*y = compact3(morton >> 1);
		*z = compact3(morton >> 2);
	}
	inline uint64_t encodeMortonCode(uint32_t x, uint32_t y, uint32_t z)
	{
		uint64_t answer = 0;
		answer |= splat3(x) | splat3(y) << 1 | splat3(z) << 2;
		return answer;
	}
}
