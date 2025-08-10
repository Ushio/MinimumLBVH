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
	template <class T>
	inline T ss_max(T x, T y)
	{
		return (x < y) ? y : x;
	}
	template <class T>
	inline T ss_min(T x, T y)
	{
		return (y < x) ? y : x;
	}
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

	inline float2 slabs(float3 ro, float3 one_over_rd, float3 lower, float3 upper )
	{
		float3 t0 = (lower - ro) * one_over_rd;
		float3 t1 = (upper - ro) * one_over_rd;

		float3 tmin = fminf(t0, t1);
		float3 tmax = fmaxf(t0, t1);
		float region_min = compMax(tmin);
		float region_max = compMin(tmax) * 1.00000024f; // Robust BVH Ray Traversal- revised

		region_min = fmaxf(region_min, 0.0f);

		return { region_min, region_max };
	}

	// Return the number of consecutive high-order zero bits in a 32-bit integer
	inline int clz(uint32_t x)
	{
#if !defined( MINIMUM_LBVH_KERNELCC )
		return __lzcnt(x);
#else
		return __clz(x);
#endif
	}
	inline int clz64(uint64_t x)
	{
#if !defined( MINIMUM_LBVH_KERNELCC )
		return _lzcnt_u64(x);
#else
		return __clzll(x);
#endif
	}

	// Find the position of the least significant bit set to 1
	inline int ffs(uint32_t x) {
#if !defined( MINIMUM_LBVH_KERNELCC )
		if (x == 0)
		{
			return 0;
		}
		return _tzcnt_u32(x) + 1;
#else
		return __ffs(x);
#endif
	}
	inline int ffs64(uint64_t x) {
#if !defined( MINIMUM_LBVH_KERNELCC )
		if (x == 0)
		{
			return 0;
		}
		return _tzcnt_u64(x) + 1;
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

		uint64_t encodeMortonCode(float3 p) const
		{
			int3 coord = make_int3((p - lower) / (upper - lower) * (float)(MORTON_MAX_VALUE_3D + 1));
			coord = clamp(coord, 0, MORTON_MAX_VALUE_3D);
			return minimum_lbvh::encodeMortonCode(coord.x, coord.y, coord.z);
		}
	};

	struct NodeIndex
	{
		NodeIndex() :m_index(0), m_isLeaf(0) {}
		NodeIndex(uint32_t index, bool isLeaf) :m_index(index), m_isLeaf(isLeaf) {}
		uint32_t m_index : 31;
		uint32_t m_isLeaf : 1;
	};

	inline bool operator==(NodeIndex a, NodeIndex b)
	{
		uint32_t aBits;
		uint32_t bBits;
		memcpy(&aBits, &a, sizeof(uint32_t));
		memcpy(&bBits, &b, sizeof(uint32_t));
		return aBits == bBits;
	}

	struct InternalNode
	{
		NodeIndex parent;
		NodeIndex children[2];
		AABB aabbs[2];
	};
	struct Stat
	{
		uint32_t oneOfEdges;
	};

	inline bool intersect_ray_triangle(float* tOut, float* uOut, float* vOut, float3* ngOut, float t_min, float t_max, float3 ro, float3 rd, float3 v0, float3 v1, float3 v2 )
	{
		float3 e0 = v1 - v0;
		float3 e1 = v2 - v1;
		float3 ng = cross(e0, e1);

		float t = dot(v0 - ro, ng) / dot(ng, rd);
		if (t_min <= t && t <= t_max)
		{
			float3 e2 = v0 - v2;
			float3 p = ro + rd * t;

			float n2TriArea0 = dot(ng, cross(e0, p - v0));  // |n| * 2 * tri_area( p, v0, v1 )
			float n2TriArea1 = dot(ng, cross(e1, p - v1));  // |n| * 2 * tri_area( p, v1, v2 )
			float n2TriArea2 = dot(ng, cross(e2, p - v2));  // |n| * 2 * tri_area( p, v2, v0 )

			if (n2TriArea0 < 0.0f || n2TriArea1 < 0.0f || n2TriArea2 < 0.0f)
			{
				return false;
			}

			float n2TriArea = n2TriArea0 + n2TriArea1 + n2TriArea2;  // |n| * 2 * tri_area( v0, v1, v2 )

			// Barycentric Coordinates
			float bW = n2TriArea0 / n2TriArea;  // tri_area( p, v0, v1 ) / tri_area( v0, v1, v2 )
			float bU = n2TriArea1 / n2TriArea;  // tri_area( p, v1, v2 ) / tri_area( v0, v1, v2 )
			float bV = n2TriArea2 / n2TriArea;  // tri_area( p, v2, v0 ) / tri_area( v0, v1, v2 )

			*tOut = t;
			*uOut = bV;
			*vOut = bW;
			*ngOut = ng;
			return true;
		}

		return false;
	}
}
