#pragma once

#include "helper_math.h"

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define MINIMUM_LBVH_KERNELCC
#endif

#if defined(MINIMUM_LBVH_KERNELCC)
#else
#include <stdint.h>
#include <intrin.h>
#include <Windows.h>
#endif

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

		static NodeIndex invalid()
		{
			return NodeIndex(0x7FFFFFFF, false);
		}
	};

	inline bool operator==(NodeIndex a, NodeIndex b)
	{
		return a.m_index == b.m_index && a.m_isLeaf == b.m_isLeaf;
	}
	inline bool operator!=(NodeIndex a, NodeIndex b)
	{
		return !(a == b);
	}

	struct InternalNode
	{
		NodeIndex parent;
		NodeIndex children[2];
		AABB aabbs[2];
		uint32_t oneOfEdges; // wasteful but for simplicity
	};

	inline void build_lbvh(
		NodeIndex* rootNode,
		InternalNode* internals,
		const Triangle* triangles,
		int nTriangles,
		const uint32_t* sortedTriangleIndices,
		const uint8_t* deltas,
		const AABB& sceneAABB,
		int i_leaf)
	{
		int nInternals = nTriangles - 1;
		uint32_t leaf_lower = i_leaf;
		uint32_t leaf_upper = i_leaf;

		uint32_t triangleIndex = sortedTriangleIndices[i_leaf];
		NodeIndex node(triangleIndex, true);

		AABB aabb; aabb.setEmpty();
		for (auto v : triangles[triangleIndex].vs)
		{
			aabb.extend(v);
		}

		bool isRoot = true;
		while (leaf_upper - leaf_lower < nInternals )
		{
			// direction from bottom
			int goLeft;
			if (leaf_lower == 0)
			{
				goLeft = 0;
			}
			else if (leaf_upper == nInternals )
			{
				goLeft = 1;
			}
			else
			{
				goLeft = deltas[leaf_lower - 1] < deltas[leaf_upper] ? 1 : 0;
			}

			int parent = goLeft ? (leaf_lower - 1) : leaf_upper;

			internals[parent].children[goLeft] = node;
			internals[parent].aabbs[goLeft] = aabb;
			if (!node.m_isLeaf)
			{
				internals[node.m_index].parent = minimum_lbvh::NodeIndex(parent, false);
			}

			uint32_t index = goLeft ? leaf_upper : leaf_lower;

			// == memory barrier ==

			index = InterlockedExchange( &internals[parent].oneOfEdges, index );

			// == memory barrier ==

			if (index == 0xFFFFFFFF)
			{
				isRoot = false;
				break;
			}

			leaf_lower = ss_min(leaf_lower, index);
			leaf_upper = ss_max(leaf_upper, index);

			node = NodeIndex(parent, false);

			AABB otherAABB = internals[parent].aabbs[goLeft ^ 0x1];
			aabb.extend(otherAABB);
		}

		if (isRoot)
		{
			*rootNode = node;
			internals[node.m_index].parent = minimum_lbvh::NodeIndex::invalid();
		}
	}

	inline bool intersect_ray_triangle(float* tOut, float* uOut, float* vOut, float3* ngOut, float t_min, float t_max, float3 ro, float3 rd, float3 v0, float3 v1, float3 v2 )
	{
		float3 e0 = v1 - v0;
		float3 e1 = v2 - v1;
		float3 ng = cross(e0, e1);

		float t = dot(v0 - ro, ng) / dot(ng, rd);
		if (t_min <= t && t <= t_max)
		{
			float3 e2 = v0 - v2;

			// Use tetrahedron volumes in space of 'ro' as the origin. note constant scale will be ignored.
			//   P0 = v0 - ro, P1 = v1 - ro, P2 = v2 - ro
			//   u_vol * 6 = ( P0 x P2 ) . rd
			//   v_vol * 6 = ( P1 x P0 ) . rd
			//   w_vol * 6 = ( P2 x P1 ) . rd
			// The cross product is unstable when ro is far away.. 
			// So let's use '2 ( a x b ) = (a - b) x (a + b)'
			//   u_vol * 12 = ( ( P0 - P2 ) x ( P0 + P2 ) ) . rd = ( ( P0 - P2 ) x ( v0 + v2 - ro * 2 ) ) . rd
			//   v_vol * 12 = ( ( P1 - P0 ) x ( P1 + P0 ) ) . rd = ( ( P1 - P0 ) x ( v1 + v0 - ro * 2 ) ) . rd
			//   w_vol * 12 = ( ( P2 - P1 ) x ( P2 + P1 ) ) . rd = ( ( P2 - P1 ) x ( v2 + v1 - ro * 2 ) ) . rd
			// As u, v, w volume are consistent on the neighbor, it is edge watertight.
			// Reference: https://github.com/RenderKit/embree/blob/v4.4.0/kernels/geometry/triangle_intersector_pluecker.h#L79-L94
			float u_vol = dot(cross(e2, v0 + v2 - ro * 2.0f), rd);
			float v_vol = dot(cross(e0, v1 + v0 - ro * 2.0f), rd);
			float w_vol = dot(cross(e1, v2 + v1 - ro * 2.0f), rd);

			if (u_vol < 0.0f || v_vol < 0.0f || w_vol < 0.0f)
			{
				return false;
			}

			float vol = u_vol + v_vol + w_vol;

			// Barycentric Coordinates
			// hit = w*v0 + u*v1 + v*v2
			//     = v0 + u*(v1 - v0) + v*(v2 - v0)
			// float bW = w_vol / vol;
			float bU = u_vol / vol;
			float bV = v_vol / vol;

			*tOut = t;
			*uOut = bU;
			*vOut = bV;
			*ngOut = ng;
			return true;
		}

		return false;
	}
}
