#pragma once

#include "helper_math.h"

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define MINIMUM_LBVH_KERNELCC
#endif

#if defined(MINIMUM_LBVH_KERNELCC)

#define MINIMUM_LBVH_ASSERT(ExpectTrue) ((void)0)

#else
#include <stdint.h>
#include <intrin.h>
#include <vector>
#include <Windows.h>
#include <ppl.h>
#define MINIMUM_LBVH_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { abort(); }

#if defined( ENABLE_EMBREE_BUILDER )
#include <embree4/rtcore.h>
#endif

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
		int i_leaf)
	{
		int nInternals = nTriangles - 1;
		int nDeltas = nInternals;
		uint32_t leaf_lower = i_leaf;
		uint32_t leaf_upper = i_leaf;

		uint32_t triangleIndex = sortedTriangleIndices[i_leaf];
		NodeIndex node(triangleIndex, true);

		AABB aabb; aabb.setEmpty();
		if (triangles)
		{
			for (auto v : triangles[triangleIndex].vs)
			{
				aabb.extend(v);
			}
		}

		bool isRoot = true;
		while (leaf_upper - leaf_lower < nInternals )
		{
			// direction from bottom
			uint32_t deltaL = 0 < leaf_lower ? deltas[leaf_lower - 1] : 0xFFFFFFFF;
			uint32_t deltaR = leaf_upper < nDeltas ? deltas[leaf_upper] : 0xFFFFFFFF;
			int goLeft = deltaL < deltaR ? 1 : 0;

			int parent = goLeft ? (leaf_lower - 1) : leaf_upper;

			internals[parent].children[goLeft] = node;
			internals[parent].aabbs[goLeft] = aabb;
			if (!node.m_isLeaf)
			{
				internals[node.m_index].parent = NodeIndex(parent, false);
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
			internals[node.m_index].parent = NodeIndex::invalid();
		}
	}

	inline bool intersectRayTriangle(float* tOut, float* uOut, float* vOut, float3* ngOut, float t_min, float t_max, float3 ro, float3 rd, float3 v0, float3 v1, float3 v2 )
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

			// +,- mixed then no hits
			if (fminf(fminf(u_vol, v_vol), w_vol) < 0.0f && 0.0f < fmaxf(fmaxf(u_vol, v_vol), w_vol))
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

	inline void validate_lbvh( NodeIndex node, const InternalNode* internals, const uint8_t* deltas, int maxDelta )
	{
		if (node.m_isLeaf)
		{
			return;
		}

		auto delta = deltas[node.m_index];
		MINIMUM_LBVH_ASSERT(delta <= maxDelta);
		validate_lbvh(internals[node.m_index].children[0], internals, deltas, delta);
		validate_lbvh(internals[node.m_index].children[1], internals, deltas, delta);
	}
#if defined( ENABLE_EMBREE_BUILDER )
	struct EmbreeBVHContext
	{
		EmbreeBVHContext() {
			nodes = 0;
			nodeHead = 0;
		}
		InternalNode* nodes;
		std::atomic<int> nodeHead;
	};

	inline void* node2ptr(NodeIndex node)
	{
		uint32_t data;
		memcpy(&data, &node, sizeof(uint32_t));
		return (char*)0 + data;
	}
	inline NodeIndex ptr2node(void* ptr)
	{
		uint32_t data = (char*)ptr - (char*)0;
		NodeIndex node;
		memcpy(&node, &data, sizeof(uint32_t));
		return node;
	}

	static void* embrreeCreateNode(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr)
	{
		MINIMUM_LBVH_ASSERT(numChildren == 2);

		EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
		int index = context->nodeHead++;

		NodeIndex node(index, false);
		return node2ptr(node);
	}
	static void embreeSetNodeChildren(void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr)
	{
		MINIMUM_LBVH_ASSERT(numChildren == 2);

		EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
		NodeIndex theParentIndex = ptr2node(nodePtr);
		InternalNode& node = context->nodes[theParentIndex.m_index];
		for (int i = 0; i < numChildren; i++)
		{
			NodeIndex childIndex = ptr2node(childPtr[i]);
			node.children[i] = childIndex;

			// add a link child to parent
			if (childIndex.m_isLeaf == 0)
			{
				context->nodes[childIndex.m_index].parent = theParentIndex;
			}
		}
	}
	static void embreeSetNodeBounds(void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr)
	{
		MINIMUM_LBVH_ASSERT(numChildren == 2);

		EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
		InternalNode& node = context->nodes[ptr2node(nodePtr).m_index];

		for (int i = 0; i < numChildren; i++)
		{
			node.aabbs[i].lower = make_float3(bounds[i]->lower_x, bounds[i]->lower_y, bounds[i]->lower_z);
			node.aabbs[i].upper = make_float3(bounds[i]->upper_x, bounds[i]->upper_y, bounds[i]->upper_z);
		}
	}
	static void* embreeCreateLeaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr)
	{
		MINIMUM_LBVH_ASSERT(numPrims == 1);
		NodeIndex node(prims->primID, true /*is leaf*/);
		return node2ptr(node);
	}
#endif

	class BVHCPUBuilder
	{
	public:
		void build(const Triangle *triangles, int nTriangles, bool isParallel )
		{
			m_internals.clear();
			m_internals.resize(nTriangles - 1);
			for (int i = 0; i < m_internals.size(); i++)
			{
				m_internals[i].oneOfEdges = 0xFFFFFFFF;
			}
			m_deltas.resize(nTriangles - 1);

			// Scene AABB
			AABB sceneAABB;
			sceneAABB.setEmpty();

			for (int i = 0 ; i < nTriangles ; i++)
			{
				for (int j = 0; j < 3; ++j)
				{
					sceneAABB.extend(triangles[i].vs[j]);
				}
			}

			std::vector<uint64_t> mortons(nTriangles);
			std::vector<uint32_t> sortedTriangleIndices(nTriangles);

			for (int i = 0; i < nTriangles; i++)
			{
				Triangle tri = triangles[i];
				float3 center = (tri.vs[0] + tri.vs[1] + tri.vs[2]) / 3.0f;
				mortons[i] = sceneAABB.encodeMortonCode(center);
				sortedTriangleIndices[i] = i;
			}

			std::sort(sortedTriangleIndices.begin(), sortedTriangleIndices.end(), [&mortons](uint32_t a, uint32_t b) {
				return mortons[a] < mortons[b];
			});

			for (int i = 0; i < m_deltas.size(); i++)
			{
				uint64_t mA = mortons[sortedTriangleIndices[i]];
				uint64_t mB = mortons[sortedTriangleIndices[i + 1]];
				m_deltas[i] = delta(mA, mB);
			}

			if (isParallel)
			{
				concurrency::parallel_for(size_t(0), mortons.size(), [&](uint32_t i_leaf) {
					build_lbvh(
						&m_rootNode,
						m_internals.data(),
						triangles,
						nTriangles,
						sortedTriangleIndices.data(),
						m_deltas.data(),
						i_leaf
					);
				});
			}
			else
			{
				for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
				{
					build_lbvh(
						&m_rootNode,
						m_internals.data(),
						triangles,
						nTriangles,
						sortedTriangleIndices.data(),
						m_deltas.data(),
						i_leaf
					);
				}
			}
		}
#if defined( ENABLE_EMBREE_BUILDER )
		void buildByEmbree(const Triangle* triangles, int nTriangles, RTCBuildQuality buildQuality)
		{
			RTCDevice device = rtcNewDevice("");
			RTCBVH bvh = rtcNewBVH(device);

			rtcSetDeviceErrorFunction(device, [](void* userPtr, RTCError code, const char* str) {
				printf("Embree Error [%d] %s\n", code, str);
			}, 0);

			std::vector<RTCBuildPrimitive> primitives(nTriangles);
			for (int i = 0; i < nTriangles; i++)
			{
				AABB aabb; aabb.setEmpty();
				for (auto v : triangles[i].vs)
				{
					aabb.extend(v);
				}
				RTCBuildPrimitive prim = {};
				prim.lower_x = aabb.lower.x;
				prim.lower_y = aabb.lower.y;
				prim.lower_z = aabb.lower.z;
				prim.geomID = 0;
				prim.upper_x = aabb.upper.x;
				prim.upper_y = aabb.upper.y;
				prim.upper_z = aabb.upper.z;
				prim.primID = i;
				primitives[i] = prim;
			}

			// allocation
			m_internals.clear();
			m_internals.resize(nTriangles  - 1);

			EmbreeBVHContext context;
			context.nodes = m_internals.data();

			RTCBuildArguments arguments = rtcDefaultBuildArguments();
			arguments.maxDepth = 64;
			arguments.byteSize = sizeof(arguments);
			arguments.buildQuality = buildQuality;
			arguments.maxBranchingFactor = 2;
			arguments.bvh = bvh;
			arguments.primitives = primitives.data();
			arguments.primitiveCount = primitives.size();
			arguments.primitiveArrayCapacity = primitives.size();
			arguments.minLeafSize = 1;
			arguments.maxLeafSize = 1;
			arguments.createNode = embrreeCreateNode;
			arguments.setNodeChildren = embreeSetNodeChildren;
			arguments.setNodeBounds = embreeSetNodeBounds;
			arguments.createLeaf = embreeCreateLeaf;
			arguments.splitPrimitive = nullptr;
			arguments.userPtr = &context;
			void* bvh_root = rtcBuildBVH(&arguments);

			rtcReleaseBVH(bvh);
			rtcReleaseDevice(device);

			m_rootNode = ptr2node(bvh_root);
			m_internals[m_rootNode.m_index].parent = NodeIndex::invalid();
		}
#endif
		bool empty() const
		{
			return m_internals.empty();
		}
		void validate() const
		{
			validate_lbvh(m_rootNode, m_internals.data(), m_deltas.data(), INT_MAX);
		}

		NodeIndex m_rootNode;
		std::vector<InternalNode> m_internals;
		std::vector<uint8_t> m_deltas;
	};
	struct Hit
	{
		float t = FLT_MAX;
		float2 uv = {};
		float3 ng = {};
		uint32_t triangleIndex = 0xFFFFFFFF;
	};

	inline float3 invRd(float3 rd)
	{
		return clamp(1.0f / rd, -FLT_MAX, FLT_MAX);
	}

	// stackful traversal for reference
	inline void intersect_stackfull(
		Hit* hit,
		const InternalNode* nodes,
		const Triangle* triangles,
		NodeIndex rootNode,
		float3 ro,
		float3 rd,
		float3 one_over_rd)
	{
		int sp = 1;
		NodeIndex stack[64];
		stack[0] = rootNode;

		while (sp)
		{
			NodeIndex node = stack[--sp];

			if (node.m_isLeaf)
			{
				float t;
				float u, v;
				float3 ng;
				const Triangle& tri = triangles[node.m_index];
				if (intersectRayTriangle(&t, &u, &v, &ng, 0.0f, hit->t, ro, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
				{
					hit->t = t;
					hit->uv = make_float2(u, v);
					hit->ng = ng;
					hit->triangleIndex = node.m_index;
				}
				continue;
			}

			const AABB& L = nodes[node.m_index].aabbs[0];
			const AABB& R = nodes[node.m_index].aabbs[1];

			float2 rangeL = slabs(ro, one_over_rd, L.lower, L.upper);
			float2 rangeR = slabs(ro, one_over_rd, R.lower, R.upper);
			bool hitL = rangeL.x <= rangeL.y;
			bool hitR = rangeR.x <= rangeR.y;

			if (hitL && hitR)
			{
				int mask = 0x0;
				if (rangeR.x < rangeL.x)
				{
					mask = 0x1;
				}
				stack[sp++] = nodes[node.m_index].children[1 ^ mask];
				stack[sp++] = nodes[node.m_index].children[0 ^ mask];
			}
			else if (hitL || hitR)
			{
				stack[sp++] = nodes[node.m_index].children[hitL ? 0 : 1];
			}
		}
	}

	inline void intersect(
		Hit* hit,
		const InternalNode* nodes,
		const Triangle* triangles,
		NodeIndex node,
		float3 ro,
		float3 rd,
		float3 one_over_rd)
	{
		NodeIndex curr_node = node;
		NodeIndex prev_node = NodeIndex::invalid();

		while (curr_node != NodeIndex::invalid())
		{
			if (curr_node.m_isLeaf)
			{
				float t;
				float u, v;
				float3 ng;
				const Triangle& tri = triangles[curr_node.m_index];
				if (intersectRayTriangle(&t, &u, &v, &ng, 0.0f, hit->t, ro, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
				{
					hit->t = t;
					hit->uv = make_float2(u, v);
					hit->triangleIndex = curr_node.m_index;
					hit->ng = ng;
				}

				std::swap(curr_node, prev_node);
				continue;
			}

			AABB L = nodes[curr_node.m_index].aabbs[0];
			AABB R = nodes[curr_node.m_index].aabbs[1];
			float2 rangeL = slabs(ro, one_over_rd, L.lower, L.upper);
			float2 rangeR = slabs(ro, one_over_rd, R.lower, R.upper);
			bool hitL = rangeL.x <= rangeL.y;
			bool hitR = rangeR.x <= rangeR.y;

			NodeIndex parent_node = nodes[curr_node.m_index].parent;
			NodeIndex near_node = nodes[curr_node.m_index].children[0];
			NodeIndex far_node = nodes[curr_node.m_index].children[1];

			int nHits = 0;
			if (hitL && hitR)
			{
				if (rangeR.x < rangeL.x)
				{
					std::swap(near_node, far_node);
				}
				nHits = 2;
			}
			else if (hitL || hitR)
			{
				nHits = 1;
				near_node = hitR ? far_node : near_node;
			}

			NodeIndex next_node;
			if (prev_node == parent_node)
			{
				next_node = 0 < nHits ? near_node : parent_node;
			}
			else if (prev_node == near_node)
			{
				next_node = nHits == 2 ? far_node : parent_node;
			}
			else
			{
				next_node = parent_node;
			}

			prev_node = curr_node;
			curr_node = next_node;
		}
	}
}
