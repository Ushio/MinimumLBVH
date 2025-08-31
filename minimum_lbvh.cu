#include "minimum_lbvh.h"
#include "helper_math.h"

using namespace minimum_lbvh;

template <typename T>
__device__ T ReduceMinBlock(T val, T* smem, int blockDim)
{
	smem[threadIdx.x] = val;
	__syncthreads();
	for (int i = 1; i < blockDim; i *= 2)
	{
		if (threadIdx.x < (threadIdx.x ^ i))
			smem[threadIdx.x] = fminf(smem[threadIdx.x], smem[threadIdx.x ^ i]);
		__syncthreads();
	}
	return smem[0];
}
template <typename T>
__device__ T ReduceMaxBlock(T val, T* smem, int blockDim)
{
	smem[threadIdx.x] = val;
	__syncthreads();
	for (int i = 1; i < blockDim; i *= 2)
	{
		if (threadIdx.x < (threadIdx.x ^ i))
			smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x ^ i]);
		__syncthreads();
	}
	return smem[0];
}

__device__ float atomicMaxFloat(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}
__device__ float atomicMinFloat(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__float_as_int(fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

extern "C" __global__ void getSceneAABB(AABB* sceneAABB, const Triangle* triangles, int nTriangles)
{
	__shared__ float3 s_mem[256];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	AABB aabb = AABB::empty();
	if (idx < nTriangles)
	{
		Triangle tri = triangles[idx];
		for (int i = 0; i < 3; i++)
		{
			aabb.extend(tri.vs[i]);
		}
	}

	aabb.lower = ReduceMinBlock(aabb.lower, s_mem, blockDim.x);
	aabb.upper = ReduceMaxBlock(aabb.upper, s_mem, blockDim.x);

	if (threadIdx.x == 0)
	{
		atomicMinFloat(&sceneAABB->lower.x, aabb.lower.x);
		atomicMinFloat(&sceneAABB->lower.y, aabb.lower.y);
		atomicMinFloat(&sceneAABB->lower.z, aabb.lower.z);
		atomicMaxFloat(&sceneAABB->upper.x, aabb.upper.x);
		atomicMaxFloat(&sceneAABB->upper.y, aabb.upper.y);
		atomicMaxFloat(&sceneAABB->upper.z, aabb.upper.z);
	}
}

extern "C" __global__ void buildMortons(IndexedMorton* indexedMortons, const Triangle *triangles, int nTriangles)
{
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//if (nTriangles <= idx)
	//{
	//	return;
	//}

	//Triangle tri = triangles[i];
	//float3 center = (tri.vs[0] + tri.vs[1] + tri.vs[2]) / 3.0f;
	//indexedMortons[i].morton = (uint32_t)(sceneAABB.encodeMortonCode(center) >> 31); // take higher 32bits out of 63bits
	//indexedMortons[i].index = i;
}