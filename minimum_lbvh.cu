#include "minimum_lbvh.h"
#include "helper_math.h"

using namespace minimum_lbvh;

template <class T, class F>
__device__ T ReduceBlock(T val, T* smem, int blockDim, F f)
{
	smem[threadIdx.x] = val;
	__syncthreads();
	for (int i = 1; i < blockDim; i *= 2)
	{
		if (threadIdx.x < (threadIdx.x ^ i))
			smem[threadIdx.x] = f(smem[threadIdx.x], smem[threadIdx.x ^ i]);
		__syncthreads();
	}
	return smem[0];
}


// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/72461459#72461459
__device__ float atomicMinFloat(float* addr, float value) {
	float old;
	old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

	return old;
}

__device__ float atomicMaxFloat(float* addr, float value) {
	float old;
	old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

	return old;
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

	aabb.lower = ReduceBlock(aabb.lower, s_mem, blockDim.x, [](float3 a, float3 b) { return fminf(a, b); });
	aabb.upper = ReduceBlock(aabb.upper, s_mem, blockDim.x, [](float3 a, float3 b) { return fmaxf(a, b); });

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