#pragma once

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
	template <class T>
	inline T ss_clamp(T x, T lower, T upper)
	{
		return ss_min(ss_max(x, lower), upper);
	}
	struct Vec2
	{
		float xs[2];
		float& operator[](int i) { return xs[i]; }
		const float& operator[](int i) const { return xs[i]; }
	};
	struct Vec3
	{
		float xs[3];
		float& operator[](int i) { return xs[i]; }
		const float& operator[](int i) const { return xs[i]; }
	};

	inline Vec3 operator-(Vec3 a, Vec3 b)
	{
		Vec3 r;
		for (int i = 0; i < 3; i++)
		{
			r[i] = a[i] - b[i];
		}
		return r;
	}
	inline Vec3 operator+(Vec3 a, Vec3 b)
	{
		Vec3 r;
		for (int i = 0; i < 3; i++)
		{
			r[i] = a[i] + b[i];
		}
		return r;
	}
	inline Vec3 operator*(Vec3 a, float s)
	{
		Vec3 r;
		for (int i = 0; i < 3; i++)
		{
			r[i] = a[i] * s;
		}
		return r;
	}
	inline Vec3 operator*(Vec3 a, Vec3 b)
	{
		Vec3 r;
		for (int i = 0; i < 3; i++)
		{
			r[i] = a[i] * b[i];
		}
		return r;
	}

	inline Vec3 ss_min(Vec3 a, Vec3 b)
	{
		Vec3 r;
		for (int axis = 0; axis < 3; axis++)
		{
			r[axis] = ss_min(a[axis], b[axis]);
		}
		return r;
	}
	inline Vec3 ss_max(Vec3 a, Vec3 b)
	{
		Vec3 r;
		for (int axis = 0; axis < 3; axis++)
		{
			r[axis] = ss_max(a[axis], b[axis]);
		}
		return r;
	}
	inline float compMin(Vec3 v)
	{
		return ss_min(ss_min(v[0], v[1]), v[2]);
	}
	inline float compMax(Vec3 v)
	{
		return ss_max(ss_max(v[0], v[1]), v[2]);
	}

	inline Vec2 slabs(Vec3 ro, Vec3 one_over_rd, Vec3 lower, Vec3 upper)
	{
		Vec3 t0 = (lower - ro) * one_over_rd;
		Vec3 t1 = (upper - ro) * one_over_rd;

		Vec3 tmin = ss_min(t0, t1);
		Vec3 tmax = ss_max(t0, t1);
		float region_min = compMax(tmin);
		float region_max = compMin(tmax);

		region_min = fmaxf(region_min, 0.0f);

		return { region_min, region_max };
	}

	inline float dot(Vec3 a, Vec3 b)
	{
		float r = 0.0f;
		for (int axis = 0; axis < 3; axis++)
		{
			r += a[axis] * b[axis];
		}
		return r;
	}

	inline Vec3 cross(Vec3 a, Vec3 b)
	{
		return {
			a[1] * b[2] - b[1] * a[2],
			a[2] * b[0] - b[2] * a[0],
			a[0] * b[1] - b[0] * a[1]
		};
	}

	struct Triangle
	{
		Vec3 vs[3];
	};

	struct AABB
	{
		Vec3 lower;
		Vec3 upper;

		void setEmpty()
		{
			for (int i = 0; i < 3; i++)
			{
				lower[i] = +FLT_MAX;
				upper[i] = -FLT_MAX;
			}
		}
		void extend(const Vec3& p)
		{
			for (int i = 0; i < 3; i++)
			{
				lower[i] = ss_min(lower[i], p[i]);
				upper[i] = ss_max(upper[i], p[i]);
			}
		}

		void extend(const AABB& b)
		{
			for (int i = 0; i < 3; i++)
			{
				lower[i] = ss_min(lower[i], b.lower[i]);
				upper[i] = ss_max(upper[i], b.upper[i]);
			}
		}
		float surface_area() const
		{
			Vec3 size = upper - lower;
			return (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]) * 2.0f;
		}

		bool isEmpty()
		{
			for (int i = 0; i < 3; i++)
			{
				if (upper[i] <= lower[i])
				{
					return true;
				}
			}
			return false;
		}
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

	inline float remap(float value, float inputMin, float inputMax, float outputMin, float outputMax)
	{
		return (value - inputMin) * ((outputMax - outputMin) / (inputMax - inputMin)) + outputMin;
	}

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
