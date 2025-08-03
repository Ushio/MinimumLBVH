#pragma once

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define MINIMUM_LBVH_KERNELCC
#endif

#include <stdint.h>
#include <intrin.h>

#define MORTON_MAX_VALUE_3D 0x1FFFFF

namespace min_lbvh
{
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

	inline uint32_t getThirdBits(uint64_t m)
	{
		const uint64_t masks[6] = { 0x1fffffllu, 0x1f00000000ffffllu, 0x1f0000ff0000ffllu, 0x100f00f00f00f00fllu, 0x10c30c30c30c30c3llu, 0x1249249249249249llu };
		uint64_t x = m & masks[5];
		x = (x ^ (x >> 2)) & masks[4];
		x = (x ^ (x >> 4)) & masks[3];
		x = (x ^ (x >> 8)) & masks[2];
		x = (x ^ (x >> 16)) & masks[1];
		x = (x ^ (x >> 32)) & masks[0];
		return static_cast<uint32_t>(x);
	}

	// method to seperate bits from a given integer 3 positions apart
	inline uint64_t splitBy3(uint32_t a)
	{
		uint64_t x = a & 0x1FFFFF;
		x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
		x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
		x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
		x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
		x = (x | x << 2) & 0x1249249249249249;
		return x;
	}

	inline void decodeMortonCode(uint64_t morton, uint32_t* x, uint32_t* y, uint32_t* z)
	{
		*x = getThirdBits(morton);
		*y = getThirdBits(morton >> 1);
		*z = getThirdBits(morton >> 2);
	}
	inline uint64_t encodeMortonCode(uint32_t x, uint32_t y, uint32_t z)
	{
		uint64_t answer = 0;
		answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
		return answer;
	}
}
