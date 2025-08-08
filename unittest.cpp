#include "catch_amalgamated.hpp"
#include "prp.hpp"
#include "minimum_lbvh.h"
#include <bitset>

TEST_CASE("Morton") {
	using namespace minimum_lbvh;
	using namespace pr;
	PCG random;
	for (int x = 0; x <= MORTON_MAX_VALUE_3D; x++)
	{
		std::bitset<64> splatted(splat3(x));
		for (int i = 0; i < 21; i++)
		{
			bool setbit = x & (1LLU << i);
			REQUIRE(splatted[i * 3] == setbit);
		}
		REQUIRE(x == compact3(splatted.to_ullong()));
	}

	for (int i = 0; i < 10000000; i++)
	{
		uint32_t x = random.uniformi() & MORTON_MAX_VALUE_3D; // 21 bits
		uint32_t y = random.uniformi() & MORTON_MAX_VALUE_3D; // 21 bits
		uint32_t z = random.uniformi() & MORTON_MAX_VALUE_3D; // 21 bits
		
		uint64_t m0 = encodeMortonCode_Naive(x, y, z);
		uint64_t m1 = encodeMortonCode(x, y, z);
		REQUIRE(m0 == m1);

		uint32_t x2, y2, z2;
		decodeMortonCode(m0, &x2, &y2, &z2);
		REQUIRE(x == x2);
		REQUIRE(y == y2);
		REQUIRE(z == z2);
	}
}

TEST_CASE("clz") {
	using namespace minimum_lbvh;
	using namespace pr;
	
	/*
	printf("--32bit--\n");
	for (int i = 0; i < 32; i++)
	{
		uint32_t input = (0x80000000) >> i;
		printf("%x, %d\n", input, clz(input));
	}
	printf("%x, %d\n", 0, clz(0));

	printf("--64bit--\n");
	for (int i = 0; i < 64; i++)
	{
		uint64_t input = (0x8000000000000000LLU) >> i;
		printf("%llx, %d\n", input, clz64(input));
	}
	printf("%llx, %d\n", 0LLU, clz64(0));
	*/
	REQUIRE(clz(0) == 32);
	for (int i = 0; i < 32; i++)
	{
		uint32_t input = (0x80000000) >> i;
		REQUIRE(clz(input) == i);
		REQUIRE(clz(input | (input - 1)) == i);
	}
	REQUIRE(clz64(0) == 64);
	for (int i = 0; i < 64; i++)
	{
		uint64_t input = (0x8000000000000000LLU) >> i;
		REQUIRE(clz64(input) == i);
		REQUIRE(clz64(input | (input - 1)) == i);
	}
}

TEST_CASE("ffs") {
	using namespace minimum_lbvh;
	using namespace pr;
	/*
	printf("--32bit--\n");
	printf("%x, %d\n", 0, ffs(0));
	for (int i = 0; i < 32; i++)
	{
		uint32_t input = 0x1u << i;
		printf("%x, %d\n", input, ffs(input));
		printf("%x, %d\n", input, ffs(input | ~(input - 1)));
	}
	*/

	REQUIRE(ffs(0) == 0);
	for (int i = 0; i < 32; i++)
	{
		uint32_t input = 0x1u << i;
		REQUIRE(ffs(input) == i + 1);
		REQUIRE(ffs(input | ~(input - 1)) == i + 1);
	}

	REQUIRE(ffs64(0) == 0);
	for (int i = 0; i < 64; i++)
	{
		uint64_t input = 0x1llu << i;
		REQUIRE(ffs64(input) == i + 1);
		REQUIRE(ffs64(input | ~(input - 1)) == i + 1);
	}
}
