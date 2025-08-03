#include "catch_amalgamated.hpp"
#include "prp.hpp"
#include "minimum_lbvh.h"

TEST_CASE("Morton") {
	using namespace min_lbvh;
	using namespace pr;
	PCG random;
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
	using namespace min_lbvh;
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

	for (int i = 0; i < 32; i++)
	{
		uint32_t input = (0x80000000) >> i;
		REQUIRE(clz(input) == i);
		REQUIRE(clz(input | (input - 1)) == i);
	}
	for (int i = 0; i < 64; i++)
	{
		uint64_t input = (0x8000000000000000LLU) >> i;
		REQUIRE(clz64(input) == i);
		REQUIRE(clz64(input | (input - 1)) == i);
	}
}
