include "libs/PrLib"

workspace "MinimumLBVH"
    location "build"
    configurations { "Debug", "Release" }
    startproject "main"

architecture "x86_64"

externalproject "prlib"
	location "libs/PrLib/build" 
    kind "StaticLib"
    language "C++"

project "main"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main.cpp" }

    -- Embree
    libdirs { "libs/embree-4.4.0/lib" }
    includedirs { "libs/embree-4.4.0/include" }
    links{
        "embree4",
        "tbb12",
    }
    postbuildcommands {
        "{COPYFILE} ../libs/embree-4.4.0/bin/*.dll %{cfg.targetdir}/*.dll",
    }

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("Main_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("Main")
        optimize "Full"
    filter{}

project "main_gpu"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main_gpu.cpp" }
    files { "shader.h", "typedbuffer.h", "typedbuffer.natvis" }

    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }
    
    -- Cuda
    defines {"OROCHI_ENABLE_CUEW"}
    includedirs {"$(CUDA_PATH)/include"}

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("Main_GPU_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("Main_GPU")
        optimize "Full"
    filter{}

project "unittest"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }
    cppdialect "C++14"
    buildoptions "/bigobj"

    -- Src
    files { "unittest.cpp", "catch_amalgamated.cpp", "catch_amalgamated.hpp" }
    includedirs { "." }

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("Unittest_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("Unittest")
        optimize "Full"
    filter{}

project "copying"    
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }
    postbuildcommands { 
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }