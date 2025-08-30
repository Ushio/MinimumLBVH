#include "pr.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <stack>

#include "Orochi/Orochi.h"
#include "tinyhiponesweep.h"
#include "minimum_lbvh.h"

inline glm::vec3 to(float3 p)
{
    return { p.x, p.y, p.z };
}
inline float3 to(glm::vec3 p)
{
    return { p.x, p.y, p.z };
}

struct TriangleAttrib
{
    float3 shadingNormals[3];
};

int main() {
    using namespace pr;

    if (oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0))
    {
        printf("failed to init..\n");
        return 0;
    }

    int DEVICE_INDEX = 0;
    oroInit(0);
    oroDevice device;
    oroDeviceGet(&device, DEVICE_INDEX);
    oroCtx ctx;
    oroCtxCreate(&ctx, 0, device);
    oroCtxSetCurrent(ctx);

    oroDeviceProp props;
    oroGetDeviceProperties(&props, device);

    bool isNvidia = oroGetCurAPI(0) & ORO_API_CUDADRIVER;

    printf("Device: %s\n", props.name);
    printf("Cuda: %s\n", isNvidia ? "Yes" : "No");

    tinyhiponesweep::OnesweepSort onesweep(device);

    PCG rng;
    int N = 10000;
    std::vector<uint32_t> xs(N);
    for (int i = 0; i < N; i++)
    {
        xs[i] = rng.uniform();
    }

    uint32_t* xsBuffer;
    uint32_t* xsTmp;
    oroMalloc((void**)&xsBuffer, sizeof(uint32_t) * xs.size());
    oroMalloc((void**)&xsTmp, sizeof(uint32_t) * xs.size());
    oroMemcpyHtoD(xsBuffer, xs.data(), sizeof(uint32_t) * xs.size());

    onesweep.sort({ xsBuffer, nullptr }, { xsTmp, nullptr }, N, 0, 32, 0);

    oroMemcpyDtoH(xs.data(), xsBuffer, sizeof(uint32_t) * xs.size());

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 8.0f, 8.0f, 8.0f };
    camera.lookat = { 0, 0, 0 };

    //camera.fovy = 0.002f;
    //camera.origin = { 500, 500, 500 };

    SetDataDir(ExecutableDir());
    std::string err;
    std::shared_ptr<FScene> scene = ReadWavefrontObj(GetDataPath("test.obj"), err);

    double e = GetElapsedTime();
    bool showWire = false;
    bool smooth = false;

    // BVH
    std::vector<minimum_lbvh::Triangle> triangles;
    minimum_lbvh::BVHCPUBuilder builder;
    std::vector<TriangleAttrib> triangleAttribs;

    ITexture* texture = CreateTexture();

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        //ClearBackground(0.1f, 0.1f, 0.1f, 1);
        ClearBackground(texture);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        static glm::vec3 test_p;
        ManipulatePosition(camera, &test_p, 0.3f);

        scene->visitPolyMesh([&](std::shared_ptr<const FPolyMeshEntity> polymesh) {
            if (polymesh->visible() == false)
            {
                return;
            }
            ColumnView<int32_t> faceCounts(polymesh->faceCounts());
            ColumnView<int32_t> indices(polymesh->faceIndices());
            ColumnView<glm::vec3> positions(polymesh->positions());
            ColumnView<glm::vec3> normals(polymesh->normals());

            triangles.clear();
            triangleAttribs.clear();

            int indexBase = 0;
            for (int i = 0; i < faceCounts.count(); i++)
            {
                int nVerts = faceCounts[i];
                PR_ASSERT(nVerts == 3);
                minimum_lbvh::Triangle tri;
                TriangleAttrib attrib;
                for (int j = 0; j < nVerts; ++j)
                {
                    glm::vec3 p = positions[indices[indexBase + j]];
                    tri.vs[j] = { p.x, p.y, p.z };

                    glm::vec3 ns = normals[indexBase + j];
                    attrib.shadingNormals[j] = { ns.x, ns.y, ns.z };
                }

                float3 e0 = tri.vs[1] - tri.vs[0];
                float3 e1 = tri.vs[2] - tri.vs[1];
                float3 e2 = tri.vs[0] - tri.vs[2];

                triangles.push_back(tri);
                triangleAttribs.push_back(attrib);
                indexBase += nVerts;
            }

            if (showWire)
            {
                pr::PrimBegin(pr::PrimitiveMode::Lines);

                for (auto tri : triangles)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        float3 v0 = tri.vs[j];
                        float3 v1 = tri.vs[(j + 1) % 3];
                        pr::PrimVertex(to(v0), {255, 255, 255});
                        pr::PrimVertex(to(v1), {255, 255, 255});
                    }
                }

                pr::PrimEnd();
            }

            if (builder.empty())
            {
#if 1
                Stopwatch sw;
                builder.build(triangles.data(), triangles.size(), true /* isParallel */);
                printf("build %f\n", sw.elapsed());

                builder.validate();
#else
                Stopwatch sw;
                builder.buildByEmbree(triangles.data(), triangles.size(), RTC_BUILD_QUALITY_LOW);
                printf("embree build %f\n", sw.elapsed());
#endif
            }
        });

        //traverse(internals.data(), rootNode, 0);

        int stride = 2;
        Image2DRGBA8 image;
        image.allocate(GetScreenWidth() / stride, GetScreenHeight() / stride);

        CameraRayGenerator rayGenerator(GetCurrentViewMatrix(), GetCurrentProjMatrix(), image.width(), image.height());

        //for (int j = 0; j < image.height(); ++j)
        ParallelFor(image.height(), [&](int j) {
            for (int i = 0; i < image.width(); ++i)
            {
                glm::vec3 ro, rd;
                rayGenerator.shoot(&ro, &rd, i, j, 0.5f, 0.5f);

                minimum_lbvh::Hit hit;
                minimum_lbvh::intersect(&hit, builder.m_internals.data(), triangles.data(), builder.m_rootNode, to(ro), to(rd), minimum_lbvh::invRd(to(rd)));
                if (hit.t != FLT_MAX)
                {
                    float3 n = normalize(hit.ng);
                    if (smooth)
                    {
                        TriangleAttrib attrib = triangleAttribs[hit.triangleIndex];
                        n = attrib.shadingNormals[0] +
                            (attrib.shadingNormals[1] - attrib.shadingNormals[0]) * hit.uv.x +
                            (attrib.shadingNormals[2] - attrib.shadingNormals[0]) * hit.uv.y;
                        n = normalize(n);
                    }
                    float3 color = (n + make_float3(1.0f)) * 0.5f;
                    image(i, j) = { 255 * color.x, 255 * color.y, 255 * color.z, 255 };
                }
                else
                {
                    image(i, j) = { 0, 0, 0, 255 };
                }
            }
        }
        );

        texture->upload(image);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::Checkbox("showWire", &showWire);
        ImGui::Checkbox("smooth", &smooth);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
