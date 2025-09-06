#include "pr.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <stack>

#include "Orochi/Orochi.h"
#include "tinyhiponesweep.h"

#define ENABLE_GPU_BUILDER
#include "minimum_lbvh.h"
#include "camera.h"

#include "typedbuffer.h"
#include "shader.h"

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

class DeviceStopwatch
{
public:
    DeviceStopwatch(oroStream stream)
    {
        m_stream = stream;
        oroEventCreateWithFlags(&m_start, oroEventDefault);
        oroEventCreateWithFlags(&m_stop, oroEventDefault);
    }
    ~DeviceStopwatch()
    {
        oroEventDestroy(m_start);
        oroEventDestroy(m_stop);
    }
    DeviceStopwatch(const DeviceStopwatch&) = delete;
    void operator=(const DeviceStopwatch&) = delete;

    void start() { oroEventRecord(m_start, m_stream); }
    void stop() { oroEventRecord(m_stop, m_stream); }

    float getElapsedMs() const
    {
        oroEventSynchronize(m_stop);
        float ms = 0;
        oroEventElapsedTime(&ms, m_start, m_stop);
        return ms;
    }
private:
    oroStream m_stream;
    oroEvent m_start;
    oroEvent m_stop;
};

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);
    SetDataDir(ExecutableDir());

    if (oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0))
    {
        printf("failed to init..\n");
        return 0;
    }

    int DEVICE_INDEX = 2;
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

    //using ValType = uint64_t;
    //PCG rng;
    //for (int j = 0; j < 3; j++)
    //{
    //    int N = (1u << 23) + 3;

    //    TypedBuffer<ValType> xs(TYPED_BUFFER_HOST);
    //    TypedBuffer<uint32_t> indices(TYPED_BUFFER_HOST);
    //    xs.allocate(N);
    //    indices.allocate(N);
    //    for (int i = 0; i < N; i++)
    //    {
    //        xs[i] = rng.uniformi();
    //        indices[i] = i;
    //    }

    //    TypedBuffer<ValType> xsBuffer(TYPED_BUFFER_DEVICE);
    //    xs.copyTo(&xsBuffer);
    //    TypedBuffer<uint32_t> indicesBuffer(TYPED_BUFFER_DEVICE);
    //    indices.copyTo(&indicesBuffer);
    //    TypedBuffer<ValType> xsTmp(TYPED_BUFFER_DEVICE);
    //    TypedBuffer<uint32_t> indicesTmp(TYPED_BUFFER_DEVICE);
    //    xsTmp.allocate(xs.size());
    //    indicesTmp.allocate(xs.size());

    //    DeviceStopwatch sw(0);
    //    sw.start();
    //    onesweep.sort({ xsBuffer.data(), indicesBuffer.data() }, { xsTmp.data(), indicesTmp.data() }, N, 0, sizeof(ValType) * 8, 0);
    //    sw.stop();
    //    printf("%f\n", sw.getElapsedMs());

    //    TypedBuffer<ValType> sortedXs = xsBuffer.copyToHost();
    //    TypedBuffer<uint32_t> sortedIndices = indicesBuffer.copyToHost();
    //    for (int i = 0; i < N - 1; i++)
    //    {
    //        MINIMUM_LBVH_ASSERT(sortedXs[i] == xs[sortedIndices[i]]);
    //        MINIMUM_LBVH_ASSERT(sortedXs[i] <= sortedXs[i + 1]);
    //    }
    //}

    minimum_lbvh::BVHGPUBuilder gpuBuilder(
        GetDataPath("../minimum_lbvh.cu").c_str(),
        GetDataPath("../").c_str()
    );

    std::vector<std::string> options;
    options.push_back("-I");
    options.push_back(GetDataPath("../"));
    Shader shader(GetDataPath("../main_gpu.cu").c_str(), "main_gpu", options);

    TypedBuffer<uint32_t> pixels(TYPED_BUFFER_DEVICE);

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
    TypedBuffer<minimum_lbvh::Triangle> triangles(TYPED_BUFFER_HOST);
    TypedBuffer<minimum_lbvh::Triangle> trianglesDevice(TYPED_BUFFER_DEVICE);

    minimum_lbvh::BVHCPUBuilder builder;
    TypedBuffer<TriangleAttrib> triangleAttribs(TYPED_BUFFER_HOST);

    minimum_lbvh::BVHGPUStackBuffer stackBuffer;

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

            triangles.allocate(faceCounts.count());
            triangleAttribs.allocate(faceCounts.count());

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

                triangles[i] = tri;
                triangleAttribs[i] = attrib;
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

            if (gpuBuilder.empty())
            {
                triangles.copyTo(&trianglesDevice);
                gpuBuilder.build(trianglesDevice.data(), trianglesDevice.size(), onesweep, 0 /*stream*/);
            }
        });

        int imageWidth = GetScreenWidth();
        int imageHeight = GetScreenHeight();

        pixels.allocate(imageWidth * imageHeight);
        {
            DeviceStopwatch sw(0);
            sw.start();

            RayGenerator rayGenerator;
            rayGenerator.lookat(to(camera.origin), to(camera.lookat), to(camera.up), camera.fovy, imageWidth, imageHeight);

            //shader.launch("render",
            //    ShaderArgument()
            //    .value(pixels.data())
            //    .value(int2{ imageWidth, imageHeight })
            //    .value(rayGenerator)
            //    .value(gpuBuilder.m_rootNode)
            //    .value(gpuBuilder.m_internals)
            //    .value(trianglesDevice.data())
            //    .value(stackBuffer.getBuffer()),
            //    div_round_up64(imageWidth, 16), div_round_up64(imageHeight, 16), 1,
            //    16, 16, 1,
            //    0
            //);

            shader.launch("ao",
                ShaderArgument()
                .value(pixels.data())
                .value(int2{ imageWidth, imageHeight })
                .value(rayGenerator)
                .value(gpuBuilder.m_rootNode)
                .value(gpuBuilder.m_internals)
                .value(trianglesDevice.data())
                .value(stackBuffer.getBuffer()),
                div_round_up64(imageWidth, 16), div_round_up64(imageHeight, 16), 1,
                16, 16, 1,
                0
            );

            sw.stop();
            printf("render %f ms\n", sw.getElapsedMs());
        }

        static Image2DRGBA8 image;
        image.allocate(imageWidth, imageHeight);
        oroMemcpyDtoH(image.data(), pixels.data(), pixels.bytes());

        texture->upload(image);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::Checkbox("showWire", &showWire);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
