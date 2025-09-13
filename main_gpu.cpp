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
#include "sobol.h"

#include "tiny_obj_loader.h"

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
    float3 reflectance;
    float3 emissive;
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
    TypedBuffer<float4> accumulators(TYPED_BUFFER_DEVICE);

    Camera3D camera;
    camera.origin = { 0.0f, 1.0f, 4.0f };
    camera.lookat = { 0, 1.0f, 0 };

    //camera.fovy = 0.002f;
    //camera.origin = { 500, 500, 500 };

    SetDataDir(ExecutableDir());
    std::string err;
    //std::shared_ptr<FScene> scene = ReadWavefrontObj(GetDataPath("assets/blocks_ao.obj"), err);
    std::shared_ptr<FScene> scene = ReadWavefrontObj(GetDataPath("assets/cornelbox.obj"), err);

    double e = GetElapsedTime();
    bool showWire = false;
    bool useSobol = true;

    enum {
        MODE_NORMAL,
        MODE_AO,
        MODE_PT,
    };
    int mode = 2;
    int maxSPP = 16;

    // BVH
    std::vector<minimum_lbvh::Triangle> triangles(TYPED_BUFFER_HOST);
    TypedBuffer<minimum_lbvh::Triangle> trianglesDevice(TYPED_BUFFER_DEVICE);

    minimum_lbvh::BVHCPUBuilder builder;
    TypedBuffer<TriangleAttrib> triangleAttribs(TYPED_BUFFER_HOST);

    ITexture* texture = CreateTexture();

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            if (UpdateCameraBlenderLike(&camera))
            {
                oroMemsetD8(accumulators.data(), 0, accumulators.size() * sizeof(float4));
            }
        }
        
        //ClearBackground(0.1f, 0.1f, 0.1f, 1);
        ClearBackground(texture);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XZ, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

#if 1
        scene->visitPolyMesh([&](std::shared_ptr<const FPolyMeshEntity> polymesh) {
            if (polymesh->visible() == false)
            {
                return;
            }
            ColumnView<int32_t> faceCounts(polymesh->faceCounts());
            ColumnView<int32_t> indices(polymesh->faceIndices());
            ColumnView<glm::vec3> positions(polymesh->positions());
            ColumnView<glm::vec3> normals(polymesh->normals());

            triangles.resize(faceCounts.count());
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
                trianglesDevice << triangles;
                // triangles.copyTo(&trianglesDevice);
                gpuBuilder.build(trianglesDevice.data(), trianglesDevice.size(), onesweep, 0 /*stream*/);
            }
        });
#else
        if (builder.empty())
        {
            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;

            std::string objError;
            std::string objWarn;
            std::string filename = GetDataPath("assets/cornelbox.obj");
            bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &objWarn, &objError, filename.c_str(), GetPathDirname(filename).c_str());
            if (!success)
            {
                printf("%s\n", objError.c_str());
                abort();
            }
            if (!objWarn.empty())
            {
                printf("warn: %s\n", objWarn.c_str());
            }

            for (size_t s = 0; s < shapes.size(); s++)
            {
                // Loop over faces(polygon)
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
                {
                    size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
                    assert(fv == 3);
                    // Loop over vertices in the face.
                    float3 vertices[3];
                    minimum_lbvh::Triangle triangle;
                    TriangleAttrib triangleAttrib;

                    for (size_t v = 0; v < fv; v++)
                    {
                        // access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                        tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                        tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                        tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                        tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                        tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                        tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

                        triangle.vs[v] = { vx, vy, vz };
                        triangleAttrib.shadingNormals[v] = { nx, ny, nz };
                    }
                    index_offset += fv;

                    // per-face material
                    int matId = shapes[s].mesh.material_ids[f];
                    if (!materials.empty())
                    {
                        triangleAttrib.reflectance = { materials[matId].diffuse[0], materials[matId].diffuse[1], materials[matId].diffuse[2] };
                        triangleAttrib.emissive = { materials[matId].emission[0], materials[matId].emission[1], materials[matId].emission[2] };
                    }
                    triangles.push_back(triangle);
                    triangleAttribs.push_back(triangleAttrib);
                }
            }

            printf("");
        }
#endif

        int imageWidth = GetScreenWidth();
        int imageHeight = GetScreenHeight();

        pixels.allocate(imageWidth * imageHeight);

        if (accumulators.size() != imageWidth * imageHeight)
        {
            accumulators.allocate(imageWidth* imageHeight);
            oroMemsetD8(accumulators.data(), 0, accumulators.size() * sizeof(float4));
        }

        {
            DeviceStopwatch sw(0);
            sw.start();

            RayGenerator rayGenerator;
            rayGenerator.lookat(to(camera.origin), to(camera.lookat), to(camera.up), camera.fovy, imageWidth, imageHeight);

            if (mode == 0)
            {
                shader.launch("normal",
                    ShaderArgument()
                    .value(pixels.data())
                    .value(int2{ imageWidth, imageHeight })
                    .value(rayGenerator)
                    .value(gpuBuilder.m_rootNode)
                    .value(gpuBuilder.m_internals)
                    .value(trianglesDevice.data()),
                    div_round_up64(imageWidth, 16), div_round_up64(imageHeight, 16), 1,
                    16, 16, 1,
                    0
                );
            }
            else if( mode == 1)
            {
                shader.launch("ao",
                    ShaderArgument()
                    .value(pixels.data())
                    .value(int2{ imageWidth, imageHeight })
                    .value(rayGenerator)
                    .value(gpuBuilder.m_rootNode)
                    .value(gpuBuilder.m_internals)
                    .value(trianglesDevice.data())
                    .value(useSobol ? 1 : 0),
                    div_round_up64(imageWidth, 16), div_round_up64(imageHeight, 16), 1,
                    16, 16, 1,
                    0
                );
            }
            else
            {
                shader.launch("pt",
                    ShaderArgument()
                    .value(pixels.data())
                    .value(accumulators.data())
                    .value(int2{ imageWidth, imageHeight })
                    .value(rayGenerator)
                    .value(gpuBuilder.m_rootNode)
                    .value(gpuBuilder.m_internals)
                    .value(trianglesDevice.data())
                    .value(useSobol ? 1 : 0)
                    .value(maxSPP),
                    div_round_up64(imageWidth, 16), div_round_up64(imageHeight, 16), 1,
                    16, 16, 1,
                    0
                );
            }

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
        if (ImGui::Checkbox("use sobol", &useSobol))
        {
            oroMemsetD8(accumulators.data(), 0, accumulators.size() * sizeof(float4));
        }
        ImGui::InputInt("max spp", &maxSPP);

        float4 pix;
        oroMemcpyDtoH(&pix, accumulators.data(), sizeof(pix));
        ImGui::Text("spp : %d", (int)pix.w);

        ImGui::RadioButton("normal", &mode, 0);
        ImGui::RadioButton("ao", &mode, 1);
        ImGui::RadioButton("pt", &mode, 2);
        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
