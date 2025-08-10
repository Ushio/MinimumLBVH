#include "pr.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <sstream>

#include "minimum_lbvh.h"

inline glm::vec3 to(float3 p)
{
    return { p.x, p.y, p.z };
}
inline float3 to(glm::vec3 p)
{
    return { p.x, p.y, p.z };
}

void runToyExample()
{
    std::vector<uint32_t> mortons = {
        0b0010,
        0b0011,
        0b0100,
        0b0101,
        0b1000,
        0b1100,
        0b1101,
        0b1111,
    };

    std::vector<uint8_t> deltas(mortons.size() - 1);
    for (int i = 0; i < deltas.size(); i++)
    {
        deltas[i] = minimum_lbvh::delta(mortons[i], mortons[i + 1]);
    }

    std::vector<minimum_lbvh::Stat> stats(mortons.size() - 1);
    for (int i = 0; i < stats.size(); i++)
    {
        stats[i].oneOfEdges = 0xFFFFFFFF;
    }

    std::vector<minimum_lbvh::InternalNode> internals;
    internals.resize(mortons.size() - 1);

    std::stringstream ss;
    ss << "digraph Tree {\n";
    ss << "    node [shape=circle];\n";

    for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
    {
        uint32_t leaf_lower = i_leaf;
        uint32_t leaf_upper = i_leaf;
        minimum_lbvh::NodeIndex node(i_leaf, true);

        while (leaf_upper - leaf_lower < internals.size())
        {
            // direction from bottom
            int goLeft;
            if (leaf_lower == 0)
            {
                goLeft = 0;
            }
            else if (leaf_upper == stats.size())
            {
                goLeft = 1;
            }
            else
            {
                goLeft = deltas[leaf_lower - 1] < deltas[leaf_upper] ? 1 : 0;
            }

            int parent = goLeft ? (leaf_lower - 1) : leaf_upper;

            internals[parent].children[goLeft] = node;

            ss << "    " << parent << " -> " << (node.m_isLeaf ? "L" : "") << node.m_index << "\n";

            uint32_t index = goLeft ? leaf_upper : leaf_lower;
            std::swap(stats[parent].oneOfEdges, index);

            if (index == 0xFFFFFFFF)
            {
                break;
            }

            leaf_lower = minimum_lbvh::ss_min(leaf_lower, index);
            leaf_upper = minimum_lbvh::ss_max(leaf_upper, index);

            node = minimum_lbvh::NodeIndex(parent, false);
        }
    }

    ss << "    { rank = same; ";
    for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
    {
        ss << "L" << i_leaf << "; ";
    }
    ss << "}\n";

    ss << "}\n";
    printf("%s", ss.str().c_str());
}

int g_debug_index = 0;

void traverse(const minimum_lbvh::InternalNode* nodes, minimum_lbvh::NodeIndex node, int depth )
{
    if (node.m_isLeaf)
    {
        return;
    }

    const minimum_lbvh::InternalNode& cur = nodes[node.m_index];

    for (int i = 0; i < 2; i++)
    {
        if (depth == g_debug_index)
        {
            pr::DrawAABB(to(cur.aabbs[i].lower), to(cur.aabbs[i].upper), { 128, 128, 0 });
        }
        traverse(nodes, cur.children[i], depth + 1);
    }
}
struct Hit
{
    float t = FLT_MAX;
    uint32_t triangleIndex = 0xFFFFFFFF;
    float3 ng = {};
};

inline float3 invRd(float3 rd)
{
    return clamp(1.0f / rd, -FLT_MAX, FLT_MAX);
}
void intersect(
    Hit* hit, 
    const minimum_lbvh::InternalNode* nodes, 
    const minimum_lbvh::Triangle* triangles, 
    minimum_lbvh::NodeIndex node, 
    float3 ro, 
    float3 rd,
    float3 one_over_rd)
{
    if (node.m_isLeaf)
    {
        float t;
        float u, v;
        float3 ng;
        const minimum_lbvh::Triangle& tri = triangles[node.m_index];
        if (minimum_lbvh::intersect_ray_triangle(&t, &u, &v, &ng, 0.0f, hit->t, ro, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
        {
            hit->t = t;
            hit->triangleIndex = node.m_index;
            hit->ng = ng;
        }
        return;
    }

    const minimum_lbvh::AABB& L = nodes[node.m_index].aabbs[0];
    const minimum_lbvh::AABB& R = nodes[node.m_index].aabbs[1];

    float2 rangeL = minimum_lbvh::slabs(ro, one_over_rd, L.lower, L.upper);
    float2 rangeR = minimum_lbvh::slabs(ro, one_over_rd, R.lower, R.upper);

    if (rangeL.x <= rangeL.y)
    {
        intersect(hit, nodes, triangles, nodes[node.m_index].children[0], ro, rd, one_over_rd);
    }
    if (rangeR.x <= rangeR.y)
    {
        intersect(hit, nodes, triangles, nodes[node.m_index].children[1], ro, rd, one_over_rd);
    }
}

int main() {
    using namespace pr;

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 0, 0, 4 };
    camera.lookat = { 0, 0, 0 };

    SetDataDir(ExecutableDir());
    std::string err;
    std::shared_ptr<FScene> scene = ReadWavefrontObj(GetDataPath("test.obj"), err);

    double e = GetElapsedTime();

    runToyExample();


    // BVH
    minimum_lbvh::NodeIndex rootNode;
    std::vector<minimum_lbvh::InternalNode> internals;
    std::vector<minimum_lbvh::Triangle> sorted_triangles;

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

            std::vector<minimum_lbvh::Triangle> triangles;
            int indexBase = 0;
            for (int i = 0; i < faceCounts.count(); i++)
            {
                int nVerts = faceCounts[i];
                PR_ASSERT(nVerts == 3);
                minimum_lbvh::Triangle tri;
                for (int j = 0; j < nVerts; ++j)
                {
                    glm::vec3 p = positions[indices[indexBase + j]];
                    tri.vs[j] = { p.x, p.y, p.z };
                }

                float3 e0 = tri.vs[1] - tri.vs[0];
                float3 e1 = tri.vs[2] - tri.vs[1];
                float3 e2 = tri.vs[0] - tri.vs[2];

                triangles.push_back(tri);
                indexBase += nVerts;
            }

            //if (showWire)
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

            if (internals.empty())
            {
                // printf("build\n");

                internals.resize(triangles.size() - 1);
                sorted_triangles.resize(triangles.size());

                // Scene AABB
                minimum_lbvh::AABB sceneAABB;
                sceneAABB.setEmpty();

                for (auto tri : triangles)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        sceneAABB.extend(tri.vs[j]);
                    }
                }
                // DrawAABB(to(sceneAABB.lower), to(sceneAABB.upper), { 255, 255, 255 });

                std::vector<uint64_t> mortons(triangles.size());
                std::vector<uint32_t> triangleIndices(triangles.size());
                for (int i = 0; i < triangles.size(); i++)
                {
                    minimum_lbvh::Triangle tri = triangles[i];
                    float3 center = (tri.vs[0] + tri.vs[1] + tri.vs[2]) / 3.0f;
                    mortons[i] = sceneAABB.encodeMortonCode(center);
                    triangleIndices[i] = i;
                }

                std::sort(triangleIndices.begin(), triangleIndices.end(), [&mortons](uint32_t a, uint32_t b) {
                    return mortons[a] < mortons[b];
                });

                std::vector<uint64_t> sorted_mortons(mortons.size());
            
                for (int i = 0; i < sorted_mortons.size(); i++)
                {
                    sorted_mortons[i] = mortons[triangleIndices[i]];
                    sorted_triangles[i] = triangles[triangleIndices[i]];
                }

                std::vector<uint8_t> deltas(sorted_mortons.size() - 1);
                for (int i = 0; i < deltas.size(); i++)
                {
                    deltas[i] = minimum_lbvh::delta(mortons[i], mortons[i + 1]);
                }

                std::vector<minimum_lbvh::Stat> stats(mortons.size() - 1);
                for (int i = 0; i < stats.size(); i++)
                {
                    stats[i].oneOfEdges = 0xFFFFFFFF;
                }

                for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
                {
                    uint32_t leaf_lower = i_leaf;
                    uint32_t leaf_upper = i_leaf;
                    minimum_lbvh::NodeIndex node(i_leaf, true);

                    minimum_lbvh::AABB aabb; aabb.setEmpty();
                    for (auto v : sorted_triangles[i_leaf].vs)
                    {
                        aabb.extend(v);
                    }

                    bool isRoot = true;
                    while (leaf_upper - leaf_lower < internals.size())
                    {
                        // direction from bottom
                        int goLeft;
                        if (leaf_lower == 0)
                        {
                            goLeft = 0;
                        }
                        else if (leaf_upper == stats.size())
                        {
                            goLeft = 1;
                        }
                        else
                        {
                            goLeft = deltas[leaf_lower - 1] < deltas[leaf_upper] ? 1 : 0;
                        }

                        int parent = goLeft ? (leaf_lower - 1) : leaf_upper;

                        internals[parent].children[goLeft] = node;
                        internals[parent].aabbs[goLeft] = aabb;
                        if (!node.m_isLeaf)
                        {
                            internals[node.m_index].parent = minimum_lbvh::NodeIndex(parent, false);
                        }

                        uint32_t index = goLeft ? leaf_upper : leaf_lower;

                        // == memory barrier ==

                        std::swap(stats[parent].oneOfEdges, index);

                        if (index == 0xFFFFFFFF)
                        {
                            isRoot = false;
                            break;
                        }

                        leaf_lower = minimum_lbvh::ss_min(leaf_lower, index);
                        leaf_upper = minimum_lbvh::ss_max(leaf_upper, index);

                        node = minimum_lbvh::NodeIndex(parent, false);

                        minimum_lbvh::AABB otherAABB = internals[parent].aabbs[goLeft ^ 0x1];
                        aabb.extend(otherAABB);
                    }

                    if (isRoot)
                    {
                        rootNode = node;
                    }
                }

            }
        });

        int stride = 2;
        Image2DRGBA8 image;
        image.allocate(GetScreenWidth() / stride, GetScreenHeight() / stride);

        CameraRayGenerator rayGenerator(GetCurrentViewMatrix(), GetCurrentProjMatrix(), image.width(), image.height());

        for (int j = 0; j < image.height(); ++j)
        {
            for (int i = 0; i < image.width(); ++i)
            {
                glm::vec3 ro, rd;
                rayGenerator.shoot(&ro, &rd, i, j, 0.5f, 0.5f);

                Hit hit;
                intersect(&hit, internals.data(), sorted_triangles.data(), rootNode, to(ro), to(rd), invRd(to(rd)));
                if (hit.t != FLT_MAX)
                {
                    float3 n = normalize(hit.ng);
                    float3 color = (n + make_float3(1.0f)) * 0.5f;
                    image(i, j) = { 255 * color.x, 255 * color.y, 255 * color.z, 255 };
                }
                else
                {
                    image(i, j) = { 0, 0, 0, 255 };
                }
            }
        }

        texture->upload(image);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::InputInt("debug index", &g_debug_index);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
