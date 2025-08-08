#include "pr.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include "minimum_lbvh.h"

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

    std::vector<uint32_t> stats(mortons.size() - 1);
    for (int i = 0; i < stats.size(); i++)
    {
        stats[i] = 0xFFFFFFFF;
    }

    std::vector<minimum_lbvh::LBVHNode> internals;
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
            int isLeft;
            if (leaf_lower == 0)
            {
                isLeft = 0;
            }
            else if (leaf_upper == stats.size())
            {
                isLeft = 1;
            }
            else
            {
                isLeft = deltas[leaf_lower - 1] < deltas[leaf_upper] ? 1 : 0;
            }

            int parent = isLeft ? (leaf_lower - 1) : leaf_upper;

            internals[parent].children[isLeft] = node;

            ss << "    " << parent << " -> " << (node.m_isLeaf ? "L" : "") << node.m_index << "\n";

            uint32_t index = isLeft ? leaf_upper : leaf_lower;
            std::swap(stats[parent], index);

            if (index == 0xFFFFFFFF)
            {
                break;
            }

            if (leaf_upper < index)
            {
                leaf_upper = index;
            }
            else
            {
                leaf_lower = index;
            }

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

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

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

                minimum_lbvh::Vec3 e0 = tri.vs[1] - tri.vs[0];
                minimum_lbvh::Vec3 e1 = tri.vs[2] - tri.vs[1];
                minimum_lbvh::Vec3 e2 = tri.vs[0] - tri.vs[2];

                //tri.ng = cross(e0, e1);

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
                        minimum_lbvh::Vec3 v0 = tri.vs[j];
                        minimum_lbvh::Vec3 v1 = tri.vs[(j + 1) % 3];
                        pr::PrimVertex({ v0.xs[0], v0.xs[1], v0.xs[2] }, { 255, 255, 255 });
                        pr::PrimVertex({ v1.xs[0], v1.xs[1], v1.xs[2] }, { 255, 255, 255 });
                    }
                }

                pr::PrimEnd();
            }

            std::vector<uint64_t> mortons;
        });

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
