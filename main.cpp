#include "pr.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include "minimum_lbvh.h"

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

    double e = GetElapsedTime();

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
        deltas[i] = min_lbvh::delta(mortons[i], mortons[i + 1]);
    }

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        float digit_h = 0.1f;
        for (int i = 0; i < mortons.size(); i++)
        {
            float x = min_lbvh::remap(i, 0, mortons.size() - 1, -1, 1);
            uint32_t code = mortons[i];

            for (int d = 0; d < 4; d++)
            {
                char* digit = ((code >> d) & 0x1) ? "1" : "0";
                DrawText({ x, -digit_h * 4 + digit_h * d, 0 }, std::string(digit));
            }
        }

        for (int i = 0; i < mortons.size() - 1; i++)
        {
            float x0 = min_lbvh::remap(i,   0, mortons.size() - 1, -1, 1);
            float x1 = min_lbvh::remap(i+1, 0, mortons.size() - 1, -1, 1);

            int d = deltas[i];
            float y = -digit_h * 5 + digit_h * d;
            DrawLine({ x0, y, 0.0f }, { x1, y, 0.0f }, { 255, 0, 0 }, 3);
        }

        std::vector<uint32_t> stats(mortons.size() - 1);
        for (int i = 0; i < stats.size(); i++)
        {
            stats[i] = 0xFFFFFFFF;
        }

        std::vector<LBVHNode> internals;
        internals.resize(mortons.size() - 1);

        std::stringstream ss;
        ss << "digraph Tree {\n";
        ss << "    node [shape=circle];\n";

        for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
        {
            uint32_t leaf_lower = i_leaf;
            uint32_t leaf_upper = i_leaf;
            NodeIndex node(i_leaf, true);

            while(leaf_upper - leaf_lower < internals.size())
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

                node = NodeIndex(parent, false);
            }
        }

        ss << "    { rank = same; ";
        for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
        {
            ss << "L" << i_leaf << "; ";
        }
        ss << "}\n";

        ss << "}\n";
        // printf("%s", ss.str().c_str());

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
