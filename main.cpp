#include "pr.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <stack>

#include "minimum_lbvh.h"
#include <embree4/rtcore.h>

inline glm::vec3 to(float3 p)
{
    return { p.x, p.y, p.z };
}
inline float3 to(glm::vec3 p)
{
    return { p.x, p.y, p.z };
}

void printTree(const minimum_lbvh::InternalNode* nodes, minimum_lbvh::NodeIndex node)
{
    std::stringstream ss;
    ss << "digraph Tree {\n";
    ss << "    node [shape=circle];\n";

    uint32_t maxLeaf = 0;
    std::stack<minimum_lbvh::NodeIndex> stack;
    stack.push(node);
    while (!stack.empty())
    {
        minimum_lbvh::NodeIndex cur = stack.top();
        stack.pop();

        if (cur.m_isLeaf)
        {
            maxLeaf = minimum_lbvh::ss_max(maxLeaf, cur.m_index);
        }
        else
        {
            minimum_lbvh::NodeIndex L = nodes[cur.m_index].children[0];
            minimum_lbvh::NodeIndex R = nodes[cur.m_index].children[1];
            stack.push(L);
            stack.push(R);

            ss << "    " << cur.m_index << " -> " << (L.m_isLeaf ? "L" : "") << L.m_index << "\n";
            ss << "    " << cur.m_index << " -> " << (R.m_isLeaf ? "L" : "") << R.m_index << "\n";
        }
    }
    ss << "    { rank = same; ";
    for (uint32_t i_leaf = 0; i_leaf <= maxLeaf; i_leaf++)
    {
        ss << "L" << i_leaf << "; ";
    }
    ss << "}\n";

    ss << "}\n";
    printf("%s", ss.str().c_str());
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

    minimum_lbvh::NodeIndex rootNode;
    std::vector<minimum_lbvh::InternalNode> internals;
    internals.resize(mortons.size() - 1);

    for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
    {
        uint32_t leaf_lower = i_leaf;
        uint32_t leaf_upper = i_leaf;
        minimum_lbvh::NodeIndex node(i_leaf, true);

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

            uint32_t index = goLeft ? leaf_upper : leaf_lower;
            std::swap(stats[parent].oneOfEdges, index);

            if (index == 0xFFFFFFFF)
            {
                isRoot = false;
                break;
            }

            leaf_lower = minimum_lbvh::ss_min(leaf_lower, index);
            leaf_upper = minimum_lbvh::ss_max(leaf_upper, index);

            node = minimum_lbvh::NodeIndex(parent, false);
        }

        if (isRoot)
        {
            rootNode = node;
        }
    }

    printTree(internals.data(), rootNode);

    for (int i = 0; i < internals.size(); i++)
    {
        minimum_lbvh::NodeIndex node(i, false /*is leaf*/);

        int leaf_lower = INT_MAX;
        int leaf_upper = -1;
        std::stack<minimum_lbvh::NodeIndex> stack;
        stack.push(node);
        while (!stack.empty())
        {
            minimum_lbvh::NodeIndex cur = stack.top();
            stack.pop();

            if (cur.m_isLeaf)
            {
                leaf_lower = minimum_lbvh::ss_min(leaf_lower, (int)cur.m_index);
                leaf_upper = minimum_lbvh::ss_max(leaf_upper, (int)cur.m_index);
            }
            else
            {
                stack.push(internals[cur.m_index].children[0]);
                stack.push(internals[cur.m_index].children[1]);
            }
        }

        int maxDelta = deltas[i];

        // for all delta under the node
        for (int j = leaf_lower; j < leaf_upper; j++)
        {
            if (maxDelta < deltas[j] )
            {
                abort();
            }
        }
    }
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
    float2 uv = {};
    float3 ng = {};
    uint32_t triangleIndex = 0xFFFFFFFF;
};

inline float3 invRd(float3 rd)
{
    return clamp(1.0f / rd, -FLT_MAX, FLT_MAX);
}
void intersect(
    Hit* hit, 
    const minimum_lbvh::InternalNode* nodes, 
    const minimum_lbvh::Triangle* triangles, 
    minimum_lbvh::NodeIndex rootNode, 
    float3 ro, 
    float3 rd,
    float3 one_over_rd)
{
    int sp = 1;
    minimum_lbvh::NodeIndex stack[64];
    stack[0] = rootNode;

    while (sp)
    {
        minimum_lbvh::NodeIndex node = stack[--sp];

        if (node.m_isLeaf)
        {
            float t;
            float u, v;
            float3 ng;
            const minimum_lbvh::Triangle& tri = triangles[node.m_index];
            if (minimum_lbvh::intersect_ray_triangle(&t, &u, &v, &ng, 0.0f, hit->t, ro, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
            {
                hit->t = t;
                hit->uv = make_float2(u, v);
                hit->ng = ng;
                hit->triangleIndex = node.m_index;
            }
            continue;
        }

        const minimum_lbvh::AABB& L = nodes[node.m_index].aabbs[0];
        const minimum_lbvh::AABB& R = nodes[node.m_index].aabbs[1];

        float2 rangeL = minimum_lbvh::slabs(ro, one_over_rd, L.lower, L.upper);
        float2 rangeR = minimum_lbvh::slabs(ro, one_over_rd, R.lower, R.upper);
        bool hitL = rangeL.x <= rangeL.y;
        bool hitR = rangeR.x <= rangeR.y;

        if (hitL && hitR)
        {
            int mask = 0x0;
            if (rangeR.x < rangeL.x)
            {
                mask = 0x1;
            }
            stack[sp++] = nodes[node.m_index].children[1 ^ mask];
            stack[sp++] = nodes[node.m_index].children[0 ^ mask];
        }
        else if (hitL || hitR)
        {
            stack[sp++] = nodes[node.m_index].children[hitL ? 0 : 1];
        }
    }
}

void intersect_stackfree(
    Hit* hit,
    const minimum_lbvh::InternalNode* nodes,
    const minimum_lbvh::Triangle* triangles,
    minimum_lbvh::NodeIndex node,
    float3 ro,
    float3 rd,
    float3 one_over_rd)
{
    bool decent = true;
    minimum_lbvh::NodeIndex curr_node = node;
    minimum_lbvh::NodeIndex prev_node(0x7FFFFFFF, false);

    while(curr_node.m_index != 0x7FFFFFFF)
    {
        if (curr_node.m_isLeaf)
        {
            float t;
            float u, v;
            float3 ng;
            const minimum_lbvh::Triangle& tri = triangles[curr_node.m_index];
            if (minimum_lbvh::intersect_ray_triangle(&t, &u, &v, &ng, 0.0f, hit->t, ro, rd, tri.vs[0], tri.vs[1], tri.vs[2]))
            {
                hit->t = t;
                hit->uv = make_float2(u, v);
                hit->triangleIndex = curr_node.m_index;
                hit->ng = ng;
            }

            std::swap(curr_node, prev_node);
            decent = false;
            continue;
        }

        minimum_lbvh::AABB L = nodes[curr_node.m_index].aabbs[0];
        minimum_lbvh::AABB R = nodes[curr_node.m_index].aabbs[1];
        float2 rangeL = minimum_lbvh::slabs(ro, one_over_rd, L.lower, L.upper);
        float2 rangeR = minimum_lbvh::slabs(ro, one_over_rd, R.lower, R.upper);
        bool hitL = rangeL.x <= rangeL.y;
        bool hitR = rangeR.x <= rangeR.y;

        minimum_lbvh::NodeIndex parent_node = nodes[curr_node.m_index].parent;
        minimum_lbvh::NodeIndex near_node = nodes[curr_node.m_index].children[0];
        minimum_lbvh::NodeIndex far_node  = nodes[curr_node.m_index].children[1];

        int nHits = 0;
        if (hitL && hitR)
        {
            if (rangeR.x < rangeL.x)
            {
                std::swap(near_node, far_node);
            }
            nHits = 2;
        }
        else if (hitL || hitR)
        {
            nHits = 1;
            if (hitR)
            {
                std::swap(near_node, far_node);
            }
        }

        minimum_lbvh::NodeIndex next_node;
        if (nHits == 0)
        {
            next_node = parent_node;
        }
        else if (decent)
        {
            next_node = near_node;
        }
        else if (prev_node == near_node)
        {
            next_node = nHits == 2 ? far_node : parent_node;
        }
        else
        {
            next_node = parent_node;
        }

        decent = !(next_node == parent_node);
        prev_node = curr_node;
        curr_node = next_node;
    }
}




// embree
struct EmbreeBVHContext
{
    EmbreeBVHContext() {
        nodes = 0;
        nodeHead = 0;
    }
    minimum_lbvh::InternalNode* nodes;
    std::atomic<int> nodeHead;
};

inline void* node2ptr(minimum_lbvh::NodeIndex node)
{
    uint32_t data;
    memcpy(&data, &node, sizeof(uint32_t));
    return (char*)0 + data;
}
inline minimum_lbvh::NodeIndex ptr2node(void* ptr)
{
    uint32_t data = (char*)ptr - (char*)0;
    minimum_lbvh::NodeIndex node;
    memcpy(&node, &data, sizeof(uint32_t));
    return node;
}

static void* embrreeCreateNode(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr)
{
    PR_ASSERT(numChildren == 2);

    EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
    int index = context->nodeHead++;

    minimum_lbvh::NodeIndex node(index, false);
    return node2ptr(node);
}
static void embreeSetNodeChildren(void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr)
{
    PR_ASSERT(numChildren == 2);

    EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
    minimum_lbvh::InternalNode& node = context->nodes[ptr2node(nodePtr).m_index];
    for (int i = 0; i < numChildren; i++)
    {
        node.children[i] = ptr2node(childPtr[i]);
    }
}
static void embreeSetNodeBounds(void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr)
{
    PR_ASSERT(numChildren == 2);

    EmbreeBVHContext* context = (EmbreeBVHContext*)userPtr;
    minimum_lbvh::InternalNode& node = context->nodes[ptr2node(nodePtr).m_index];
    
    for (int i = 0; i < numChildren; i++)
    {
        node.aabbs[i].lower = make_float3(bounds[i]->lower_x, bounds[i]->lower_y, bounds[i]->lower_z);
        node.aabbs[i].upper = make_float3(bounds[i]->upper_x, bounds[i]->upper_y, bounds[i]->upper_z);
    }
}
static void* embreeCreateLeaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr)
{
    PR_ASSERT(numPrims == 1);
    minimum_lbvh::NodeIndex node(prims->primID, true /*is leaf*/);
    return node2ptr(node);
}

struct TriangleAttrib
{
    float3 shadingNormals[3];
};

int main() {
    using namespace pr;

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

    runToyExample();


    // BVH
    minimum_lbvh::NodeIndex rootNode;
    std::vector<minimum_lbvh::InternalNode> internals;
    std::vector<minimum_lbvh::Triangle> triangles;
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

            if (internals.empty())
            {
#if 1
                internals.clear();
                internals.resize(triangles.size() - 1);

                std::vector<uint64_t> mortons(triangles.size());
                std::vector<uint32_t> sortedTriangleIndices(triangles.size());
                std::vector<minimum_lbvh::Stat> stats(mortons.size() - 1);

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

                for (int i = 0; i < triangles.size(); i++)
                {
                    minimum_lbvh::Triangle tri = triangles[i];
                    float3 center = (tri.vs[0] + tri.vs[1] + tri.vs[2]) / 3.0f;
                    mortons[i] = sceneAABB.encodeMortonCode(center);
                    sortedTriangleIndices[i] = i;

                    // DrawSphere(to(center), 0.04f, { 255, 0, 0 });
                }

                std::sort(sortedTriangleIndices.begin(), sortedTriangleIndices.end(), [&mortons](uint32_t a, uint32_t b) {
                    return mortons[a] < mortons[b];
                });

                std::vector<uint8_t> deltas(mortons.size() - 1);
                for (int i = 0; i < deltas.size(); i++)
                {
                    uint64_t mA = mortons[sortedTriangleIndices[i]];
                    uint64_t mB = mortons[sortedTriangleIndices[i + 1]];
                    deltas[i] = minimum_lbvh::delta(mA, mB);
                }
               
                for (int i = 0; i < stats.size(); i++)
                {
                    stats[i].oneOfEdges = 0xFFFFFFFF;
                }

                Stopwatch sw;
#if 0
                for (uint32_t i_leaf = 0; i_leaf < mortons.size(); i_leaf++)
                {
                    minimum_lbvh::build_lbvh(
                        &rootNode,
                        internals.data(),
                        stats.data(),
                        triangles.data(),
                        triangles.size(),
                        sortedTriangleIndices.data(),
                        deltas.data(),
                        sceneAABB,
                        i_leaf
                    );
                }
#else
                ParallelFor(mortons.size(), [&](int i_leaf) {
                    minimum_lbvh::build_lbvh(
                        &rootNode,
                        internals.data(),
                        stats.data(),
                        triangles.data(),
                        triangles.size(),
                        sortedTriangleIndices.data(),
                        deltas.data(),
                        sceneAABB,
                        i_leaf
                    );
                });
#endif
                printf("build %f\n", sw.elapsed());

#if 1
                // Validation - delta under the internal node must less or eq of the split.
                for (int i = 0; i < internals.size(); i++)
                {
                    minimum_lbvh::NodeIndex node(i, false /*is leaf*/);

                    std::vector<uint64_t> allMortons;
                    std::stack<minimum_lbvh::NodeIndex> stack;
                    stack.push(node);
                    while (!stack.empty())
                    {
                        minimum_lbvh::NodeIndex cur = stack.top();
                        stack.pop();

                        if (cur.m_isLeaf)
                        {
                            minimum_lbvh::Triangle tri = triangles[cur.m_index];
                            float3 center = (tri.vs[0] + tri.vs[1] + tri.vs[2]) / 3.0f;
                            uint64_t morton = sceneAABB.encodeMortonCode(center);
                            allMortons.push_back(morton);
                        }
                        else
                        {
                            stack.push(internals[cur.m_index].children[0]);
                            stack.push(internals[cur.m_index].children[1]);
                        }
                    }

                    int maxDelta = deltas[i];

                    // for all delta under the node
                    for (int j = 0; j < allMortons.size() - 1; j++)
                    {
                        if (maxDelta < minimum_lbvh::delta(allMortons[j], allMortons[j + 1]))
                        {
                            abort();
                        }
                    }
                }
#endif
                // triangles
#else
                RTCDevice device = rtcNewDevice("");
                RTCBVH bvh = rtcNewBVH(device);

                rtcSetDeviceErrorFunction(device, [](void* userPtr, RTCError code, const char* str) {
                    printf("Embree Error [%d] %s\n", code, str);
                }, 0);

                std::vector<RTCBuildPrimitive> primitives(triangles.size());
                for (int i = 0; i < triangles.size(); i++)
                {
                    minimum_lbvh::AABB aabb; aabb.setEmpty();
                    for (auto v : triangles[i].vs)
                    {
                        aabb.extend(v);
                    }
                    RTCBuildPrimitive prim = {};
                    prim.lower_x = aabb.lower.x;
                    prim.lower_y = aabb.lower.y;
                    prim.lower_z = aabb.lower.z;
                    prim.geomID = 0;
                    prim.upper_x = aabb.upper.x;
                    prim.upper_y = aabb.upper.y;
                    prim.upper_z = aabb.upper.z;
                    prim.primID = i;
                    primitives[i] = prim;
                }
                
                // allocation
                internals.resize(triangles.size() - 1);

                EmbreeBVHContext context;
                context.nodes = internals.data();

                RTCBuildArguments arguments = rtcDefaultBuildArguments();
                arguments.maxDepth = 64;
                arguments.byteSize = sizeof(arguments);
                arguments.buildQuality = RTC_BUILD_QUALITY_LOW;
                arguments.maxBranchingFactor = 2;
                arguments.bvh = bvh;
                arguments.primitives = primitives.data();
                arguments.primitiveCount = primitives.size();
                arguments.primitiveArrayCapacity = primitives.size();
                arguments.minLeafSize = 1;
                arguments.maxLeafSize = 1;
                arguments.createNode = embrreeCreateNode;
                arguments.setNodeChildren = embreeSetNodeChildren;
                arguments.setNodeBounds = embreeSetNodeBounds;
                arguments.createLeaf = embreeCreateLeaf;
                arguments.splitPrimitive = nullptr;
                arguments.userPtr = &context;
                void* bvh_root = rtcBuildBVH(&arguments);

                rtcReleaseBVH(bvh);
                rtcReleaseDevice(device);

                rootNode = ptr2node(bvh_root);
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

                Hit hit;
                intersect(&hit, internals.data(), triangles.data(), rootNode, to(ro), to(rd), invRd(to(rd)));
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
        ImGui::InputInt("debug index", &g_debug_index);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
