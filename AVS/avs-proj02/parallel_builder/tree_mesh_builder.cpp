/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  FULL NAME <xkaspa48@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    unsigned totalTriangles = 0;
    #pragma omp parallel default(none) shared(totalTriangles, field)
    #pragma omp single nowait
    totalTriangles = breakSpace(Vec3_t<float>(), field, mGridSize);
    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());
    float value = std::numeric_limits<float>::max();
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        value = std::min(value, distanceSquared);
    }

    return sqrtf(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle) {
    #pragma omp critical
    mTriangles.push_back(triangle);
}

unsigned TreeMeshBuilder::breakSpace(const Vec3_t<float> &pos, const ParametricScalarField &field, const unsigned int gridSize) {
    if (isEmpty(pos, field, gridSize)) {
        return 0;
    }
    if (gridSize == 1) {
        return buildCube(pos, field);
    }
    unsigned totalTriangles = 0;
    const unsigned int newGridSize = gridSize / 2;
    for (int i = 0; i < 8; i++) {
        #pragma omp task default(none) firstprivate(i) shared(pos, field, newGridSize, totalTriangles)
        {
            Vec3_t<float> newCubeOffset(
                pos.x + (i % 2) * newGridSize,
                pos.y + int(((i / 2) % 2)) * newGridSize,
                pos.z + int(i / 4) * newGridSize
            );
            unsigned int triangles = breakSpace(newCubeOffset, field, newGridSize);
            #pragma omp atomic update
            totalTriangles += triangles;
        }
    }
    #pragma omp taskwait
	return totalTriangles;
}

bool TreeMeshBuilder::isEmpty(const Vec3_t<float> &pos, const ParametricScalarField &field, const unsigned int gridSize) {
    float edgeLength = gridSize * mGridResolution;
    Vec3_t<float> point (
        pos.x * mGridResolution + (edgeLength / 2.0f),
        pos.y * mGridResolution + (edgeLength / 2.0f),
        pos.z * mGridResolution + (edgeLength / 2.0f)
    );
    return evaluateFieldAt(point, field) > mIsoLevel + (sqrtf(3.0f) / 2.0f) * edgeLength;
}
