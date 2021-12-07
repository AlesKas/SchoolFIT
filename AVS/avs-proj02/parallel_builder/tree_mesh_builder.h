/**
 * @file    tree_mesh_builder.h
 *
 * @author  FULL NAME <xkaspa48@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    DATE
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }
    std::vector<Triangle_t> mTriangles;
    unsigned breakSpace(const Vec3_t<float> &pos, const ParametricScalarField &field, const unsigned int gridSize);
    bool isEmpty(const Vec3_t<float> &pos, const ParametricScalarField &field, const unsigned int gridSize);
};

#endif // TREE_MESH_BUILDER_H
