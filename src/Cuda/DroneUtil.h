#ifndef DRONE_UTIL_H
#define DRONE_UTIL_H

#include <vector>
#include <cmath>
#include <cfloat>
#include <chrono> 
#include <algorithm>
#include <cuda_runtime.h>
#include "GeneticAlgorithmPARA.h"
#include "../Util/hungarian_algorithm_cpu.h"
#include "../Util/hungarian_algorithm_para.h"
#include "../Util/GeneticAlgorithm.h"

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    float distance(const Vec3& other) const;
    Vec3 interpolate(const Vec3& target, float t) const;
};


class DroneUtil {
public:
    enum class AlgorithmType { HUNGARIAN, GA, HUNGARIAN_CPU, GA_PARA};
    DroneUtil(int vertexCountA, int vertexCountB, float radius, AlgorithmType algorithmType = AlgorithmType::HUNGARIAN);
    
    // モデルの初期化
    void initializeModels();
    double initializeModels(int seed);

    // 頂点の対応付けを最適化
    double optimizeVertexMapping();

    // 対応付けを評価
    float evaluateMapping();

    const Vec3& getModelA(int index) const;

    const Vec3& getModelB(int index) const;

    // 頂点リストと対応付け結果を取得
    const std::vector<int>& getVertexMapping() const;

    bool checkForDuplicateMappings();

private:
    AlgorithmType algorithmType;
    std::vector<Vec3> modelA, modelB;  // モデルの頂点リスト
    std::vector<int> vertexMapping;    // 頂点対応付け結果
    int droneCount;                    // ドローン数（頂点数）
    int vertexCountA, vertexCountB;    // モデルAとBの頂点数
    float radius;                      // モデルAの球面の半径
};

#endif // DRONE_UTIL_H
