#include "DroneUtil.h"

// Vec3の距離計算
float Vec3::distance(const Vec3& other) const {
    return std::sqrt((x - other.x) * (x - other.x) +
                     (y - other.y) * (y - other.y) +
                     (z - other.z) * (z - other.z));
}

// Vec3の線形補間
Vec3 Vec3::interpolate(const Vec3& target, float t) const {
    return Vec3(x + t * (target.x - x), y + t * (target.y - y), z + t * (target.z - z));
}

// コンストラクタ
DroneUtil::DroneUtil(int vertexCountA, int vertexCountB, float radius)
    : vertexCountA(vertexCountA), vertexCountB(vertexCountB), radius(radius), droneCount(0) {}

// モデルの初期化
void DroneUtil::initializeModels() {
    srand(100);  // ランダムシードの初期化

    // モデルAの頂点を生成
    for (int i = 0; i < vertexCountA; ++i) {
        float phi = acos(1 - 2 * (i + 0.5f) / vertexCountA);
        float theta = M_PI * (1 + sqrt(5)) * i;

        float x = radius * sin(phi) * cos(theta);
        float y = radius * sin(phi) * sin(theta);
        float z = radius * cos(phi);

        modelA.push_back(Vec3(x, y, z));
    }

    // モデルBの頂点を生成
    for (int i = 0; i < vertexCountB; ++i) {
        float x = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        float y = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        float z = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        modelB.push_back(Vec3(x, y, z));
    }

    droneCount = std::max(modelA.size(), modelB.size());
    optimizeVertexMapping();

    float totalDistance = evaluateMapping();
    std::cout << "総移動距離: " << totalDistance << std::endl;
}

// 頂点の対応付けを最適化
void DroneUtil::optimizeVertexMapping() {
    HungarianAlgorithm hungarianAlgorithmInstance;
    vertexMapping.resize(droneCount);
    int newSize = std::max(vertexCountA, vertexCountB);

    std::vector<float> costMatrix(newSize * newSize, FLT_MAX);

    for (int i = 0; i < vertexCountA; ++i) {
        for (int j = 0; j < vertexCountB; ++j) {
            costMatrix[i * newSize + j] = modelA[i].distance(modelB[j]);
        }
    }

    hungarianAlgorithmInstance.addDummyElements(costMatrix.data(), vertexCountA, vertexCountB, newSize);

    float *d_matrix, *d_minRow, *d_minCol;
    cudaMalloc((void**)&d_matrix, newSize * newSize * sizeof(float));
    cudaMalloc((void**)&d_minRow, newSize * sizeof(float));
    cudaMalloc((void**)&d_minCol, newSize * sizeof(float));

    cudaMemcpy(d_matrix, costMatrix.data(), newSize * newSize * sizeof(float), cudaMemcpyHostToDevice);
    hungarianAlgorithmInstance.runHungarianAlgorithm(d_matrix, d_minRow, d_minCol, newSize);
    cudaMemcpy(costMatrix.data(), d_matrix, newSize * newSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<int> assignments;
    hungarianAlgorithmInstance.findOptimalAssignment(costMatrix.data(), newSize, newSize, assignments);

    for (int i = 0; i < vertexCountA; ++i) {
        vertexMapping[i] = assignments[i];
    }

    for (int i = vertexCountA; i < droneCount; ++i) {
        vertexMapping[i] = i % vertexCountB;
    }

    cudaFree(d_matrix);
    cudaFree(d_minRow);
    cudaFree(d_minCol);
}

// 対応付けを評価
float DroneUtil::evaluateMapping() {
    float totalDistance = 0.0f;
    for (int i = 0; i < modelA.size(); ++i) {
        totalDistance += modelA[i].distance(modelB[vertexMapping[i]]);
    }
    return totalDistance;
}

const Vec3& DroneUtil::getModelA(int index) const {
    return modelA[index];
}

const Vec3& DroneUtil::getModelB(int index) const {
    return modelB[index];
}

// 頂点対応付け結果の取得
const std::vector<int>& DroneUtil::getVertexMapping() const {
    return vertexMapping;
}
