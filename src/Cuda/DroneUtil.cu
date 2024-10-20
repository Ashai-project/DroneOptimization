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
DroneUtil::DroneUtil(int vertexCountA, int vertexCountB, float radius, AlgorithmType algorithmType)
    : vertexCountA(vertexCountA), vertexCountB(vertexCountB), radius(radius), algorithmType(algorithmType), droneCount(0) {}

// モデルの初期化
void DroneUtil::initializeModels() {
    initializeModels(100);
}

double DroneUtil::initializeModels(int seed) {
    srand(seed);  // ランダムシードの初期化

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

    double elapsedMillTime = optimizeVertexMapping();

    float totalDistance = evaluateMapping();
    std::cout << "総移動距離: " << totalDistance << std::endl;
    return elapsedMillTime;
}

// 頂点の対応付けを最適化
double DroneUtil::optimizeVertexMapping() {
    int newSize = std::max(vertexCountA, vertexCountB);
    vertexMapping.resize(droneCount);
    std::vector<float> costMatrix(newSize * newSize, FLT_MAX);

    // コスト行列の作成
    for (int i = 0; i < vertexCountA; ++i) {
        for (int j = 0; j < vertexCountB; ++j) {
            costMatrix[i * newSize + j] = modelA[i].distance(modelB[j]);
        }
    }

    // アルゴリズムの選択に応じて処理を切り替える
    auto start = std::chrono::high_resolution_clock::now();
    
    if (algorithmType == AlgorithmType::HUNGARIAN) {
        HungarianAlgorithm hungarianAlgorithmInstance;

        // ダミー要素を追加
        hungarianAlgorithmInstance.addDummyElements(costMatrix.data(), vertexCountA, vertexCountB, newSize);

        std::vector<std::vector<float>> costMatrix2D(newSize, std::vector<float>(newSize, FLT_MAX));
        for (int i = 0; i < vertexCountA; ++i) {
            for (int j = 0; j < vertexCountB; ++j) {
                costMatrix2D[i][j] = costMatrix[i * newSize + j];
            }
        }

        std::vector<int> assignments = hungarianAlgorithmInstance.findOptimalAssignment(costMatrix2D);

        for (int i = 0; i < vertexCountA; ++i) {
            vertexMapping[i] = assignments[i];
        }

        for (int i = vertexCountA; i < droneCount; ++i) {
            vertexMapping[i] = i % vertexCountB;
        }

    } else if (algorithmType == AlgorithmType::GA) {
        std::vector<std::vector<float>> costMatrix2D(newSize, std::vector<float>(newSize, FLT_MAX));
        for (int i = 0; i < vertexCountA; ++i) {
            for (int j = 0; j < vertexCountB; ++j) {
                costMatrix2D[i][j] = costMatrix[i * newSize + j];
            }
        }

        // 遺伝的アルゴリズムによる最適化
        GeneticAlgorithm ga(10 * newSize, 500, 0.15, newSize, costMatrix2D);
        vertexMapping = ga.optimize();
    } else if (algorithmType == AlgorithmType::HUNGARIAN_CPU) {
        std::vector<std::vector<float>> costMatrix2D(newSize, std::vector<float>(newSize, FLT_MAX));
        for (int i = 0; i < vertexCountA; ++i) {
            for (int j = 0; j < vertexCountB; ++j) {
                costMatrix2D[i][j] = costMatrix[i * newSize + j];
            }
        }

        HungarianAlgorithmCPU hungarianCPUAlgorithmInstance;
        std::vector<int> assignments = hungarianCPUAlgorithmInstance.findOptimalAssignment(costMatrix2D);

        for (int i = 0; i < vertexCountA; ++i) {
            vertexMapping[i] = assignments[i];
        }

        for (int i = vertexCountA; i < droneCount; ++i) {
            vertexMapping[i] = i % vertexCountB;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    return elapsed.count();
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
