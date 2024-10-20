#ifndef HUNGARIAN_ALGORITHM_CPU_H
#define HUNGARIAN_ALGORITHM_CPU_H

#include <vector>

class HungarianAlgorithmCPU {
public:
    // コンストラクタ
    HungarianAlgorithmCPU();

    // 行列から最適な割り当てを見つける
    std::vector<int> findOptimalAssignment(const std::vector<std::vector<float>>& costMatrix);

    // コストを計算する
    float calculateCost(const std::vector<int>& assignment, const std::vector<std::vector<float>>& costMatrix);
};

#endif // HUNGARIAN_ALGORITHM_CPU_H
