#include "../src/Util/hungarian_algorithm_cpu.h"
#include <iostream>

int main() {
    HungarianAlgorithmCPU solver;

    // コスト行列の定義
    std::vector<std::vector<float>> costMatrix = {
        {9.0f, 11.0f, 14.0f, 11.0f, 7.0f},
        {6.0f, 15.0f, 13.0f, 13.0f, 10.0f},
        {12.0f, 13.0f, 6.0f, 8.0f, 8.0f},
        {11.0f, 9.0f, 10.0f, 12.0f, 9.0f},
        {7.0f, 12.0f, 14.0f, 10.0f, 14.0f}
    };

    // 最適な割り当ての計算
    std::vector<int> assignment = solver.findOptimalAssignment(costMatrix);

    // 結果の表示
    std::cout << "Optimal assignment:" << std::endl;
    for (size_t i = 0; i < assignment.size(); ++i) {
        std::cout << "Worker " << i << " assigned to job " << assignment[i] << std::endl;
    }

    // 総コストの計算
    float totalCost = solver.calculateCost(assignment, costMatrix);
    std::cout << "Total cost: " << totalCost << std::endl;

    return 0;
}
