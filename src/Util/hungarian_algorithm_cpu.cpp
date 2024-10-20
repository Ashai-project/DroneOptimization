#include "hungarian_algorithm_cpu.h"
#include <algorithm>
#include <limits>
#include <iostream>

HungarianAlgorithmCPU::HungarianAlgorithmCPU() {}

// 行列から最適な割り当てを見つける
std::vector<int> HungarianAlgorithmCPU::findOptimalAssignment(const std::vector<std::vector<float>>& costMatrix) {
    int rows = costMatrix.size();
    int cols = costMatrix[0].size();
    int newSize = std::max(rows, cols);

    // コスト行列の正方行列化（ダミー要素を追加）
    std::vector<std::vector<float>> matrix(newSize, std::vector<float>(newSize, std::numeric_limits<float>::max()));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = costMatrix[i][j];
        }
    }

    // 各行の最小値を引く
    for (int i = 0; i < newSize; ++i) {
        float rowMin = *std::min_element(matrix[i].begin(), matrix[i].end());
        for (int j = 0; j < newSize; ++j) {
            matrix[i][j] -= rowMin;
        }
    }

    // 各列の最小値を引く
    for (int j = 0; j < newSize; ++j) {
        float colMin = std::numeric_limits<float>::max();
        for (int i = 0; i < newSize; ++i) {
            colMin = std::min(colMin, matrix[i][j]);
        }
        for (int i = 0; i < newSize; ++i) {
            matrix[i][j] -= colMin;
        }
    }

    std::vector<int> rowAssignment(newSize, -1);
    std::vector<int> colAssignment(newSize, -1);
    std::vector<bool> rowCovered(newSize, false);
    std::vector<bool> colCovered(newSize, false);
    bool optimalFound = false;

    while (!optimalFound) {
        bool newAssignment = false;

        // 割り当て可能なゼロを見つける
        for (int i = 0; i < newSize; ++i) {
            if (!rowCovered[i]) {
                for (int j = 0; j < newSize; ++j) {
                    if (matrix[i][j] == 0 && !colCovered[j]) {
                        rowAssignment[i] = j;
                        colAssignment[j] = i;
                        rowCovered[i] = true;
                        colCovered[j] = true;
                        newAssignment = true;
                        break;
                    }
                }
            }
        }

        // 割り当てが完了しているか確認
        if (std::count(rowAssignment.begin(), rowAssignment.end(), -1) == 0) {
            optimalFound = true;
        }

        if (!newAssignment) {
            // カバーされていない行と列の最小値を探す
            float minVal = std::numeric_limits<float>::max();
            for (int i = 0; i < newSize; ++i) {
                if (!rowCovered[i]) {
                    for (int j = 0; j < newSize; ++j) {
                        if (!colCovered[j]) {
                            minVal = std::min(minVal, matrix[i][j]);
                        }
                    }
                }
            }

            // カバーされていない行から最小値を引き、カバーされている列に最小値を加える
            for (int i = 0; i < newSize; ++i) {
                if (!rowCovered[i]) {
                    for (int j = 0; j < newSize; ++j) {
                        matrix[i][j] -= minVal;
                    }
                }
            }

            for (int j = 0; j < newSize; ++j) {
                if (colCovered[j]) {
                    for (int i = 0; i < newSize; ++i) {
                        matrix[i][j] += minVal;
                    }
                }
            }

            // カバー解除を行い、再度探索を行う
            std::fill(rowCovered.begin(), rowCovered.end(), false);
            std::fill(colCovered.begin(), colCovered.end(), false);
        }
    }

    return rowAssignment;
}

// コストを計算する
float HungarianAlgorithmCPU::calculateCost(const std::vector<int>& assignment, const std::vector<std::vector<float>>& costMatrix) {
    float totalCost = 0.0f;
    for (int i = 0; i < assignment.size(); ++i) {
        if (assignment[i] != -1 && i < costMatrix.size() && assignment[i] < costMatrix[0].size()) {
            totalCost += costMatrix[i][assignment[i]];
        }
    }
    return totalCost;
}
