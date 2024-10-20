#ifndef HUNGARIAN_ALGORITHM_H
#define HUNGARIAN_ALGORITHM_H

#include <vector>
#include <cfloat>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <iostream>

class HungarianAlgorithm {
public:
    HungarianAlgorithm();
    ~HungarianAlgorithm();
    
    void addDummyElements(float* matrix, int rows, int cols, int newSize);
    // 最適割り当てを見つけるメソッド
    std::vector<int> findOptimalAssignment(const std::vector<std::vector<float>>& costMatrix);
    
    // コストを計算する
    float calculateCost(const std::vector<int>& assignment, const std::vector<std::vector<float>>& costMatrix);

private:
    float* d_matrix;   // デバイス側の行列
    float* d_minRow;   // 行ごとの最小値を保存する配列
    float* d_minCol;   // 列ごとの最小値を保存する配列
    float* d_minVal;   // 全体の最小値

    void subtractRowMin(float* matrix, int rows, int cols);   // 行の最小値を引く関数
    void subtractColMin(float* matrix, int rows, int cols);   // 列の最小値を引く関数
    void updateMatrixForUncovered(float* matrix, bool* rowCovered, bool* colCovered, int rows, int cols);
};

#endif // HUNGARIAN_ALGORITHM_H
