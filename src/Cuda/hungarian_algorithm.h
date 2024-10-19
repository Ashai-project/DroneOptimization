#ifndef HUNGARIAN_ALGORITHM_H
#define HUNGARIAN_ALGORITHM_H

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <float.h>

class HungarianAlgorithm {
public:
    HungarianAlgorithm();  // コンストラクタ
    ~HungarianAlgorithm(); // デストラクタ

    // ダミー行または列を追加して正方行列を作成
    void addDummyElements(float *matrix, int rows, int cols, int newSize);

    // カバーを更新する補正操作メソッド
    void updateMatrixForUncovered(float *matrix, int *rowCovered, int *colCovered, int rows, int cols);

    // 割り当てを最適化するためのメソッド
    void findOptimalAssignment(float *matrix, int rows, int cols, std::vector<int>& assignments);

    // ホストからカーネルを実行するラッパーメソッド
    void runHungarianAlgorithm(float *d_matrix, float *d_minRow, float *d_minCol, int size);

private:
    // メンバ変数
    float *d_minVal;  // 最小値用のデバイスメモリポインタ
};

#endif // HUNGARIAN_ALGORITHM_H
