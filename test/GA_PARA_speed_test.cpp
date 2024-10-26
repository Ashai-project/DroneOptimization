#include <iostream>
#include <vector>
#include <chrono> 
#include "../src/Cuda/DroneUtil.h"

// テストの回数と固定シードのリスト
const int testCount = 1;  // テスト回数
int N ; //要素数


// 最適解の計算時間を求める関数
void calculateAverageTime() {
    double totalDuration = 0.0;  // 全体の計算時間を保持する変数
    std::vector<int> seedList(testCount); 
    for (int i = 0; i < testCount; ++i) {
        seedList[i] = i * 100 + 100;
    }

    // 要素数の入力
    std::cout << "要素数を整数値で入力してください: ";
    std::cin >> N;

    for (int i = 0; i < testCount; ++i) {
        int currentSeed = seedList[i];  // テストごとのシードを取得

        // DroneUtil クラスのインスタンスを作成 
        DroneUtil droneUtil(N, N, 5.0f, DroneUtil::AlgorithmType::GA_PARA);

        // モデルの初期化と頂点対応付けの最適化
        double elapsed = droneUtil.initializeModels(currentSeed);

        droneUtil.checkForDuplicateMappings();

        // 経過時間の計算（ミリ秒単位）
        totalDuration += elapsed;

        std::cout << "Test " << i + 1 << " with seed " << currentSeed << ": " 
                  << elapsed << " ms" << std::endl;
    }

    // 平均時間の計算
    double averageDuration = totalDuration / testCount;
    std::cout << "Average computation time: " << averageDuration << " ms" << std::endl;
}

int main() {
    calculateAverageTime();  // 平均計算時間を求める関数を実行
    return 0;
}
