#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include <iostream>
#include <numeric>
#include <limits>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_set>

class GeneticAlgorithm {
public:
    // コンストラクタで設定を受け取る
    GeneticAlgorithm(int population_size, int generations, double mutation_rate, 
                     int n, const std::vector<std::vector<float>>& cost_matrix);

    // 最適化実行関数
    std::vector<int> optimize();
    
    // 最小コストの取得
    float getBestCost() const;

private:
    using Individual = std::vector<int>;

    int population_size;     // 個体群のサイズ
    int generations;         // 世代数
    double mutation_rate;     // 突然変異率
    int n;                   // 問題のサイズ（行列のサイズ）
    std::vector<std::vector<float>> cost_matrix;  // コスト行列
    Individual best_individual;  // 最良の割り当て
    float best_fitness;        // 最小コスト

    // 適応度を計算する関数
    float fitness(const Individual &individual) const;

    std::vector<int> greedy_initial_solution(const std::vector<std::vector<float>>& cost_matrix, int n, int offset);

    // 個体をランダムに生成
    Individual random_individual() const;

    // AREXによる交叉
    std::vector<GeneticAlgorithm::Individual> pmx_crossover(const std::vector<Individual>& parents);

    // 突然変異関数
    void mutate(Individual &individual) const;

    // トーナメント選択
    Individual tournament_selection(const std::vector<Individual> &population) const;

    // JGGによる世代交代
    void jgg_generation(std::vector<Individual>& population, const std::vector<Individual>& offspring);
};

#endif // GENETIC_ALGORITHM_H
