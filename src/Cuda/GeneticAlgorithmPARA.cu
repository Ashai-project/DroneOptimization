#include "GeneticAlgorithmPARA.h"

// CUDAカーネル: 適応度を計算
__global__ void calculate_fitness_kernel(float* fitness, int* population, float* cost_matrix, int n, int population_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        float total_cost = 0.0f;
        for (int i = 0; i < n; ++i) {
            int col = population[idx * n + i];
            total_cost += cost_matrix[i * n + col];
        }
        fitness[idx] = total_cost;
    }
}

// CUDAカーネル: トーナメント選択
__global__ void tournament_selection_kernel(int* population, float* fitness, int* selected_parents, int population_size, int n, int tournament_size, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < population_size) {
        // 1つ目の親個体のトーナメント選択
        int best_index1 = curand(&rand_states[idx]) % population_size;
        for (int i = 1; i < tournament_size; ++i) {
            int competitor_index = curand(&rand_states[idx]) % population_size;
            if (fitness[competitor_index] < fitness[best_index1]) {
                best_index1 = competitor_index;
            }
        }

        // 2つ目の親個体のトーナメント選択
        int best_index2 = curand(&rand_states[idx]) % population_size;
        for (int i = 1; i < tournament_size; ++i) {
            int competitor_index = curand(&rand_states[idx]) % population_size;
            if (fitness[competitor_index] < fitness[best_index2]) {
                best_index2 = competitor_index;
            }
        }

        // 1つ目の親個体を格納
        selected_parents[idx * 2] = best_index1;
        
        // 2つ目の親個体を格納
        selected_parents[idx * 2 + 1] = best_index2;
    }
}

// CUDAカーネル: 部分写像交叉（PMX）
__global__ void pmx_crossover_kernel(int* population, int* selected_parents, int* new_population,
                                     int population_size, int n, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < population_size) {
        // 親のインデックスを取得
        int parent1_idx = selected_parents[idx * 2];
        int parent2_idx = selected_parents[idx * 2 + 1];

        // 交叉点の選択
        int start = curand(&rand_states[idx]) % n;
        int end = curand(&rand_states[idx]) % n;
        if (start > end) {
            int temp = start;
            start = end;
            end = temp;
        }

        int* child = &new_population[idx * n];
        int* parent1 = &population[parent1_idx * n];
        int* parent2 = &population[parent2_idx * n];

        // 子個体の初期化
        for (int i = 0; i < n; ++i) {
            child[i] = -1;
        }

        // 親1から部分コピー
        for (int i = start; i <= end; ++i) {
            child[i] = parent1[i];
        }

        // マッピングの確立と残りの遺伝子の埋め込み
        for (int i = 0; i < n; ++i) {
            if (i >= start && i <= end) {
                continue; // 交叉区間はスキップ
            }

            int val = parent2[i];

            // マッピングを使用して遺伝子を決定
            while (true) {
                bool found = false;
                for (int j = start; j <= end; ++j) {
                    if (val == child[j]) {
                        val = parent2[j];
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    break;
                }
            }

            child[i] = val;
        }
    }
}

// CUDAカーネル: 突然変異
__global__ void mutate_kernel(int* population, int population_size, int n, curandState* rand_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {   
            int i = curand(&rand_states[idx]) % n;
            int j = curand(&rand_states[idx]) % n;
            // ランダムに2つの要素を入れ替える
            int temp = population[idx * n + i];
            population[idx * n + i] = population[idx * n + j];
            population[idx * n + j] = temp;
    }
}

// カーネルで使うランダム状態を初期化
__global__ void init_rand_kernel(curandState* rand_states, unsigned long seed, int population_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < population_size) {
        curand_init(seed, idx, 0, &rand_states[idx]);
    }
}


// コンストラクタの実装
GeneticAlgorithmPARA::GeneticAlgorithmPARA(int population_size, int generations, double mutation_rate, 
                                   int n, const std::vector<float>& cost_matrix)
    : population_size(population_size), generations(generations), mutation_rate(mutation_rate), 
      n(n), cost_matrix(cost_matrix), best_fitness(std::numeric_limits<float>::max()) {}

// 適応度を計算する
float GeneticAlgorithmPARA::fitness(const Individual &individual) const {
    float total_cost = 0.0f;
    for (int i = 0; i < n; ++i) {
        total_cost += cost_matrix[i * n + individual[i]];
    }
    return total_cost;
}

// ドラフト制度による初期解生成
std::vector<GeneticAlgorithmPARA::Individual> GeneticAlgorithmPARA::draft_initial_solutions(const std::vector<float>& cost_matrix, int n, int max_rounds, int num_solutions) {
    std::vector<Individual> solutions;

    for (int s = 0; s < num_solutions; ++s) {
        Individual solution(n, -1);  // 最終的なマッチングを格納
        std::vector<bool> assigned_row(n, false);  // 行の割り当て状況を保持
        std::vector<bool> assigned_column(n, false);  // 列の割り当て状況を保持
        std::vector<std::vector<std::pair<int, float>>> row_candidates(n);  // 各行の候補ペア（列とコスト）

        // 各行の最小コスト候補を最大max_roundsまで収集
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<int, float>> candidates;
            for (int j = 0; j < n; ++j) {
                candidates.push_back({j, cost_matrix[i * n + j]});
            }
            // コストの昇順でソート
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
            // 最大max_roundsの候補を格納
            row_candidates[i].assign(candidates.begin(), candidates.begin() + std::min(max_rounds, (int)candidates.size()));
        }

        // ドラフトステップで優先的にマッチング
        for (int round = 1; round <= max_rounds; ++round) {
            for (int j = 0; j < n; ++j) {  // 列ごとの処理
                if (assigned_column[j]) continue;  // すでにマッチングされた列はスキップ

                int best_row = -1;
                float best_cost = std::numeric_limits<float>::max();

                // 各行を見て、その列がまだマッチングされていない場合、コストを比較
                for (int i = 0; i < n; ++i) {
                    if (!assigned_row[i] && row_candidates[i].size() >= round) {
                        int candidate_column = row_candidates[i][round - 1].first;
                        if (candidate_column == j && row_candidates[i][round - 1].second < best_cost) {
                            best_cost = row_candidates[i][round - 1].second;
                            best_row = i;
                        }
                    }
                }

                // 最良の行をこの列にマッチング
                if (best_row != -1) {
                    solution[best_row] = j;
                    assigned_row[best_row] = true;
                    assigned_column[j] = true;
                }
            }
        }

        // 残りの未割り当ての行をランダムオフセットを付けて貪欲法で埋める
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<> offset_dist(0, n-1);
        int offset = offset_dist(rng);  // ランダムオフセット

        for (int i = 0; i < n; ++i) {
            if (solution[i] == -1) {
                // ランダムオフセットを考慮して、割り当てられていない列を見つけて割り当て
                for (int j = 0; j < n; ++j) {
                    int candidate_column = (j + offset) % n;
                    if (!assigned_column[candidate_column]) {
                        solution[i] = candidate_column;
                        assigned_column[candidate_column] = true;
                        break;
                    }
                }
            }
        }

        // 解を保存
        solutions.push_back(solution);
    }

    return solutions;
}

// 貪欲法による初期解生成
GeneticAlgorithmPARA::Individual GeneticAlgorithmPARA::greedy_initial_solution(const std::vector<float>& cost_matrix, int n, int offset) {
    Individual solution(n, -1); 
    std::vector<bool> assigned(n, false); 

    for (int i = 0; i < n; ++i) {
        int best_j = -1;
        float best_cost = std::numeric_limits<float>::max();
        int row = (i + offset) % n;  

        for (int j = 0; j < n; ++j) {
            if (!assigned[j] && cost_matrix[row * n + j] < best_cost) {
                best_cost = cost_matrix[row * n + j];
                best_j = j;
            }
        }

        solution[row] = best_j;
        assigned[best_j] = true;
    }

    return solution;
}

// ランダムな個体を生成
GeneticAlgorithmPARA::Individual GeneticAlgorithmPARA::random_individual() const {
    Individual individual(n);
    std::iota(individual.begin(), individual.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(individual.begin(), individual.end(), g);
    return individual;
}

// ランダムな貪欲法
GeneticAlgorithmPARA::Individual GeneticAlgorithmPARA::random_greedy_individual(const std::vector<float>& cost_matrix) const {
    Individual individual(n);
    Individual solution(n); 
    std::iota(individual.begin(), individual.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(individual.begin(), individual.end(), g);
    std::vector<bool> assigned(n, false); 

    for (int i = 0; i < n; ++i) {
        int best_j = -1;
        float best_cost = std::numeric_limits<float>::max();
        int row = individual[i];  

        for (int j = 0; j < n; ++j) {
            if (!assigned[j] && cost_matrix[row * n + j] < best_cost) {
                best_cost = cost_matrix[row * n + j];
                best_j = j;
            }
        }

        solution[row] = best_j;
        assigned[best_j] = true;
    }

    return solution;
}

// 最適化実行
GeneticAlgorithmPARA::Individual GeneticAlgorithmPARA::optimize() {
    // 個体群の初期化
    std::vector<int> population(population_size * n);
    
    // ドラフト制度による初期解生成
    // int draftSize = std::min(population_size / 2, (n / 2) );
    // std::vector<Individual> draft_solutions = draft_initial_solutions(cost_matrix, n, 15, draftSize);
    // for (int i = 0; i < draftSize; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         population[i * n + j] = draft_solutions[i][j];
    //     }
    //     Individual solution(population.begin() + i * n, population.begin() + (i + 1) * n);
    //     std::cout << "Draft solution " << i << " fitness: " << fitness(solution) << std::endl;
    // }
    int draftSize = 0;

    // 貪欲法による初期解生成
    // int greedySize = std::min(population_size - draftSize, n);
    // for (int i = draftSize; i < greedySize + draftSize; ++i) {
    //     Individual greedy_solutions = greedy_initial_solution(cost_matrix, n, i);
    //     for (int j = 0; j < n; ++j) {
    //         population[i * n + j] = greedy_solutions[j];
    //     }
    //     Individual solution(population.begin() + i * n, population.begin() + (i + 1) * n);
    //     std::cout << "Greedy solution with offset " << i << " fitness: " << fitness(solution) << std::endl;
    // }
    int greedySize = 0;


    // 残りはランダム貪欲法による初期解生成
    for (int i = greedySize + draftSize; i < population_size; ++i) {
        // Individual random_solutions = random_individual();
        Individual random_solutions = random_greedy_individual(cost_matrix);
        for (int j = 0; j < n; ++j) {
            population[i * n + j] = random_solutions[j];
        }
    }

    std::vector<float> fitness(population_size);

       // CUDAデバイスメモリの準備
    int* d_population;
    int* d_new_population;
    int* d_selected_parents;
    float* d_fitness;
    float* d_cost_matrix;
    curandState* d_rand_states;
    int* d_child;
    bool* d_used;

    cudaMalloc(&d_population, population_size * n * sizeof(int));
    cudaMalloc(&d_new_population, population_size * n * sizeof(int));
    cudaMalloc(&d_selected_parents, population_size * 2 * sizeof(int));
    cudaMalloc(&d_fitness, population_size * sizeof(float));
    cudaMalloc(&d_cost_matrix, n * n * sizeof(float));
    cudaMalloc(&d_rand_states, population_size * sizeof(curandState));
    cudaMalloc(&d_child, population_size * n * sizeof(int));
    cudaMalloc(&d_used, population_size * n * sizeof(bool));

    // ホストからデバイスへデータ転送
    cudaMemcpy(d_cost_matrix, cost_matrix.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_population, population.data(), population_size * n * sizeof(int), cudaMemcpyHostToDevice);

    // ランダム初期化カーネル
    int threads_per_block = 256;
    int blocks = (population_size + threads_per_block - 1) / threads_per_block;
    init_rand_kernel<<<blocks, threads_per_block>>>(d_rand_states, time(0), population_size);

    for (int generation = 0; generation < generations; ++generation) {
        // 適応度計算の並列化
        calculate_fitness_kernel<<<blocks, threads_per_block>>>(d_fitness, d_population, d_cost_matrix, n, population_size);

        // トーナメント選択の並列化
        tournament_selection_kernel<<<blocks, threads_per_block>>>(d_population, d_fitness, d_selected_parents, population_size, n, 7, d_rand_states);

        // 交叉の並列化
        pmx_crossover_kernel<<<blocks , threads_per_block>>>(d_population, d_selected_parents, d_new_population, population_size, n, d_rand_states);

        // 突然変異の並列化
        mutate_kernel<<<blocks, threads_per_block>>>(d_new_population, population_size, n, d_rand_states);

        // 新しい個体群に置き換え
        cudaMemcpy(d_population, d_new_population, population_size * n * sizeof(int), cudaMemcpyDeviceToDevice);

        // 最良の個体を記録
        if (generation % 100 == 0) {
            cudaMemcpy(population.data(), d_population, population_size * n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(fitness.data(), d_fitness, population_size * sizeof(float), cudaMemcpyDeviceToHost);
        
            for (int i = 0; i < population_size; ++i) {
                Individual individual(population.begin() + i * n, population.begin() + (i + 1) * n);
                float current_fitness = this->fitness(individual);
                if (current_fitness < best_fitness) {
                    best_fitness = current_fitness;
                    best_individual = individual;
                }
            }
            std::cout << "Generation " << generation << " - Best Fitness: " << best_fitness << std::endl;
        }
    }
    // 最良の個体を記録
    cudaMemcpy(population.data(), d_population, population_size * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(fitness.data(), d_fitness, population_size * sizeof(float), cudaMemcpyDeviceToHost);
       
    for (int i = 0; i < population_size; ++i) {
        Individual individual(population.begin() + i * n, population.begin() + (i + 1) * n);
        float current_fitness = this->fitness(individual);
        if (current_fitness < best_fitness) {
            best_fitness = current_fitness;
            best_individual = individual;
        }
    }

    // デバイスメモリの解放
    cudaFree(d_population);
    cudaFree(d_new_population);
    cudaFree(d_fitness);
    cudaFree(d_cost_matrix);
    cudaFree(d_selected_parents);
    cudaFree(d_rand_states);
    cudaFree(d_child);
    cudaFree(d_used);

    // 最良の個体を返す
    return best_individual;
}

// 最小コストの取得
float GeneticAlgorithmPARA::getBestCost() const {
    return best_fitness;
}
