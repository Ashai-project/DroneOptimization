import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# N要素のランダムなコスト行列を生成する
N = 2000  # 要素の数（必要に応じて変更）
cost_matrix = np.random.rand(N, N)

# ハンガリアンアルゴリズムで解く
start_time = time.time()  # 計算時間の計測開始

row_ind, col_ind = linear_sum_assignment(cost_matrix)

end_time = time.time()  # 計算時間の計測終了

# 最適な割り当てとそのコストを表示
optimal_cost = cost_matrix[row_ind, col_ind].sum()
print(f"Optimal cost: {optimal_cost}")
print(f"Time taken: {end_time - start_time} seconds")
