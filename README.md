## 背景
ライブの演出やオリンピックの開会式などで見かけた多数ドローンのアニメーションは三次元空間に頂点をプロットすることで作られている  
東京オリンピックの開会式では1800台以上のドローンが使われたとか  

## 問題設定
シーンが移行するとき、各ドローンはできるだけ飛行距離が少ないほうが効率が良い  
そこで2モデルの頂点に対して、総移動距離を最小化するような対応付けを行い、描画を行う  
この最適化は最小重み2部マッチング問題として扱える  
実装では球面に均等に配置した頂点からランダムに配置した頂点への移動を考える  
モデルをロードして頂点の生成を行う

## 最適化
### Hungarian Algorithm
アルゴリズムはHungarian Algorithm(Kuhn-MunkresAlgorithm)を採用した  
全体の計算量 : $O(N^3)$  
一部並列化可能なループが存在するが、GPU並列化とスレッド並列化、どちらの場合も並列化可能部分の処理負荷が大きくないため、並列化のオーバヘッドが並列化による計算時間短縮を大きく上回ってしまった  
Hungarian Algorithmはシングルスレッドでの実装となっている  
むしろ、要素数が多い場合には最適解を諦め、部分問題に分割することで計算時間を短縮することが望ましいと考える(これはそのうち実装する)  
Hungarian Algorithmは最適解が求まる  

### Genetic Algorithm
遺伝的アルゴリズムの実装も行った  
交叉はPMXを模したもの  
並列化が容易でありGPU実装も行った 
パラメータの最適化はまだ未検証

## 結果
頂点数:2000
貪欲法(総移動距離: 4929.29)
![greedy](https://github.com/user-attachments/assets/85c41be8-d17b-4659-b130-b0f6d8ed19db)
Hungarian Algorithm(総移動距離: 2472.41)
![hungarian (1)](https://github.com/user-attachments/assets/be7f5667-deb4-4444-be73-ee1960d546d1)

## 高速化
GAのGPU実装の高速化を試みた
Nsightでプロファイル結果は下の通り  
改善前:
![Screenshot from 2024-10-27 18-08-39](https://github.com/user-attachments/assets/ba282a32-8626-43de-9dc5-4f6cb75d5971)
改善後:
![Screenshot from 2024-10-27 18-12-54](https://github.com/user-attachments/assets/366f0dab-479a-4c00-afc2-97dac86f1adb)
カーネル実行時間の99.4%を占めていたPMXの交叉が87.1%まで低下した  
しかし依然交叉関数がボトルネックであることに変わりないため、今後より高速な交叉アルゴリズムの導入を検討する

## 実行
環境はLinuxを想定  
実行にはOpenGLとGLUTとCUDAが必要  
buildディレクトリ作成
```
mkdir build
cd build 
```
ビルド
```
cmake ..
make
```
実行
```
cd ../bin
```
2000頂点の貪欲法によるマッチングと描画
```
./DroneOptSample
```
2000頂点のハンガリアンによるマッチング最適化と描画
```
./DroneOptHungarian
```
実装速度計測  
任意の頂点数で最適化にかかる時間を計測  
コード内のtestCountを変更すればテストの実行回数を変更し、平均実行時間を計測できる
```
./{$Type}_speed_test
```
{$Type}説明  
GA:遺伝的アルゴリズム(GA)のcpu実装、交叉はPMX  
GA_PARA:GAのGPU実装、交叉はPMX  
hungarian_cpu:ハンガリアンのシングルスレッド実装  
hungarian_PARA:マルチスレッド実装であるが、並列化のオーバーヘッドが大きすぎるため、ほとんどの並列化を無効にしてある
