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
貪欲法
![greedy](https://github.com/user-attachments/assets/85c41be8-d17b-4659-b130-b0f6d8ed19db)
Hungarian Algorithm(まだ0マッチングの実装がいけない)
![hungarian](https://github.com/user-attachments/assets/1a103d6a-48fe-44ab-98ef-c664d8bcd0f8)

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
./DroneOptimization
```
