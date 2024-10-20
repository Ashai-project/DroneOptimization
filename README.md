## 背景
ライブの演出やオリンピックの開会式などで見かけた多数ドローンのアニメーションは三次元空間に頂点をプロットすることで作られている  
東京オリンピックの開会式では1800台以上のドローンが使われたとか  

## 問題設定
シーンが移行するとき、各ドローンはできるだけ飛行距離が少ないほうが効率が良い  
そこで2モデルの頂点に対して、総移動距離を最小化するような対応付けを行い、描画を行う
これは最小重み2部マッチング問題として扱える  

## 最適化
アルゴリズムが単純であるためGPU実装しやすいという理由からHungarian Algorithmを採用した  
全体の計算量 : $O(N^3)$  
行列操作を並列化することで行列操作の計算量を$O(N^2)$から$O(N)$に減らしている  
Hungarian Algorithmは割当問題ではメジャーな解法で最適解が求まるが、今回の問題のように必ず最適解である必要がない場合はメタヒューリスティックな解法と比較するのもいいかもしれない

## 結果
[greedy.webm](https://github.com/user-attachments/assets/392c0c43-cca2-4a41-a83b-e52e1379dc58)
[hungarian.webm](https://github.com/user-attachments/assets/a98d6c2e-150b-41d1-977b-166221164770)

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
