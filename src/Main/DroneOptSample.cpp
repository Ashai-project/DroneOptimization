#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <GL/glut.h>

// Vec3構造体で頂点座標を表現
struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    float distance(const Vec3& other) const {
        return std::sqrt((x - other.x) * (x - other.x) +
                         (y - other.y) * (y - other.y) +
                         (z - other.z) * (z - other.z));
    }

    Vec3 interpolate(const Vec3& target, float t) const {
        return Vec3(x + t * (target.x - x), y + t * (target.y - y), z + t * (target.z - z));
    }
};

// 頂点リスト
std::vector<Vec3> modelA, modelB;
std::vector<int> vertexMapping; // 頂点対応付け結果
int droneCount = 0;
int currentFrame = 0;
int totalFrames = 100; // アニメーションフレーム数
int modelAFrames = 30; // モデルA表示フレーム数
int modelBFrames = 30; // モデルB表示フレーム数

// ウィンドウサイズ指定用変数
int windowWidth = 1600;
int windowHeight = 1200;
float pointSize = 5.0f;  // 点の大きさ

// カメラパラメータ
float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 20.0f;  // カメラ位置
float cameraLookAtX = 0.0f, cameraLookAtY = 0.0f, cameraLookAtZ = 0.0f;  // 見る方向
float cameraUpX = 0.0f, cameraUpY = 1.0f, cameraUpZ = 0.0f;  // カメラの上方向

// 関数プロトタイプ宣言
void optimizeVertexMapping();
float evaluateMapping();

// モデルの初期化
void initializeModels() {
    // モデルAとモデルBの頂点をここでロードまたは生成する
    int vertexCountA = 1800; // モデルAの頂点数
    int vertexCountB = 1800; // モデルBの頂点数
    // srand(static_cast<unsigned int>(time(0))); // ランダムシードの初期化
    srand(100); // ランダムシードの初期化
    // モデルAの頂点を生成
    float radius = 5.0f; // 球面の半径
    for (int i = 0; i < vertexCountA; ++i) {
        float phi = acos(1 - 2 * (i + 0.5f) / vertexCountA);  // 仰角（phi）
        float theta = M_PI * (1 + sqrt(5)) * i;               // 方位角（theta）

        // 球面座標から直交座標への変換
        float x = radius * sin(phi) * cos(theta);
        float y = radius * sin(phi) * sin(theta);
        float z = radius * cos(phi);

        modelA.push_back(Vec3(x, y, z));
    }

    // モデルBの頂点を生成
    for (int i = 0; i < vertexCountB; ++i) {
        float x = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        float y = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        float z = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        modelB.push_back(Vec3(x, y, z));
    }

    // ドローン数をモデルの最大頂点数に設定
    droneCount = std::max(modelA.size(), modelB.size());

    // 頂点対応付けの最適化
    optimizeVertexMapping();

    // 対応付けが完了したら総移動距離を評価
    float totalDistance = evaluateMapping();
    std::cout << "総移動距離: " << totalDistance << std::endl;
}

void setProjection() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)windowWidth / (double)windowHeight, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
}


// OpenGLで描画するための設定
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    setProjection();
    glLoadIdentity();

    // カメラの設定 (視点の位置、見ている方向、カメラの上方向)
    gluLookAt(cameraPosX, cameraPosY, cameraPosZ,
              cameraLookAtX, cameraLookAtY, cameraLookAtZ,
              cameraUpX, cameraUpY, cameraUpZ);

    // アニメーション中の頂点の位置を計算
    float t = static_cast<float>(currentFrame - modelAFrames) / (totalFrames - modelAFrames - modelBFrames);
    if (t < 0) t = 0;
    if (t > 1) t = 1;

    // 頂点の大きさを指定
    glPointSize(pointSize);

    glBegin(GL_POINTS);
    for (int i = 0; i < droneCount; ++i) {
        Vec3 currentPos = modelA[i].interpolate(modelB[vertexMapping[i]], t);
        glVertex3f(currentPos.x, currentPos.y, currentPos.z);
    }
    glEnd();

    glutSwapBuffers();
}

float evaluateMapping() {
    float totalDistance = 0.0f;
    for (int i = 0; i < modelA.size(); ++i) {
        totalDistance += modelA[i].distance(modelB[vertexMapping[i]]);
    }
    return totalDistance;
}

// 頂点の対応付けを最適化
void optimizeVertexMapping() {
    vertexMapping.resize(droneCount);
    std::vector<bool> usedB(modelB.size(), false);

    for (int i = 0; i < modelA.size(); ++i) {
        int bestIndex = -1;
        float bestDistance = std::numeric_limits<float>::max();
        for (int j = 0; j < modelB.size(); ++j) {
            if (!usedB[j]) {
                float distance = modelA[i].distance(modelB[j]);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestIndex = j;
                }
            }
        }
        if (bestIndex != -1) {
            vertexMapping[i] = bestIndex;
            usedB[bestIndex] = true;
        }
    }

    // 余ったドローン（頂点）には対応するBの頂点を与える
    for (int i = modelA.size(); i < droneCount; ++i) {
        vertexMapping[i] = i % modelB.size();
    }
}

void update(int value) {
    currentFrame++;
    if (currentFrame > totalFrames) currentFrame = 0;
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    // ウィンドウサイズを指定
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Drone Animation");

    initializeModels();

    // 深度テストの有効化
    glEnable(GL_DEPTH_TEST);

    glutDisplayFunc(display);
    glutTimerFunc(16, update, 0);
    glutMainLoop();

    return 0;
}
