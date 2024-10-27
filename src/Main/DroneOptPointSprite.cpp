#include <GL/glut.h>
#include "../Cuda/DroneUtil.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // 画像読み込み用

// ウィンドウサイズ指定用変数
int windowWidth = 1600;
int windowHeight = 1200;
float pointSize = 28.0f;  // 点スプライトのサイズ
float radius = 6.0f;
int vertexCountA = 2000, vertexCountB = 2000;
int currentFrame = 0;
int totalFrames = 100; // アニメーションフレーム数
int modelAFrames = 30; // モデルA表示フレーム数
int modelBFrames = 30; // モデルB表示フレーム数 

// カメラパラメータ
double fovy = 45.0, near = 0.1, far = 100;
float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 18.0f;  // カメラ位置
float cameraLookAtX = 0.0f, cameraLookAtY = 0.0f, cameraLookAtZ = 0.0f;  // 見る方向
float cameraUpX = 0.0f, cameraUpY = 1.0f, cameraUpZ = 0.0f;  // カメラの上方向
DroneUtil droneUtil(vertexCountA, vertexCountB, radius, DroneUtil::AlgorithmType::HUNGARIAN_CPU);

GLuint textureID;

void loadTexture() {
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // 画像を読み込んでテクスチャとして使用
    int width, height, nrChannels;
    unsigned char* image = stbi_load("../resource/images/test2.png", &width, &height, &nrChannels, 0);
    if (!image) {
        printf("Failed to load texture\n");
        return;
    }

    // テクスチャの設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, (nrChannels == 4 ? GL_RGBA : GL_RGB), GL_UNSIGNED_BYTE, image);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    stbi_image_free(image);
}

void setProjection() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fovy, (double)windowWidth / (double)windowHeight, near, far);
    glMatrixMode(GL_MODELVIEW);
}

// OpenGLで描画するための設定
void display() {
    glClearColor(0.3f, 0.3f, 1.0f, 1.0f);
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

    // 点スプライトの設定
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);  // 深度バッファへの書き込みを無効化
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // 透過部分での重ね合わせを設定
    glEnable(GL_TEXTURE_2D);
    glPointSize(pointSize);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

    glBegin(GL_POINTS);
    const auto& vertexMapping = droneUtil.getVertexMapping();
    for (int i = 0; i < vertexMapping.size(); ++i) {
        Vec3 currentPos = droneUtil.getModelA(i).interpolate(droneUtil.getModelB(vertexMapping[i]), t);
        glVertex3f(currentPos.x, currentPos.y, currentPos.z);
    }
    glEnd();

    glDisable(GL_POINT_SPRITE);
    glDisable(GL_TEXTURE_2D);
    glDepthMask(GL_TRUE);  // 深度バッファへの書き込みを有効化
    glDisable(GL_BLEND);
    glutSwapBuffers();
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

    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Drone Animation");

    droneUtil.initializeModels();  // モデルを初期化、組み合わせ最適化

    glEnable(GL_DEPTH_TEST);
    loadTexture();  // テクスチャを読み込む

    glutDisplayFunc(display);
    glutTimerFunc(16, update, 0);
    glutMainLoop();

    return 0;
}