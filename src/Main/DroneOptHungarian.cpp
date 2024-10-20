#include <GL/glut.h>
#include "../Cuda/DroneUtil.h"

// ウィンドウサイズ指定用変数
int windowWidth = 1600;
int windowHeight = 1200;
float pointSize = 5.0f;
int vertexCountA = 1800, vertexCountB = 1800;
int currentFrame = 0;
int totalFrames = 100; // アニメーションフレーム数
int modelAFrames = 30; // モデルA表示フレーム数
int modelBFrames = 30; // モデルB表示フレーム数 

// カメラパラメータ
double fovy = 45.0, near = 0.1, far = 100;
float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 20.0f;  // カメラ位置
float cameraLookAtX = 0.0f, cameraLookAtY = 0.0f, cameraLookAtZ = 0.0f;  // 見る方向
float cameraUpX = 0.0f, cameraUpY = 1.0f, cameraUpZ = 0.0f;  // カメラの上方向
DroneUtil droneUtil(vertexCountA, vertexCountB, pointSize); 

void setProjection() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fovy, (double)windowWidth / (double)windowHeight, near, far);
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
    const auto& vertexMapping = droneUtil.getVertexMapping();
    for (int i = 0; i < vertexMapping.size(); ++i) {
        Vec3 currentPos = droneUtil.getModelA(i).interpolate(droneUtil.getModelB(vertexMapping[i]), t);
        glVertex3f(currentPos.x, currentPos.y, currentPos.z);
    }
    glEnd();
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
    glutDisplayFunc(display);
    glutTimerFunc(16, update, 0);
    glutMainLoop();

    return 0;
}
