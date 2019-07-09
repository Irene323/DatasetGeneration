#include "DatasetGenerator.h"

int main(int argc, char **argv) {
//    readDataIntoMat(2042, "feature", "/home/yirenli/dev/DatasetGenerationHelper/out/keypoints/", "");
//    readDataIntoMat(2042, "depth", "", "/home/yirenli/dev/DatasetGenerationHelper/out/depth_maps/");

//    cout << getExternal(2042, "/home/yirenli/dev/DatasetGenerationHelper/out/cameras_images_txt/images.txt");
//    int width = 0;
//    int height = 0;
//    cout << readInternal(2057, "/home/yirenli/dev/DatasetGenerationHelper/out/cameras_images_txt/cameras.txt", &width, &height);

    float sigma = 1.6;
    float s = 3.0;
    for (int a = 0; a < 4; a++) {
        for (int i = 0; i <= s + 2; i++) {//s+3
            cout << pow(2, (double) a) * pow(2, (double) i / s) * sigma << endl;
        }
        cout << endl;
    }
}