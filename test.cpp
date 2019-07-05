#include "DatasetGenerator.h"

int main(int argc, char **argv) {
//    readDataIntoMat(2042, "feature", "/home/yirenli/dev/DatasetGenerationHelper/out/keypoints/", "");
//    readDataIntoMat(2042, "depth", "", "/home/yirenli/dev/DatasetGenerationHelper/out/depth_maps/");

//    cout << getExternal(2042, "/home/yirenli/dev/DatasetGenerationHelper/out/cameras_images_txt/images.txt");
    vector<int> width;
    vector<int> height;
    cout << readInternal(2057, "/home/yirenli/dev/DatasetGenerationHelper/out/cameras_images_txt/cameras.txt", width,
                         height);
}