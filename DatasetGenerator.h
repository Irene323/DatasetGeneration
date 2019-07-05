//
// Created by yirenli on 02.07.19.
//

#ifndef DATASETGENERATION_DATASETGENERATOR_H
#define DATASETGENERATION_DATASETGENERATOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

using namespace std;
using namespace cv;
using namespace Eigen;

#define _CRT_SECURE_NO_WARNINGS
#define _pi 3.1415926
#define FIRSTIMAGENAME 2042

inline vector<string> split(const string &str, const string &c);

inline int countLines(const string &filePath);

Point2i getDepthMapSize(const string &filePath);

template<typename T>
Eigen::Matrix<T, 4, 4> composeAugmentedProjectionMatrix(const Eigen::Matrix<T, 3, 3> &internalMatrix,
                                                        const Eigen::Matrix<T, 4, 4> &externalMatrix) {
    Eigen::Matrix<T, 4, 4> augmentedInternal;
    augmentedInternal(0, 0) = internalMatrix(0, 0) * (1.0f / internalMatrix(2, 2));
    augmentedInternal(0, 1) = internalMatrix(0, 1) * (1.0f / internalMatrix(2, 2));
    augmentedInternal(0, 2) = internalMatrix(0, 2) * (1.0f / internalMatrix(2, 2));
    augmentedInternal(0, 3) = 0.0f;

    augmentedInternal(1, 0) = internalMatrix(1, 0) * (1.0f / internalMatrix(2, 2));
    augmentedInternal(1, 1) = internalMatrix(1, 1) * (1.0f / internalMatrix(2, 2));
    augmentedInternal(1, 2) = internalMatrix(1, 2) * (1.0f / internalMatrix(2, 2));
    augmentedInternal(1, 3) = 0.0f;

    augmentedInternal(2, 0) = 0.0f;
    augmentedInternal(2, 1) = 0.0f;
    augmentedInternal(2, 2) = 0.0f;
    augmentedInternal(2, 3) = 1.0f;

    augmentedInternal(3, 0) = 0.0f;
    augmentedInternal(3, 1) = 0.0f;
    augmentedInternal(3, 2) = 1.0f;
    augmentedInternal(3, 3) = 0.0f;

    return augmentedInternal * externalMatrix;
}

void writePair(int picID1, int featureID1, int picID2, int featureID2, const string &flag, const string &matchtablepath,
               const string &nonmatchtablepath);

Mat readDataIntoMat(int name, const string &flag, const string &feature_address, const string &depth_address);

vector<int> readDataIntVec(const string &namelistfile);

Mat locationtrans(float x1, float y1, float z1, Eigen::Matrix3f &internal0, Eigen::Matrix3f &internal1,
                  Eigen::Matrix4f &external0, Eigen::Matrix4f &external1);

Mat
decision(Mat p1, Mat p2, float threshloc, float threshdepth, float realdepthscale, float scalethresh, float orithresh);

Eigen::Matrix4f getExternal(int imagename, const string &externalfile);

Mat getRotScale(Mat p1, Mat p2);

int checkRotScale(/*Mat point1,Mat point2,*/Mat as, float scalethresh, float orithresh);

Eigen::Matrix3f readInternal(int imagename, const string &internalfile, int *width, int *height);

Mat rotpoint(float tmpx, float tmpy, float cx, float cy, float ori0);

void crop(int picId0, Mat point0, string patchpath, const string& scaledpath);

#endif //DATASETGENERATION_DATASETGENERATOR_H
