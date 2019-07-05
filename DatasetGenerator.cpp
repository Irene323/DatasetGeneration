#include "DatasetGenerator.h"

//This function is a small helper to split string str with c
inline vector<string> split(const string &str, const string &c) {
    vector<string> r;
    if (str.empty()) {
        return r;
    }
    unsigned long i = 0;
    while (true) {
        auto j = str.find(c, i);
        if (j != string::npos) {
            r.push_back(str.substr(i, j - i));
            i = j + c.length();
        } else {
            r.push_back(str.substr(i));
            break;
        }
    }
    return r;
}

//This function counts the number of lines of a file
inline int countLines(const string &filePath) {
    fstream fin(filePath, ios::in);
    if (!fin) {
        cerr << "can not open file" << endl;
        return -1;
    }
    char c;
    int lineCnt = 0;
    while (fin.get(c)) {
        if (c == '\n')
            lineCnt++;
    }
    cout << "File: " << filePath << " has " << lineCnt << " lines." << endl;
    return lineCnt;
}

Point2i getDepthMapSize(const string &filePath) {
    Point2i size(-1, -1);
    fstream fin(filePath, ios::in);
    if (!fin) {
        cerr << "can not open file" << endl;
        return size;
    }
    string s;
    int lineCnt = 0;
    vector<string> a;
    while (getline(fin, s)) {
        if (lineCnt == 0) {
            a = split(s, "\t");
        }
        lineCnt++;
    }
    size.x = lineCnt;
    size.y = a.size() - 1;
    cout << "Depth map " << filePath << " is of size: " << size << endl;
    return size;
}

//This function writes the matching or nonmatching pair into a txt file
//every line is one pair
//in each line, format is: imageID1	featureID1	imageID2	featureID2
//separated with "\t"
void writePair(int picID1, int featureID1, int picID2, int featureID2, const string &flag, const string &matchtablepath,
               const string &nonmatchtablepath) {
    fstream file;
    char *fileadd;
    if (flag == "match") fileadd = const_cast<char *>(matchtablepath.c_str());
    else fileadd = const_cast<char *>(nonmatchtablepath.c_str());
    file.open(fileadd, ios::out | ios::app);
    if (file.fail()) {
        cout << "can not open file" << endl;
    }

    file << picID1 << "\t" << featureID1 << "\t" << picID2 << "\t" << featureID2 << "\n";
    file.close();
}


//This function reads every table of FEATURE EXTRACTION or DEPTH MAP of an image
//do not forget to set the matrix of depth map according to the picture size
Mat readDataIntoMat(int name, const string &flag, const string &feature_address, const string &depth_address) {
    string names = to_string(name);
    Mat matrix;
    string add;

    //read in feature detection
    if (flag == "feature") {
        cout << "feature detection of pic " << names << endl;
        add = feature_address + names + ".txt";
        matrix = Mat::zeros(countLines(add), 7, CV_32FC1);
    }
        //read in depth map
    else if (flag == "depth") {
        cout << "depth map of pic " << name << endl;
        add = depth_address + names + ".txt";
        Point depthMapSize = getDepthMapSize(add);
        matrix = Mat::zeros(depthMapSize.x, depthMapSize.y, CV_32FC1);
    }

    ifstream fin(add);

    string s0;
    int linenumber = 0;
    while (getline(fin, s0)) {
        char *s = (char *) s0.data();

        const char *d = "\t";
        char *p;
        p = strtok(s, d);
        int colnumber = 0;

        while (p) {
            float nump;
            nump = atof(p);
            matrix.at<float>(linenumber, colnumber) = nump;
            p = strtok(nullptr, d);
            colnumber++;
        }
        linenumber++;
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 6; j++)
            cout << matrix.at<float>(i, j) << "\t";
        cout << endl;
    }
    return matrix;

}

/*
This Function reads in the namelist of the pictures and store them into a vector
*/
vector<int> readDataIntVec(const string &namelistfile) {
    ifstream fin(namelistfile);
    string s0;
    int s1;
    vector<int> namelist;
    while (getline(fin, s0)) {
        s1 = atoi(s0.c_str());
        namelist.push_back(s1);
    }
    /*
    for (vector<int>::iterator it = namelist.begin(); it != namelist.end(); it++)
    cout << *it << endl;
    */
    return namelist;
}

/*
This function transforms one point in picture 1 to the corresponding location in picture 2
using depth map, internal & external metrix
z is from the depth map
return x1,y1,z1 as a 1*3 Mat
*/
Mat locationtrans(float x1, float y1, float z1, Eigen::Matrix3f &internal0, Eigen::Matrix3f &internal1,
                  Eigen::Matrix4f &external0, Eigen::Matrix4f &external1) {
    Eigen::Matrix4f P0 = composeAugmentedProjectionMatrix(internal0, external0);
    Eigen::Matrix4f P1 = composeAugmentedProjectionMatrix(internal1, external1);

    // compute reprojection matrix that transforms (x, y, 1/z, 1) from one image to the other
    Eigen::Matrix4f reprojectImg0_to_Img1 = P1 * P0.inverse();
    Eigen::Vector4f point1(x1, y1, 1 / z1, 1), point2;

    point2 = reprojectImg0_to_Img1 * point1;

    Mat rst = Mat::zeros(1, 3, CV_32FC1);
    rst.at<float>(0, 0) = point2(0, 0) / point2(3, 0);
    rst.at<float>(0, 1) = point2(1, 0) / point2(3, 0);
    rst.at<float>(0, 2) = point2(2, 0) / point2(3, 0);
    rst.at<float>(0, 2) = 1 / rst.at<float>(0, 2);

    return rst;
}


/*
This function compares two points position and depth
output = 1: match
-1: unmatch
0: nothing particular

*/
Mat
decision(Mat p1, Mat p2, float threshloc, float threshdepth, float realdepthscale, float scalethresh, float orithresh) {
    //now we are supposing that in each Mat the scale
    //col0	col1	col2	col3		col4		col5	col6
    //x		y		z		xscale0		yscale0		ori0	featureID
    //xscale and yscale is the same in this case
    //cout << "into function DECISION" << endl;

    Mat rst = Mat::zeros(1, 2, CV_32FC1);
    float flag = 0, flag0 = 0, flag1 = 0.;
    //float nnd = nearestneighbour;

    //location check
    float locdist = sqrt((p1.at<float>(0, 0) - p2.at<float>(0, 0)) * (p1.at<float>(0, 0) - p2.at<float>(0, 0))
                         + (p1.at<float>(0, 1) - p2.at<float>(0, 1)) * (p1.at<float>(0, 1) - p2.at<float>(0, 1)));
    //cout << "locdst  = " << locdist << endl;

    if (locdist < threshloc) {
        flag0 = 1;
    } else if (locdist > 2 * threshloc) {
        flag0 = -1;
    } else {
        flag0 = 0;
    }

    //depth check
    float depthdist = abs(p1.at<float>(0, 2) - p2.at<float>(0, 2)) * realdepthscale;
    if (depthdist <= threshdepth)flag1 = 1;
    else if (depthdist > 2 * threshdepth) { flag1 = -1; }
    else flag1 = 0;

    //put two decisions together
    if (flag0 == 1 && flag1 == 1) { flag = 1; }
    else if (flag0 == -1 && flag1 == -1) { //potentially non-match, check rotation and scale
        //check scale
        if (abs(p1.at<float>(0, 3) - p2.at<float>(0, 3)) > 2 * scalethresh) {
            //check orientation
            if (abs(p1.at<float>(0, 5) - p2.at<float>(0, 5)) > 2 * orithresh) {
                flag = -1;
            } else flag = 0;
        } else flag = 0;
    } else { flag = 0; }
    //cout << "flag is " << flag << endl;

    rst.at<float>(0, 0) = flag;
    //rst.at<float>(0, 1) = nnd;

    return rst;

}

/*
This function find the external parameters in image.txt based on the name of the image
*/
Eigen::Matrix4f getExternal(int imagename, const string &externalfile) {
    ifstream fin(externalfile);
    string s0;
    //parameters of external matrix:
    Mat para = Mat::zeros(1, 7, CV_32FC1);//QW, QX, QY, QZ;TX, TY, TZ;

    while (getline(fin, s0)) {
        vector<string> a = split(s0, " ");
        if (atoi(a[0].c_str()) == (imagename % FIRSTIMAGENAME + 1)) {
            for (int i = 0; i < para.cols; i++) {
                para.at<float>(0, i) = atof(a[i + 1].c_str());
            }
        }
    }

    cout << "get External" << endl;
    cout << "readed para" << para << endl;
    Eigen::Quaterniond q;
    q.w() = para.at<float>(0, 0);
    q.x() = para.at<float>(0, 1);
    q.y() = para.at<float>(0, 2);
    q.z() = para.at<float>(0, 3);

    Eigen::Matrix3d Rx = q.toRotationMatrix();
    Eigen::Matrix4f external = Matrix4f::Zero(4, 4);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            external(i, j) = Rx(i, j);
        }
        external(i, 3) = para.at<float>(0, 4 + i);
    }
    external(3, 3) = 1.;

    return external;
}

/*This function is to find the best rotation and scale
Mat1,Mat2: every row is the [x,y] of the point
cvSolvem calculates the X that minimize (src1 X - src2 )^2
thus we transpose the original Pnew = H*P to Pnew_T = P_T * H_T
|x1 y1| = |x01 y01| |scale(cos)		scale(sin)|
|x2 y2|   |x02 y02| |-scale(sin)	scale(cos)|
*/
Mat getRotScale(Mat p1, Mat p2) {
    //first normalize: translate the points, the center at origin
    float x1avg = 0, y1avg = 0, x2avg = 0, y2avg = 0;
    float angle, scale;
    CvMat *mat1 = cvCreateMat(p1.rows, 2, CV_32FC1);
    CvMat *mat2 = cvCreateMat(p1.rows, 2, CV_32FC1);
    CvMat *homo = cvCreateMat(2, 2, CV_32FC1);//注释中右边的矩阵，要求，求完之后就知道patch之间scale 和ori是多少

    for (int i = 0; i < p1.rows; i++) {
        x1avg += p1.at<float>(i, 0);
        y1avg += p1.at<float>(i, 1);
        x2avg += p2.at<float>(i, 0);
        y2avg += p2.at<float>(i, 1);
    }

    x1avg /= p1.rows;
    y1avg /= p1.rows;
    x2avg /= p1.rows;
    y2avg /= p1.rows;

    //first do translation to get the center at origin
    for (int i = 0; i < p1.rows; i++) {//先把translation 归一到中心原点

        cvmSet(mat1, i, 0, p1.at<float>(i, 0) - x1avg);
        cvmSet(mat1, i, 1, p1.at<float>(i, 1) - y1avg);
        cvmSet(mat2, i, 0, p2.at<float>(i, 0) - x2avg);
        cvmSet(mat2, i, 1, p2.at<float>(i, 1) - y2avg);

    }
//    cout << "going to least square..." << endl;
    //least square
    /*
    cout << "mat1" << endl;
    for (int i = 0; i < p1.rows; i++)
    cout << cvmGet(mat1, i, 0) << "\t" << cvmGet(mat1, i, 1) << endl;

    cout << "mat2" << endl;
    for (int i = 0; i < p1.rows; i++)
    cout << cvmGet(mat2, i, 0) << "\t" << cvmGet(mat2, i, 1) << endl;
    cout << "hier" << "===================================" << endl;*/

    cvSolve(mat1, mat2, homo, CV_SVD);//求这个homo
//    cout << "cvSolve Done" << endl;
    /*
    cout << "homography=====================================" << endl;
    for (int i = 0; i < 2; i++)
    cout << cvmGet(homo, i, 0) << "\t" << cvmGet(homo, i, 1) << endl;
    */
    angle = atan(cvmGet(homo, 0, 1) / cvmGet(homo, 0, 0));
    //to be refined, can not tell 180 degree
    if (angle < 0 && cvmGet(homo, 0, 1) > 0) angle += _pi;
    if (angle > 0 && cvmGet(homo, 0, 1) < 0) angle -= _pi;


    scale = sqrt(cvmGet(homo, 0, 0) * cvmGet(homo, 0, 0) + cvmGet(homo, 0, 1) * cvmGet(homo, 0, 1));
    Mat rst = Mat::zeros(1, 2, CV_32FC1); //extend to -pi to pi
    rst.at<float>(0, 0) = angle;
    rst.at<float>(0, 1) = scale;
    return rst;

}

/*check rotation and scale if it is in the threshold
output flag=1 match, flag=0 not match
*/
int checkRotScale(/*Mat point1,Mat point2,*/Mat as, float scalethresh, float orithresh) {
    int flag = 0;
    /*
    //x0, y0, z0, xscale, yscale, ori, featureID
    float scadist = abs(point2.at<float>(0, 3) - point1.at<float>(0, 3)*as.at<float>(0, 1));
    float tmpang = point1.at<float>(0, 5) + as.at<float>(0, 0);
    if (tmpang < -pi) tmpang += 2 * pi;
    if (tmpang > pi) tmpang -= 2 * pi;
    float angdist = abs(tmpang - point2.at<float>(0, 5));
    */
    if (abs(as.at<float>(0, 1) - 1) < scalethresh && abs(as.at<float>(0, 0)) < orithresh)
        //if (scadist < scalethresh && abs(as.at<float>(0, abs(as.at<float>(0, 0)) < orithresh)
        // if (scadist < scalethresh && abs(as.at<float>(0, 0)) < orithresh)
        flag = 1;

    return flag;
}


//This function reads in the internal matrix,
//in this project are all the pictures with the same internal matrix
//in the camera.txt file, the 4th line is with CAMERA_ID, MODEL, WIDTH, HEIGHT, focal length(pixel), pricipal pointx, principal pointy, otherpara[]
// http://ksimek.github.io/2013/08/13/intrinsic/
//PINHOLE model because the image is undistorted
//https://github.com/colmap/colmap/blob/master/src/base/camera_models.h
//CAMERA_ID, MODEL, WIDTH, HEIGHT, fx, fy, cx, cy
Eigen::Matrix3f readInternal(int imagename, const string &internalfile, int *width, int *height) {
    ifstream fin(internalfile);
    string s0;
    Eigen::Matrix3f internal = Matrix3f::Identity(3, 3);

    while (getline(fin, s0)) {
        vector<string> a = split(s0, " ");
        if (atoi(a[0].c_str()) == (imagename % FIRSTIMAGENAME + 1)) {
            *width = atoi(a[2].c_str());
            *height = atoi(a[3].c_str());
            internal(0, 0) = atof(a[4].c_str());//fx
            internal(1, 1) = atof(a[5].c_str());//fy
            internal(0, 2) = atof(a[6].c_str());//cx
            internal(1, 2) = atof(a[7].c_str());//cy
        }
    }

    cout << "internal matrix" << endl;
    cout << internal << endl;
    return internal;
}

Mat rotpoint(float tmpx, float tmpy, float cx, float cy, float ori0) {
    Mat trans0 = Mat::eye(3, 3, CV_32FC1);
    trans0.at<float>(0, 2) = -cx;
    trans0.at<float>(1, 2) = -cy;

    Mat rot0 = Mat::eye(3, 3, CV_32FC1);
    rot0.at<float>(0, 0) = cos(ori0);
    rot0.at<float>(0, 1) = -sin(ori0);
    rot0.at<float>(1, 1) = cos(ori0);
    rot0.at<float>(1, 0) = sin(ori0);

    Mat invtrans0 = Mat::eye(3, 3, CV_32FC1);
    invtrans0.at<float>(0, 2) = cx;
    invtrans0.at<float>(1, 2) = cy;

    Mat old = Mat::ones(3, 1, CV_32FC1);
    Mat rst = Mat::ones(3, 1, CV_32FC1);
    old.at<float>(0, 0) = tmpx;
    old.at<float>(1, 0) = tmpy;
    rst = invtrans0 * rot0 * trans0 * old;
    return rst;
}

/*This function normalize and then crop the patch*/
void crop(int picId0, Mat point0, string patchpath, const string& scaledpath)
{
    float featureId0 = point0.at<float>(0, 6);
    string patchname0 = to_string(picId0) + "_" + to_string(int(featureId0)) + ".png";
    string patchadd;

    patchadd = patchpath;

    //x0, y0, z0, xscale, yscale, ori, featureID
    int x0 = point0.at<float>(0, 0), y0 = point0.at<float>(0, 1);

    float scale0 = point0.at<float>(0, 3), ori0 = point0.at<float>(0, 5);

    //cout << "ori1 is"<<ori1 << endl;
    int head0 = 0; int tail0 = 0;

    float sigma = 0.8;

    for (int exp = 0; exp <5; exp++) {
        float thres0 = (sigma + sigma / pow(2, 1 / 3)) / 2;
        float thres1 = (sigma *pow(2, 1 / 3) + sigma) / 2;
        float thres2 = (sigma *pow(2, 2 / 3) + sigma *pow(2, 1 / 3)) / 2;
        float thres3 = (sigma * 2 + sigma*pow(2, 4 / 3)) / 2;

        if (scale0 >= thres0 && scale0 < thres1) { head0 = exp; tail0 = 1; }
        if (scale0 >= thres1 && scale0 < thres2) { head0 = exp; tail0 = 2; }
        if (scale0 >= thres2 && scale0 < thres3) {
            head0 = exp; tail0 = 3;
        }
        sigma = sigma * 2;
    }

    string picadd0 = scaledpath + to_string(head0) + "_" + to_string(tail0) + "/IMG_" + to_string(picId0) + ".JPG";

    //cropping patch
    Mat patch = Mat::zeros(64, 64, CV_32FC3);

    //another try, since the scale method we are doing is not working
    //use opencv GaussianBlur()
    //cout << "patch0 session:ori1 " << ori1 << endl;

    //patch0 ==========================================================================================
    Mat img0 = imread(picadd0);
    //imshow("img0", img0);
    //imshow(picahh1);
    Mat patch0;
    //Mat img0blur;
    Mat img0blur = img0.clone();
    //GaussianBlur(img0, img0blur, Size(0,0), scale0, scale0, BORDER_DEFAULT);

    //int a0 = 32 ;//multiplication

    int a0;
    if (head0 == 0) {
        a0 = 32 * 2 * scale0;
        x0 *= 2;
        y0 *= 2;
    }
    else {
        a0 = 32 * scale0 / pow(2, head0 - 1);
        x0 /= pow(2, head0 - 1);
        y0 /= pow(2, head0 - 1);
    }


    Mat trans0 = Mat::eye(3, 3, CV_32FC1);
    trans0.at<float>(0, 2) = -x0;
    trans0.at<float>(1, 2) = -y0;

    Mat rot0 = Mat::eye(3, 3, CV_32FC1);
    rot0.at<float>(0, 0) = cos(ori0);
    rot0.at<float>(0, 1) = -sin(ori0);
    rot0.at<float>(1, 1) = cos(ori0);
    rot0.at<float>(1, 0) = sin(ori0);

    Mat invtrans0 = Mat::eye(3, 3, CV_32FC1);
    invtrans0.at<float>(0, 2) = x0;
    invtrans0.at<float>(1, 2) = y0;

    Mat oldcorners0 = Mat::ones(3, 4, CV_32FC1);
    Mat newcorners0 = Mat::ones(3, 4, CV_32FC1);

    oldcorners0.at<float>(0, 0) = x0 - a0;
    oldcorners0.at<float>(1, 0) = y0 - a0;
    oldcorners0.at<float>(0, 1) = x0 + a0;
    oldcorners0.at<float>(1, 1) = y0 - a0;
    oldcorners0.at<float>(0, 2) = x0 - a0;
    oldcorners0.at<float>(1, 2) = y0 + a0;
    oldcorners0.at<float>(0, 3) = x0 + a0;
    oldcorners0.at<float>(1, 3) = y0 + a0;

    newcorners0 = invtrans0 * rot0 * trans0 * oldcorners0;
    /*
    cout << "patch 1, the rotation angle is " << ori0 << endl;
    cout << " the 4 corners are" << endl;
    cout << "up left point" << newcorners0.at<float>(0, 0) << "\t" << newcorners0.at<float>(1, 0) << endl;
    cout << "up tight point" << newcorners0.at<float>(0, 1) << "\t" << newcorners0.at<float>(1, 1) << endl;
    cout << "down left point" << newcorners0.at<float>(0, 2) << "\t" << newcorners0.at<float>(1, 2) << endl;
    cout << "down right point" << newcorners0.at<float>(0, 3) << "\t" << newcorners0.at<float>(1, 3) << endl;
    */
    Point2f src0[4], dst0[4];
    //up left point
    src0[0].x = newcorners0.at<float>(0, 0);
    src0[0].y = newcorners0.at<float>(1, 0);
    dst0[0].x = 0;
    dst0[0].y = 0;
    //up right
    src0[1].x = newcorners0.at<float>(0, 1);
    src0[1].y = newcorners0.at<float>(1, 1);
    dst0[1].x = 64;
    dst0[1].y = 0;
    //down left
    src0[2].x = newcorners0.at<float>(0, 2);
    src0[2].y = newcorners0.at<float>(1, 2);
    dst0[2].x = 0;
    dst0[2].y = 64;
    //down right
    src0[3].x = newcorners0.at<float>(0, 3);
    src0[3].y = newcorners0.at<float>(1, 3);
    dst0[3].x = 64;
    dst0[3].y = 64;

    Mat transMatrix0 = Mat::zeros(3, 3, CV_32FC1);
    transMatrix0 = getPerspectiveTransform(src0, dst0);
    warpPerspective(img0blur, patch0, transMatrix0, patch.size(), INTER_LINEAR);
    //imshow("patch0", patch0);
    imwrite(patchadd + patchname0, patch0);
}