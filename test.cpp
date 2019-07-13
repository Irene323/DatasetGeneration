#include "DatasetGenerator.h"

int rejectionFlag(Mat &table0, int picwidth0, int picheight0, Mat &depthtable0, Eigen::Matrix4f &external0, Eigen::Matrix3f &internal0,
                 Mat &table2, int picwidth2, int picheight2, Mat &depthtable2, Eigen::Matrix4f &external2, Eigen::Matrix3f &internal2,
                 const string &depthscalefactor,
                 int feature1, int feature2, const vector<float> &pyramid) {
    srand((unsigned) time(nullptr));
    float realdepthscalefactor = atof(
            depthscalefactor.c_str());//22.8232 / 0.918737;// the dense map should be scaled to real size and compare
    float locthresh = 5, depththresh = 0.3;//depththresh unit cm in full.ply
    float scalethresh = 0.25, orithresh = _pi / 8;

    //match0 and match2 saves the information of the pre-positive pairs,
    //in every round of comparison of two pictures!
    //before scale and ori check
    //the same line in match0 and match1 correspond to the two points in a pair
    //each vector contains the feature point, which has 7 elements:x0,y0,z0,xscale,yscale,ori,featureID(in this pic)
    for (int i = 0; i < table0.rows; i++) {
        if (i != feature1) {
            // feature in matchlist.txt start with index 0
            // i start with index 0
            // feature in table0.at<float>(i,0) start with index 1 (so 1 bigger)
//                    cout << "i: " << i << "\t table0.at<float>(i, 0): " << table0.at<float>(i, 0) << endl;
            continue;
        }
        if (table0.at<float>(i, 3) > 25.6) {
            return 0; // 0 - no feature found in the radius: too large scale
        }

        //point in pic1: x0,y0,z0,xscale0,yscale0,ori0,featureID
        Mat point0 = Mat::zeros(1, 7, CV_32FC1);
        point0.at<float>(0, 0) = table0.at<float>(i, 1);//x0
        point0.at<float>(0, 1) = table0.at<float>(i, 2);//y0
        point0.at<float>(0, 3) = table0.at<float>(i, 3);//xscale0
        point0.at<float>(0, 4) = table0.at<float>(i, 4);//yscale0
        point0.at<float>(0, 5) = table0.at<float>(i, 5);//orientation
        point0.at<float>(0, 6) = table0.at<float>(i, 0);//feature ID

        int numofnearest = 0;//count how many nearest point of this point in pic1, if more than 1, we do not pick this point to avoid aliasing
        Mat point2final = Mat::zeros(1, 7, CV_32FC1);

        //if the point is too close to the edge, we do not take them into account
        if (ifTooCloseToEdge(pyramid, point0, picwidth0, picheight0)) {
            return 1; // 1 - no feature found in the radius: too close to the edge
        }
//                float distthresh0 = sqrt(2) * 32 * point0.at<float>(0, 3);
//                if ((point0.at<float>(0, 0) < distthresh0 || point0.at<float>(0, 0) > picwidth0 - distthresh0
//                     || point0.at<float>(0, 1) < distthresh0 || point0.at<float>(0, 1) > picheight0 - distthresh0)) {
//                    return 1; // 1 - no feature found in the radius: too close to the edge
//                }

        //cout << "start checking point " << point0.at<float>(0, 6) << endl;

        //transform to pic2
        //notice that x and y is col and row, not row and col
        point0.at<float>(0, 2) = depthtable0.at<float>(cvRound(table0.at<float>(i, 2)),
                                                       cvRound(table0.at<float>(i, 1)));
        Mat newloc = locationtrans(point0.at<float>(0, 0), point0.at<float>(0, 1), point0.at<float>(0, 2),
                                   internal0, internal2, external0, external2);

        //point transformed: x1,y1,z1,xscale1,yscale1,ori1
        Mat point1 = point0.clone();
        for (int p = 0; p < 3; p++) {
            point1.at<float>(0, p) = newloc.at<float>(0, p);
        }

        //for every feature point in pic2
        for (int j = 0; j < table2.rows; j++) {
            if (j != feature2) {
                continue;
            }
            if (table2.at<float>(j, 3) > 25.6) {
                return 0; // 0 - no feature found in the radius: too large scale
            }
            //point in pic2: x2,y2,z2,xscale2,yscale2,ori2,featre_ID
            Mat point2 = Mat::zeros(1, 7, CV_32FC1);
            point2.at<float>(0, 0) = table2.at<float>(j, 1);//x2
            point2.at<float>(0, 1) = table2.at<float>(j, 2);//y2
            point2.at<float>(0, 3) = table2.at<float>(j, 3);//xscale2
            point2.at<float>(0, 4) = table2.at<float>(j, 4);//yscale2
            point2.at<float>(0, 5) = table2.at<float>(j, 5);//orientation
            point2.at<float>(0, 6) = table2.at<float>(j, 0);//feature ID

            if (ifTooCloseToEdge(pyramid, point2, picwidth2, picheight2)) {
                return 1; // 1 - no feature found in the radius: too close to the edge
            }
//                    float radius2 = sqrt(2) * 32 * point2.at<float>(0, 3);
//                    if (point2.at<float>(0, 0) < radius2 || point2.at<float>(0, 0) > picwidth2 - radius2
//                        || point2.at<float>(0, 1) < radius2 || point2.at<float>(0, 1) > picheight2 - radius2) {
//                        return 1; // 1 - no feature found in the radius: too close to the edge
//                    }


            point2.at<float>(0, 2) = depthtable2.at<float>(cvRound(table2.at<float>(j, 2)),
                                                           cvRound(table2.at<float>(j, 1)));

            //compare transformed point0: point1 and point2
            //cout << "point2 is pic " <<*it1<< "number "<<point2.at<float>(0,6)<< endl;

            int flag = decision(point1, point2, locthresh, depththresh, realdepthscalefactor, scalethresh,
                                orithresh);

            if (flag != 1) {
                return 3; // 3 - didn't pass position or depth threshold
            }

            numofnearest++;
            if (numofnearest > 1) {
                return 2; // 2 - multiple features in the radius
            }
            point2final = point2.clone();//nearest feature ID
        }// point 1 with all points in pic2 compare done;

        if (numofnearest == 1) {
            //check rotation and scale

            //float radius = 40 * point0.at<float>(0, 3);//changeable, emperical
            //float factor = point2final.at<float>(0, 3);
            int r = 32;
            int s0 = point0.at<float>(0, 3);
            int s1 = point2final.at<float>(0, 3);

            Mat local0 = Mat::zeros((2 * r + 1) * (2 * r + 1), 2, CV_32FC1); //4225*2
            Mat local2 = Mat::zeros((2 * r + 1) * (2 * r + 1), 2, CV_32FC1);

            int num = 0;
            Mat point = Mat::zeros(1, 7, CV_32FC1);

            for (int i = -r; i <= r; i++) {
                //for every row,i.e. the y is the same
                //cout << "i is " << i << endl;

                for (int j = -r; j <= r; j++) {
                    point.at<float>(0, 1) = point0.at<float>(0, 1) + i * s0;
                    //cout << "j is " << j << endl;
                    point.at<float>(0, 0) = point0.at<float>(0, 0) + j * s0;

                    Mat rstpoint = rotpoint(point0.at<float>(0, 0) + j * s0, point0.at<float>(0, 1) + i * s0,
                                            point0.at<float>(0, 0), point0.at<float>(0, 1),
                                            point0.at<float>(0, 5));

                    point.at<float>(0, 0) = rstpoint.at<float>(0, 0);
                    point.at<float>(0, 1) = rstpoint.at<float>(1, 0);

                    if (point.at<float>(0,0)>=depthtable2.cols || point.at<float>(0,1)>=depthtable2.rows){
                        cout << point.at<float>(0,0) << endl;
                        cout << point.at<float>(0,1) << endl;
                    }
                    point.at<float>(0, 2) = depthtable2.at<float>(int(point.at<float>(0, 1)),
                                                                  int(point.at<float>(0, 0)));

                    //cout << "local0 is "<<local0.at<float>(num,0)<<"\t"<< local0.at<float>(num, 1) << endl;
                    Mat transloc = locationtrans(point.at<float>(0, 0), point.at<float>(0, 1),
                                                 point.at<float>(0, 2), internal0, internal2, external0,
                                                 external2);
                    local0.at<float>(num, 0) = transloc.at<float>(0, 0);
                    local0.at<float>(num, 1) = transloc.at<float>(0, 1);


                    Mat rstpoint2 = rotpoint(point2final.at<float>(0, 0) + j * s1,
                                             point2final.at<float>(0, 1) + i * s1, point2final.at<float>(0, 0),
                                             point2final.at<float>(0, 1), point2final.at<float>(0, 5));

                    local2.at<float>(num, 0) = rstpoint2.at<float>(0, 0);
                    local2.at<float>(num, 1) = rstpoint2.at<float>(1, 0);


                    //cout << "local1 is " << local2.at<float>(num, 0) << "\t" << local2.at<float>(num, 1) << endl;
                    num++;
                }

            }
            Mat rotSca = Mat::zeros(1, 2, CV_32FC1);
            rotSca = getRotScale(local0, local2);
            //checkRotScale(Mat point1, Mat point2, Mat as, float scalethresh, float orithresh)
            int finalflag = checkRotScale(/*point1, point2final,*/rotSca, scalethresh,
                                                                  orithresh);// finalflag =1 match, =0 not match
            if (finalflag == 0) {
                return 4; // 4 - didn't pass rotation or scale check
            } else {
                return -1; // -1 - matched
            }
        }
    }
}


int main(int argc, char **argv) {
//    function test
/*
    readDataIntoMat(2042, "feature", "/home/yirenli/dev/DatasetGenerationHelper/out/keypoints/", "");
    readDataIntoMat(2042, "depth", "", "/home/yirenli/dev/DatasetGenerationHelper/out/depth_maps/");

    cout << getExternal(2042, "/home/yirenli/dev/DatasetGenerationHelper/out/cameras_images_txt/images.txt");
    int width = 0;
    int height = 0;
    cout << readInternal(2057, "/home/yirenli/dev/DatasetGenerationHelper/out/cameras_images_txt/cameras.txt", &width, &height);

    float sigma = 1.6;
    float s = 3.0;
    for (int a = 0; a < 4; a++) {
        for (int i = 0; i <= s + 2; i++) {//s+3
            double d = pow(2, (double) a) * pow(2, (double) i / s) * sigma;
            cout << "orig: " << d << endl;
            printf("%.1f\n", d);
        }
        cout << endl;
    }
 */

//    visualize matches
/*
    ifstream matchtxt("/home/yirenli/test_matching/matchlist.txt");
//    ifstream matchtxt("/home/yirenli/dev/DatasetGeneration/out/ao/txt/temp_match.txt");
    string s;
    string patchdir = "/home/yirenli/dev/DatasetGeneration/out/ao/all_patches/";
    int c = -1;
    int num = 100;
    Mat out = Mat::zeros(64 * num, 128, CV_32FC3);
    while (getline(matchtxt, s)) {
        c++;
//        if (c<num){
        if(c >= num && c<num+100) {
//        if(c >= num+100 && c<num+200) {
            vector <string> a = split(s, "\t");
            string p1 = a[0] + "_" + a[1] + ".png";
            string p2 = a[2] + "_" + a[3] + ".png";
            cout << c << endl;
            cout << patchdir + p1 << endl;
            Mat patch1 = imread(patchdir + p1);
//            imshow("patch1", patch1);
//            waitKey(0);
            cout << patchdir + p2 << endl;
            Mat patch2 = imread(patchdir + p2);
//            imshow("patch2", patch2);
//            waitKey(0);
            if (!patch1.data || !patch2.data) {
                continue;
            }

//            out(Rect(0, c * 64, 64, 64)) = patch1;
            patch1.copyTo(out(Rect(0, c % num * 64, 64, 64)));
//            out(Rect(64, c * 64, 64, 64)) = patch2;
            patch2.copyTo(out(Rect(64, c % num * 64, 64, 64)));
//            cvShowImage("out", out);
//            waitKey(0);
        }
    }
    imwrite("/home/yirenli/test_matching/matches.png", out);
//    imwrite("../out/ao/matches.png", out);
*/

/*
 * 0 - no feature found in the radius: too large scale
 * 1 - no feature found in the radius: too close to the edge
 * 2 - multiple features in the radius
 * 3 - didn't pass position or depth threshold
 * 4 - didn't pass rotation or scale check
 *-1 - matched
 */
    ifstream matchtxt("/home/yirenli/test_matching/matchlist.txt");
    string outdir = "../out/af/filter_test/colmap.txt";
    ofstream file(outdir, std::ios::app);
    if (file.fail()) {
        cout << "Can not open file: " << outdir << endl;
    }
    string s;

    const string feature_address = "/home/yirenli/dev/DatasetGenerationHelper/out/af/keypoints/";
    const string depth_address = "/home/yirenli/dev/DatasetGenerationHelper/out/af/depth_maps/";
    const string internalfile = "/home/yirenli/dev/DatasetGenerationHelper/out/af/cameras_images_txt/cameras.txt";
    const string externalfile = "/home/yirenli/dev/DatasetGenerationHelper/out/af/cameras_images_txt/images.txt";
    const string namelistfile = "../out/af/filter_test/namelist.txt";
    const string depthscalefactor = "18.1026";

    vector<int> namelist = readDataIntVec(namelistfile);
    FIRSTIMAGENAME = namelist[0];
    cout << "FIRSTIMAGENAME after changed in generateList: " << FIRSTIMAGENAME << endl;

    int picwidth0 = 0, picheight0 = 0;
    cout << "pic1: " << namelist[0] << endl;
    //read feature detection
    Mat table0 = readDataIntoMat(namelist[0], "feature", feature_address, depth_address);
    //read depth map
    Mat depthtable0 = readDataIntoMat(namelist[0], "depth", feature_address, depth_address);
    //external matrix
    Eigen::Matrix4f external0 = getExternal(namelist[0], externalfile);
    Eigen::Matrix3f internal0 = readInternal(namelist[0], internalfile, &picwidth0, &picheight0);
    cout << "picwidth0: " << picwidth0 << endl;
    cout << "picheight0: " << picheight0 << endl;

    int picwidth2 = 0, picheight2 = 0;
    cout << "pic2: " << namelist[1] << endl;
    Mat table2 = readDataIntoMat(namelist[1], "feature", feature_address, depth_address);
    Mat depthtable2 = readDataIntoMat(namelist[1], "depth", feature_address, depth_address);
    Eigen::Matrix4f external2 = getExternal(namelist[1], externalfile);
    Eigen::Matrix3f internal2 = readInternal(namelist[1], internalfile, &picwidth2, &picheight2);
    cout << "picwidth2: " << picwidth2 << endl;
    cout << "picheight2: " << picheight2 << endl;

    vector<float> pyramid;
    for (int i = 0; i < 13; i++) {//first four octaves
        pyramid.push_back(1.6 * pow(2, (float) i / 3));
        cout << pyramid[i] << endl;
    }

    while (getline(matchtxt, s)) {
        vector<string> a = split(s, "\t");
        cout << a[0] + "_" + a[1] + "\t" + a[2] + "_" + a[3] << endl;
        // feature in matchlist.txt start with index 0
        int flag = rejectionFlag(table0, picwidth0, picheight0, depthtable0, external0, internal0,
                                 table2, picwidth2, picheight2, depthtable2, external2, internal2,
                                 depthscalefactor, atoi(a[1].c_str()), atoi(a[3].c_str()), pyramid);
        file << a[0] + "_" + a[1] + "\t" + a[2] + "_" + a[3] + "\t" << flag << endl;
    }
}



