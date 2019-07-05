#include "DatasetGenerator.h"

bool generateAllPatches(string feature_address, string depth_address, string internalfile, string externalfile,
                        string namelistfile, string scaledpath, string patchpath) {
    bool flag = false;
    int picwidth = NULL, picheight = NULL;

    //namelist of the now processing images
    vector<int> namelist = readDataIntVec(namelistfile);

    //for every picture
#pragma omp parallel for num_threads(8)
    for (vector<int>::iterator it = namelist.begin(); it != namelist.end() ; it++) {

        cout << "pic1: " << *it << endl;
        //read feature detection
        Mat table0 = readDataIntoMat(*it, "feature", feature_address, depth_address);
        //read depth map
        Mat depthtable0 = readDataIntoMat(*it, "depth", feature_address, depth_address);
        //external matrix
        Eigen::Matrix4f external0 = getExternal(*it, externalfile);
        Eigen::Matrix3f internal = readInternal(*it, internalfile, &picwidth, &picheight);

        for (int i = 0; i < table0.rows; i++) {

            if (table0.at<float>(i, 3) > 25.3) continue; //too large scale, not in the first 4 octaves, may be wrong, skip
            if (table0.at<float>(i, 0) == 0) break;//all the features have been read
            //point in pic1: x0,y0,z0,xscale0,yscale0,ori0,featureID

            Mat point0 = Mat::zeros(1, 7, CV_32FC1);
            point0.at<float>(0, 0) = table0.at<float>(i, 1);//x0
            point0.at<float>(0, 1) = table0.at<float>(i, 2);//y0
            point0.at<float>(0, 3) = table0.at<float>(i, 3);//xscale0
            point0.at<float>(0, 4) = table0.at<float>(i, 4);//yscale0
            point0.at<float>(0, 5) = table0.at<float>(i, 5);//orientation
            point0.at<float>(0, 6) = table0.at<float>(i, 0);//feature ID

            //if the point is too close to the edge, we do not take them into account
            float distthresh0 = sqrt(2) * 32 * point0.at<float>(0, 3);
            if ((point0.at<float>(0, 0) < distthresh0 || point0.at<float>(0, 0) > picwidth - distthresh0
                 || point0.at<float>(0, 1) < distthresh0 || point0.at<float>(0, 1) > picheight - distthresh0)) {
                continue;
            }

            crop(*it, point0, patchpath, scaledpath);
        }
    }
    flag = true;
    return flag;
}

bool generateList(string feature_address, string depth_address, string internalfile, string externalfile,
                  string namelistfile, string scaledpath, string matchpath, string nonmatchpath){
    
}

/*
 Usage:
 /home/yirenli/dev/DatasetGenerationHelper/out/keypoints/ /home/yirenli/dev/DatasetGenerationHelper/out/depth_maps/ /home/yirenli/dev/DatasetGenerationHelper/out/cameras_images_txt/cameras.txt /home/yirenli/dev/DatasetGenerationHelper/out/cameras_images_txt/images.txt /home/yirenli/dev/DatasetGenerationHelper/out/namelist.txt /home/yirenli/dev/DatasetGenerationHelper/out/scale/ ../out/matchpatch/ ../out/nonmatchpatch/ ../out/match ../out/nonmatch
*/
int main(int argc, char **argv) {
    if (argc != 11) {
        cout << "Incorrect number of arguments given." << endl;
        cout << "10 required in this order:" << endl;
        cout << "- feature txt address (folder with many files with numbers as names)" << endl;
        cout << "- depth map address (.txt)" << endl;
        cout << "- internal file (cameras.txt)" << endl;
        cout << "- external file (images.txt)" << endl;
        cout << "- namelist file" << endl;
        cout << "- scaled images path" << endl;
        cout << "- match patch dir" << endl;
        cout << "- nonmatch patch path" << endl;
        cout << "- matchtablepath" << endl;
        cout << "- nonmatch table path" << endl;
        return 1;
    }
    string feature_address, depth_address, internalfile, externalfile, namelistfile, scaledpath, matchpath, nonmatchpath, matchtablepath, nonmatchtablepath;
    feature_address = argv[1];
    depth_address = argv[2];
    internalfile = argv[3];
    externalfile = argv[4];
    namelistfile = argv[5];
    scaledpath = argv[6];
    matchpath = argv[7];
    nonmatchpath = argv[8];
    matchtablepath = argv[9];
    nonmatchtablepath = argv[10];

    //若不存在 创建完没有打开
    char *matchtable = const_cast<char *>(matchtablepath.c_str());
    fstream _mfile;
    _mfile.open(matchtable, ios::in);
    if (!_mfile) {
        ofstream mfile(matchtable);
        mfile.close();
    }
    _mfile.close();

    char *nonmatchtable = const_cast<char *>(nonmatchtablepath.c_str());
    ofstream nmfile(nonmatchtable);
    fstream _nmfile;
    _nmfile.open(nonmatchtable, ios::in);
    if (!_nmfile) {
        ofstream nmfile(nonmatchtable);
        nmfile.close();
    }
    _nmfile.close();


    float realdepthscalefactor = 22.8232 / 0.918737;// the dense map should be scaled to real size and compare
    float locthresh = 5, depththresh = 0.3;//depththresh unit cm in full.ply
    float scalethresh = 0.25, orithresh = _pi / 8;
//    vector<int> picwidth, picheight;


    //namelist of the now processing images
    vector<int> namelist = readDataIntVec(namelistfile);

    //for every picture
#pragma omp parallel for num_threads(8)
    for (vector<int>::iterator it = namelist.begin(); it != namelist.end() - 1; it++) {
        int picwidth0 = 0, picheight0 = 0;

        cout << "pic1: " << *it << endl;
        //read feature detection
        Mat table0 = readDataIntoMat(*it, "feature", feature_address, depth_address);
        //read depth map
        Mat depthtable0 = readDataIntoMat(*it, "depth", feature_address, depth_address);
        //external matrix
        Eigen::Matrix4f external0 = getExternal(*it, externalfile);
        Eigen::Matrix3f internal0 = readInternal(*it, internalfile, &picwidth0, &picheight0);
        cout << "picwidth0: " << picwidth0 << endl;
        cout << "picheight0: " << picheight0 << endl;

        //compare with all the other pictures(that are behind this picture in the namelist)

        for (vector<int>::iterator it1 = it + 1; it1 != namelist.end(); it1++) {
            int picwidth2 = 0, picheight2 = 0;
            //if (*it1 != 6760) continue;
            if (*it1 - *it > 3)break;
            cout << "pic2: " << *it1 << endl;
            Mat table2 = readDataIntoMat(*it1, "feature", feature_address, depth_address);
            Mat depthtable2 = readDataIntoMat(*it1, "depth", feature_address, depth_address);
            Eigen::Matrix4f external2 = getExternal(*it1, externalfile);
            Eigen::Matrix3f internal2 = readInternal(*it1, internalfile, &picwidth2, &picheight2);
            cout << "picwidth2: " << picwidth2 << endl;
            cout << "picheight2: " << picheight2 << endl;

            //match0 and match2 saves the information of the pre-positive pairs,
            //in every round of comparison of two pictures!
            //before scale and ori check
            //the same line in match0 and match1 correspond to the two points in a pair
            //each vector contains the feature point, which has 7 elements:x0,y0,z0,xscale,yscale,ori,featureID(in this pic)

            for (int i = 0; i < table0.rows; i++) {
                //if (table0.at<float>(i, 0) != 718) continue;
                if (table0.at<float>(i, 3) > 25.3)
                    continue; //too large scale, not in the first 4 octaves, may be wrong, skip
                if (table0.at<float>(i, 0) == 0) break;//all the features have been read
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
                float distthresh0 = sqrt(2) * 32 * point0.at<float>(0, 3);
                if ((point0.at<float>(0, 0) < distthresh0 || point0.at<float>(0, 0) > picwidth0 - distthresh0
                     || point0.at<float>(0, 1) < distthresh0 || point0.at<float>(0, 1) > picheight0 - distthresh0)) {
                    continue;
                }

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
                    //if (table2.at<float>(j, 0) != 469) continue;
                    if (table2.at<float>(j, 3) > 25.3) continue;
                    if (table2.at<float>(j, 0) == 0) {
                        //cout << "end of points in pic2=================" << endl;
                        break;
                    }
                    //point in pic2: x2,y2,z2,xscale2,yscale2,ori2,featre_ID
                    Mat point2 = Mat::zeros(1, 7, CV_32FC1);
                    point2.at<float>(0, 0) = table2.at<float>(j, 1);//x2
                    point2.at<float>(0, 1) = table2.at<float>(j, 2);//y2
                    point2.at<float>(0, 3) = table2.at<float>(j, 3);//xscale2
                    point2.at<float>(0, 4) = table2.at<float>(j, 4);//yscale2
                    point2.at<float>(0, 5) = table2.at<float>(j, 5);//orientation
                    point2.at<float>(0, 6) = table2.at<float>(j, 0);//feature ID

                    float radius2 = sqrt(2) * 32 * point2.at<float>(0, 3);
                    if (point2.at<float>(0, 0) < radius2 || point2.at<float>(0, 0) > picwidth2 - radius2
                        || point2.at<float>(0, 1) < radius2 || point2.at<float>(0, 1) > picheight2 - radius2) {
                        continue;
                    }


                    point2.at<float>(0, 2) = depthtable2.at<float>(cvRound(table2.at<float>(j, 2)),
                                                                   cvRound(table2.at<float>(j, 1)));

                    //compare transformed point0: point1 and point2
                    //caomarest = [flag, nearestneighbourdst(if flag = 1)]
                    Mat comparerst = Mat::zeros(1, 2, CV_32FC1);
                    //cout << "point2 is pic " <<*it1<< "number "<<point2.at<float>(0,6)<< endl;
                    comparerst = decision(point1, point2, locthresh, depththresh, realdepthscalefactor, scalethresh,
                                          orithresh);

                    float flag = comparerst.at<float>(0, 0);

                    if (flag == 0.) { continue; }

                    if (flag == -1.) {
                        //crop non-match
                        //check if the point is too close to edge
                        writePair(*it, point0.at<float>(0, 6), *it1, point2.at<float>(0, 6), "nonmatch", matchtablepath,
                                  nonmatchtablepath);
                        //normpatch(*it, point0, *it1, point2, "unmatch");
                    }

                    if (flag == 1.) {
                        numofnearest++;
                        if (numofnearest > 1) continue;
                        point2final = point2.clone();
                    }//nearest feature ID
                }// point 1 with all points in pic2 compare done;

                float tmpangle, tmpscale;
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
                    int finalflag = checkRotScale(/*point1, point2final,*/rotSca, scalethresh, orithresh);
                    if (finalflag == 1) {
                        //cout << "after rotation and scale check a match is found!" << endl;
                        //cout << "for pic " << *it << " point " << point0.at<float>(0, 6) << endl;
                        //cout << "pic2 is " << *it1 << " point2 " << point2.at<float>(0, 6) << " is matching" << endl;
//                        normpatch(*it, point0, *it1, point2final, "match", matchpath, nonmatchpath, scaledpath);
                        writePair(*it, point0.at<float>(0, 6), *it1, point2final.at<float>(0, 6), "match",
                                  matchtablepath, nonmatchtablepath);

                    }


                }

            }
        }


    }
    return 0;
}
