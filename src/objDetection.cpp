/*
  Identify image files in a directory
*/
#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>

#include "classify.hpp"
#include "csv_util.h"
#include "image.hpp"
#include "process.hpp"

using namespace cv;
using namespace std;
using namespace process;
using namespace image;
using namespace classify;

/*
  Given a directory on the command line, scans through the directory for image files.
  Return the top matched results.
 */
int main(int argc, char *argv[]) {
    // Mat img = imread("../data/img1P3.png", IMREAD_COLOR);
    // namedWindow("image", WINDOW_AUTOSIZE);
    // cv::imshow("image", img);

    vector<cv::Mat> images;
    char dirname[256];

    strcpy(dirname, "../data/testing");
    process::loadImages(images, dirname);
    cout << "number of images: " << images.size() << "\n\n";

    // test on csv
    /*
    char csvDirName[256];
    strcpy(csvDirName, "../data/csv/trainingImages.csv");
    for (int i = 0; i < images.size(); i++) {
        char fname[64];
        snprintf(fname, sizeof fname, "%d_%d", images[i].rows, images[i].cols);
        // https://stackoverflow.com/questions/20980723/convert-mat-to-vector-float-and-vectorfloat-to-mat-in-opencv

        vector<float> V;
        V.assign((float *)images[i].datastart, (float *)images[i].dataend);
        append_image_data_csv(csvDirName, fname, V, 1);
    }

    vector<char *> filenames;
    vector<vector<float>> data;
    read_image_data_csv(csvDirName, filenames, data, 0);
    vector<cv::Mat> loadedImages;
    for (int i = 0; i < data.size(); i++) {
        // cv::Mat temp = cv::Mat(data[i]).clone();
        // memcpy(temp.data, i.data(), i.size() * sizeof(float));
        // temp = cv::imread(i);
        // cv::Mat dest = temp.reshape(3, stoi(rows));

        string fn(filenames[i]);
        string delimiter = "_";
        string rows = fn.substr(0, fn.find(delimiter));
        string cols = fn.substr(fn.find(delimiter) + 1);

        cv::Mat temp = Mat(stoi(rows), stoi(cols), CV_8UC3);
        memcpy(temp.data, data[i].data(), data[i].size() * sizeof(float));

        cv::imshow("temp", temp);
    }
    */

    vector<cv::Mat> trainingImgs;
    vector<std::string> labels;
    char trainingDir[256];
    strcpy(trainingDir, "../data/training");
    process::loadTrainingImages(trainingImgs, trainingDir, labels);

    // get feature vectors of these training models
    vector<ImgData> traingImgData;
    for (int i = 0; i < trainingImgs.size(); i++) {
        traingImgData.push_back(image::calculateImgData(trainingImgs[i]));
        traingImgData[i].label = labels[i];
        cout << i << ":  " << labels[i] << "\n";
    }
    cout << "training images & labels are loaded. \n"
         << std::endl;

    // get mean feature vector
    Feature standardFeature = classify::calculateFeatureStdDev(traingImgData);
    cout << "fill: " << standardFeature.fillRatio << endl;
    cout << "bbox: " << standardFeature.bboxDimRatio << endl;
    cout << "axis: " << standardFeature.axisDimRatio << endl;

    for (int i = 0; i < standardFeature.huMoments.size(); i++) {
        cout << "hu:   " << standardFeature.huMoments[i] << endl;
    }

    // get training images' map of labels
    map<string, vector<Feature>> db;
    for (ImgData i : traingImgData) {
        db[i.label].push_back(i.features);
    }
    // classify
    // classify::classifyObject(src, db, standardFeature);

    // vector<pair<Mat, Mat>> res = image::thresholdImages(images);
    // vector<Mat> results;
    // for (int i = 0; i < res.size(); i++) {
    //     results.push_back(image::cleanUpBinary(res[i].second));
    // }

    // vector<pair<Mat, Mat>> res = image::connectedComponents(images);
    // vector<Mat> results;
    // for (int i = 0; i < res.size(); i++) {
    //     results.push_back(res[i].second);
    // }

    vector<ImgData> res;
    for (int i = 0; i < images.size(); i++) {
        ImgData imgData = image::calculateImgData(images[i]);
        imgData.label = classify::classifyObject(imgData.features, db, standardFeature);
        string displayName = "image-" + to_string(i);
        process::displayResultsWithFeatures(displayName, imgData);
    }

    // NOTE: must add waitKey, or the program will terminate, without showing the result images
    waitKey(0);
    printf("Terminating\n");

    return (0);
}