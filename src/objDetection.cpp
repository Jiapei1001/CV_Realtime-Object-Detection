/*
  Identify image files in a directory
*/
#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>

#include "cascade.hpp"
#include "classify.hpp"
#include "csv_util.h"
#include "image.hpp"
#include "process.hpp"

using namespace cv;
using namespace std;
using namespace process;
using namespace image;
using namespace classify;
using namespace cascade;

/*
  Given a directory on the command line, scans through the directory for image files.
  Return the top matched results.
 */
int main(int argc, char *argv[]) {
    // Training Images
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
        // cout << i << ":  " << labels[i] << "\n";
    }
    cout << "Training images & their labels are loaded.\n"
         << endl;

    // get mean feature vector
    Feature standardFeature = classify::calculateFeatureStdDev(traingImgData);

    // get training images' map of labels
    map<string, vector<Feature>> db;
    for (ImgData i : traingImgData) {
        db[i.label].push_back(i.features);
    }

    // Calculation method - Euclidean distance or K-Nearest Neighbor
    cout << "Enter 'e' for Euclidean distance method, or 'k' for K-Nearest Neighbor method, or 'c' for Haar Cascade\n";
    bool finish = false;
    string method;
    while (!finish) {
        cin >> method;
        if (method == "e" || method == "k" || method == "c") {
            finish = true;
        }
    }

    if (method == "c") {
        // Reference: Haar-cascade Detection
        // https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
        cascade::cascadeVideoStream();
        return 0;
    }

    // Video or Image
    cout << "Enter 'v' for video processing, or 'p' for photo processing\n";
    string mode;
    finish = false;
    bool isVideo = false;
    while (!finish) {
        cin >> mode;
        if (mode == "p") {
            isVideo = false;
            finish = true;
        } else if (mode == "v") {
            isVideo = true;
            finish = true;
        }
    }

    if (isVideo) {
        cout << "\nStart video mode\n";
        // process::classifyObjectByVideo(db, standardFeature);
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if (!capdev->isOpened()) {
            printf("Unable to open video device\n");
            return (-1);
        }

        // get some properties of the image
        cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                      (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 2);  // identifies a window, must be different from the above one for haar cascade method

        cv::Mat frame;
        for (;;) {
            *capdev >> frame;  // get a new frame from the camera, treat as a stream
            if (frame.empty()) {
                printf("frame is empty\n");
                break;
            }

            ImgData imgData = image::calculateImgData(frame);
            if (method == "e") {
                imgData.label = classify::classifyObject(imgData.features, db, standardFeature);
            } else if (method == "k") {
                imgData.label = classify::classifyObjectByKNN(imgData.features, db, standardFeature);
            }

            process::displayResultsWithFeaturesInVideoFrame(frame, imgData);

            cv::imshow("Video", frame);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }

        delete capdev;
    } else {
        cout << "\nStart image mode\n";

        vector<cv::Mat> images;
        vector<string> actualLabels;
        char dirname[256];

        strcpy(dirname, "../data/testing");
        process::loadImages(images, dirname, actualLabels);
        cout << "number of images: " << images.size() << "\n\n";

        // threshold
        // vector<pair<Mat, Mat>> res = image::thresholdImages(images);
        // vector<Mat> results;
        // for (int i = 0; i < res.size(); i++) {
        //     // results.push_back(image::cleanUpBinary(res[i].second));
        //     results.push_back(res[i].second);
        // }
        // process::displayResults(results);

        // connected components
        // vector<Mat> results;
        // for (int i = 0; i < images.size(); i++) {
        //     pair<Mat, int> temp = image::connectedComponents(images[i]);
        //     results.push_back(temp.first);
        // }
        // process::displayResults(results);

        vector<ImgData> res;
        vector<string> detectedLabels;
        for (int i = 0; i < images.size(); i++) {
            ImgData imgData = image::calculateImgData(images[i]);

            if (method == "e") {
                imgData.label = classify::classifyObject(imgData.features, db, standardFeature);
            } else if (method == "k") {
                imgData.label = classify::classifyObjectByKNN(imgData.features, db, standardFeature);
            }

            detectedLabels.push_back(imgData.label);

            string displayName = "image-" + to_string(i);
            process::displayResultsWithFeaturesAsImage(displayName, imgData);
        }

        process::buildMatrixTable(actualLabels, detectedLabels);

        // NOTE: must add waitKey, or the program will terminate, without showing the result images
        waitKey(0);
    }

    printf("Terminating\n");

    return (0);
}

// BACK UP
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

    cv::imshow(to_string(i), temp);
}
*/