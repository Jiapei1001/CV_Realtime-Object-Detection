/*
  Identify image files in a directory
*/
#include <dirent.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <vector>

#include "image.hpp"
#include "process.hpp"

using namespace cv;
using namespace std;
using namespace process;
using namespace image;

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

    vector<cv::Mat> trainingImgs;
    vector<std::string> labels;
    char trainingDir[256];
    strcpy(trainingDir, "../data/training");
    process::loadTrainingImages(trainingImgs, trainingDir, labels);

    for (int i = 0; i < labels.size(); i++) {
        cout << i << ":  " << labels[i] << "\n";
    }

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
    vector<Mat> results;
    for (int i = 0; i < images.size(); i++) {
        ImgData imgData = image::calculateImgData(images[i]);
        Mat img = imgData.regions;

        // draw countours
        Scalar color1 = Scalar(150, 50, 255);
        cv::drawContours(img, imgData.contours, 0, color1, 4);

        // draw bounding box
        Scalar color2 = Scalar(200, 255, 100);
        Point2f rect_points[4];
        imgData.bbox.points(rect_points);
        for (int j = 0; j < 4; j++) {
            cv::line(img, rect_points[j], rect_points[(j + 1) % 4], color2, 4);
        }

        // draw axes
        Scalar color3 = Scalar(255, 200, 100);
        line(img, imgData.axisEndPoints[0], imgData.axisEndPoints[2], color3, 2);
        line(img, imgData.axisEndPoints[1], imgData.axisEndPoints[3], color3, 2);

        results.push_back(img);
    }

    process::displayResults(results);

    // NOTE: must add waitKey, or the program will terminate, without showing the result images
    waitKey(0);
    printf("Terminating\n");

    return (0);
}