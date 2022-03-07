#include "image.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Process image boundary
int processBoundary(int total, int x) {
    if (x < 0) {
        return -x - 1;
    }
    if (x >= total) {
        return 2 * total - x - 1;
    }
    return x;
}

// Apply a 5x5 Gaussian filter
int image::blur5x5(cv::Mat &src, cv::Mat &dst) {
    Mat temp;
    float r, c;

    float co[] = {1, 2, 4, 2, 1};

    temp = src.clone();
    dst = src.clone();

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float sum[] = {0.0, 0.0, 0.0};

            for (int i = -2; i <= 2; i++) {
                r = processBoundary(src.rows, y - i);

                cv::Vec3b p = src.at<cv::Vec3b>(r, x);

                sum[0] += p[0] * co[i + 2];
                sum[1] += p[1] * co[i + 2];
                sum[2] += p[2] * co[i + 2];
            }

            for (int i = 0; i < 3; i++) {
                sum[i] /= 10;
            }

            cv::Vec3b t = temp.at<cv::Vec3b>(y, x);
            t.val[0] = sum[0];
            t.val[1] = sum[1];
            t.val[2] = sum[2];
            temp.at<cv::Vec3b>(y, x) = t;
        }
    }

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float sum[] = {0.0, 0.0, 0.0};

            for (int i = -2; i <= 2; i++) {
                c = processBoundary(src.cols, x - i);

                cv::Vec3b p = temp.at<cv::Vec3b>(y, c);

                sum[0] += p[0] * co[i + 2];
                sum[1] += p[1] * co[i + 2];
                sum[2] += p[2] * co[i + 2];
            }

            for (int i = 0; i < 3; i++) {
                sum[i] /= 10;
            }

            cv::Vec3b p = dst.at<cv::Vec3b>(y, x);
            p.val[0] = sum[0];
            p.val[1] = sum[1];
            p.val[2] = sum[2];
            dst.at<cv::Vec3b>(y, x) = p;
        }
    }

    return 0;
}

// Generate the thresholded version for an image
Mat thresholdImageCustom(Mat &image) {
    Mat src(image.rows, image.cols, CV_8UC3);
    // image::blur5x5(image, src);
    src = image.clone();

    Mat dst(src.rows, src.cols, CV_8UC1);
    Mat gray(src.rows, src.cols, CV_8UC1);

    cvtColor(src, gray, COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    float thresMean = (float)mean(gray).val[0];

    bool finish = false;
    while (!finish) {
        int frontSum = 0, backSum = 0, frontCnt = 0, backCnt = 0;
        for (int i = 0; i < gray.rows; i++) {
            for (int j = 0; j < gray.cols; j++) {
                // background
                if (gray.at<unsigned char>(i, j) > thresMean) {
                    dst.at<unsigned char>(i, j) = 0;
                    backSum += gray.at<unsigned char>(i, j);
                    backCnt++;
                } else {
                    dst.at<unsigned char>(i, j) = 255;
                    frontSum += gray.at<unsigned char>(i, j);
                    frontCnt++;
                }
            }
        }
        float frontMean = (float)frontSum / (float)frontCnt;
        float backMean = (float)backSum / (float)backCnt;
        float newThresMean = (frontMean + backMean) / 2.0f;

        if (fabs(newThresMean - thresMean) < 1) {
            finish = true;
        } else {
            thresMean = newThresMean;
        }
    }

    return dst;
}

// Generate the thresholded version for an image
Mat thresholdImageCustom2(Mat &image) {
    Mat src(image.rows, image.cols, CV_8UC3);
    // image::blur5x5(image, src);
    src = image.clone();

    Mat dst(src.rows, src.cols, CV_8UC1);

    int threshold = 200;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // background
            cv::Vec3b p = src.at<cv::Vec3b>(i, j);
            if (p[0] > threshold) {
                dst.at<unsigned char>(i, j) = 255;
            } else {
                dst.at<unsigned char>(i, j) = 0;
            }
        }
    }

    return dst;
}

// Utilize OpenCV to get thresholded image
Mat thresholdImage2(Mat image) {
    cv::Mat hsv, saturation, intensity, dst;
    cv::cvtColor(image, hsv, COLOR_BGR2HSV);

    Mat src(hsv.rows, hsv.cols, CV_8UC3);
    image::blur5x5(hsv, src);

    vector<Mat> splited;
    cv::split(src, splited);

    cv::threshold(splited[1], intensity, 100, 255, THRESH_BINARY);
    cv::threshold(splited[2], saturation, 120, 255, THRESH_BINARY);

    cv::bitwise_or(saturation, intensity, dst);

    return dst;
}

// Generate the thresholded version of an image
Mat image::thresholdImage(Mat &image) {
    Mat thresholdedImg = thresholdImageCustom(image);
    Mat cleanUpImg = cleanUpBinary(thresholdedImg);
    return cleanUpImg;
}

// Generate the thresholded version for a list of images
vector<pair<Mat, Mat>> image::thresholdImages(vector<Mat> &images) {
    vector<pair<Mat, Mat>> thresholdedImgs;
    for (int i = 0; i < images.size(); i++) {
        Mat thresholdedImg = thresholdImage(images[i]);
        thresholdedImgs.push_back(make_pair(images[i], thresholdedImg));
    }

    return thresholdedImgs;
}

// Clean up the Binary image by closing. Closing is reverse of Opening, Dilation followed by Erosion.
// It is useful in closing small holes inside the foreground objects, or small black points on the object.
cv::Mat image::cleanUpBinary(cv::Mat &src) {
    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    dst = src.clone();

    Mat closingElement = getStructuringElement(MORPH_RECT, Size(4, 4), Point(0, 0));

    // 2 iterations
    cv::morphologyEx(src, dst, MORPH_CLOSE, closingElement, Point(-1, -1), 2);
    return dst;
}

// Run connected compoenents analysis for an image
pair<Mat, int> image::connectedComponents(Mat &image) {
    // run connected compoenents analysis
    Mat src = image::thresholdImage(image);
    Mat labelImage(src.size(), CV_32S);  // int
    // 8 way connectivity
    int nLabels = cv::connectedComponents(src, labelImage, 8);

    std::vector<Vec3b> colors(nLabels);
    // 0 represents the background label
    colors[0] = Vec3b(0, 0, 0);
    for (int label = 1; label < nLabels; ++label) {
        colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
    }

    Mat dst(src.size(), CV_8UC3);
    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
        }
    }

    return make_pair(dst, nLabels);
}

// Run connected compoenents analysis on the thresholded and cleaned image to get regions.
// Utilize OpenCV connected components function to get regions and their labels.
vector<pair<Mat, Mat>> image::connectedComponentsImages(vector<Mat> &images) {
    vector<pair<Mat, Mat>> res;
    for (int i = 0; i < images.size(); i++) {
        Mat regions = image::connectedComponents(images[i]).first;
        res.push_back(make_pair(images[i], regions));
    }

    return res;
}

// Calculate a group of image data of an image
ImgData image::calculateImgData(Mat &src) {
    ImgData res;

    res.original = src;
    res.thresholded = image::thresholdImage(src);

    pair<Mat, int> cc = image::connectedComponents(src);
    res.regions = cc.first;  // color, type as CV_32S
    res.numRegions = cc.second;

    // contours - https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    Mat thres = res.thresholded > 0;
    // findCountours only support type as CV_8U1
    // RetrievalModes - retrieves only the extreme outer contours
    cv::findContours(thres, res.contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // find the largest contour
    int maxIdx = 0;
    for (int i = 0; i < res.contours.size(); i++) {
        if (res.contours[i].size() >= res.contours[maxIdx].size())
            maxIdx = i;
    }
    // get largest shape's bounding box
    res.bbox = cv::minAreaRect(res.contours[maxIdx]);

    // calculate features
    res.features = image::calculateFeature(res.regions, res.contours, maxIdx, res.bbox, res.axisEndPoints);

    return res;
}

// Calculate features of an image
Feature image::calculateFeature(Mat &regions, vector<vector<Point>> &contours, int maxIdx, RotatedRect &bbox, vector<Point> &axisEndPoints) {
    Feature features;

    // fill ratio
    double regionArea = cv::contourArea(contours[maxIdx]);
    double boundingBoxArea = bbox.size.width * bbox.size.height;
    double fillRatio = regionArea / boundingBoxArea;
    features.fillRatio = fillRatio;

    // boundingbox dimension ratio
    double bboxDimRatio = bbox.size.width / bbox.size.height;
    if (bboxDimRatio > 1)
        bboxDimRatio = 1.0 / bboxDimRatio;
    features.bboxDimRatio = bboxDimRatio;

    // axises dimension ratio
    // https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gaf259efaad93098103d6c27b9e4900ffa
    RotatedRect rect = cv::fitEllipse(contours[maxIdx]);
    double axisDimRatio = rect.size.width / rect.size.height;
    if (axisDimRatio > 1)
        axisDimRatio = 1.0 / axisDimRatio;
    features.axisDimRatio = axisDimRatio;

    // axises end points, using the RotatedRect in which the fit ellipse is inscribed
    Point2f endPoints[4];
    rect.points(endPoints);
    for (int i = 0; i < 4; i++) {
        axisEndPoints.push_back(Point((endPoints[i] + endPoints[(i + 1) % 4]) / 2.0));
    }

    // hu moments
    // https://docs.opencv.org/3.4/d0/d49/tutorial_moments.html
    Moments moments = cv::moments(contours[maxIdx], true);  // bool binaryImage as true
    vector<double> huMoments;
    cv::HuMoments(moments, huMoments);  // current as 7
    features.huMoments = huMoments;

    return features;
}