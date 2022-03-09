#ifndef image_hpp
#define image_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

using namespace cv;
using namespace std;

// define Feature
struct Feature {
    double fillRatio;
    double bboxDimRatio;
    double axisDimRatio;
    vector<double> huMoments;
};

// define image data
struct ImgData {
    Mat original;
    Mat thresholded;
    Mat regions;
    int numRegions;
    RotatedRect bbox;
    Feature features;
    vector<vector<Point>> contours;
    vector<Point> axisEndPoints;
    string label;
};

namespace image {

// threshold & clean up
int blur5x5(cv::Mat &src, cv::Mat &dst);
Mat thresholdImage(Mat &image);
vector<pair<Mat, Mat>> thresholdImages(vector<Mat> &images);
cv::Mat cleanUpBinary(cv::Mat &image);

// regions
pair<Mat, int> connectedComponents(Mat &src);
vector<pair<Mat, Mat>> connectedComponentsImages(vector<Mat> &images);

// image data & features
ImgData calculateImgData(Mat &src);
Feature calculateFeatures(Mat &regions, vector<vector<Point>> &contours, int maxIdx, RotatedRect &bbox, vector<Point> &axes);

}  // namespace image

#endif /* image_hpp */
