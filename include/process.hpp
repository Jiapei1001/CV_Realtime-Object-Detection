#ifndef process_hpp
#define process_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

#include "image.hpp"

using namespace std;
using namespace image;

namespace process {
void loadImages(vector<cv::Mat> &images, const char *dirname);
void loadTrainingImages(vector<cv::Mat> &images, const char *dirname, vector<std::string> &labels);
void displayResults(vector<cv::Mat> &images);
void displayResultsInOneWindow(vector<cv::Mat> &images);
void printModeDescriptions();

// A3
void displayResultsWithFeatures(string displayName, ImgData &imgData);

}  // namespace process

#endif /* process_hpp */