#ifndef process_hpp
#define process_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

using namespace std;

namespace process {
void loadImages(vector<cv::Mat> &images, const char *dirname);
void loadTrainingImages(vector<cv::Mat> &images, const char *dirname, vector<std::string> &labels);
void displayResults(vector<cv::Mat> &images);
void displayResultsInOneWindow(vector<cv::Mat> &images);
void printModeDescriptions();

}  // namespace process

#endif /* process_hpp */