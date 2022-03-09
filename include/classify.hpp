#ifndef classify_hpp
#define classify_hpp

#include <opencv2/core/mat.hpp>
#include <vector>

#include "image.hpp"

namespace classify {

Feature calculateFeatureStdDev(vector<ImgData> &traingImgData);
double calculateStdDev(vector<double> &data);
string classifyObject(Feature &src, map<string, vector<Feature>> &db, Feature &stdDevFeature);
double euclideanDist(Feature &src, Feature &cmp, Feature &stdDevFeature);

}  // namespace classify

#endif /* classify_hpp */