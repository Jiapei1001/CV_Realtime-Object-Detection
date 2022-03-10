#include "classify.hpp"

#include <functional>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

#include "image.hpp"

using namespace cv;
using namespace std;

// Calculate each feature's standard diviations of the training image data
Feature classify::calculateFeatureStdDev(vector<ImgData> &traingImgData) {
    Feature stdDev;

    vector<double> fillRatios;
    vector<double> bboxDimRatios;
    vector<double> axisDimRatios;
    // https://stackoverflow.com/questions/21663256/how-to-initialize-a-vector-of-vectors-on-a-struct
    // only assign dimension of 6 buckets
    vector<vector<double>> huMomentsList(6);

    for (int i = 0; i < traingImgData.size(); i++) {
        ImgData curr = traingImgData[i];
        fillRatios.push_back(curr.features.fillRatio);
        bboxDimRatios.push_back(curr.features.bboxDimRatio);
        axisDimRatios.push_back(curr.features.axisDimRatio);

        // NOTE: here is to assign the corresponding huMoment to its associated bucket
        for (int j = 0; j < curr.features.huMoments.size(); j++) {
            huMomentsList[j].push_back(curr.features.huMoments[j]);
        }
    }

    stdDev.fillRatio = classify::calculateStdDev(fillRatios);
    stdDev.bboxDimRatio = classify::calculateStdDev(bboxDimRatios);
    stdDev.axisDimRatio = classify::calculateStdDev(axisDimRatios);

    vector<double> stdDevHuMoment;
    for (int i = 0; i < huMomentsList.size(); i++) {
        double stdDevAtBucket = classify::calculateStdDev(huMomentsList[i]);
        stdDevHuMoment.push_back(stdDevAtBucket);
    }
    stdDev.huMoments = stdDevHuMoment;

    return stdDev;
}

// Calculate the standard diviation given a list of numbers
// https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/variance-standard-deviation-population/a/calculating-standard-deviation-step-by-step
double classify::calculateStdDev(vector<double> &data) {
    // Step 1 : Find the mean.
    // Step 2 : For each data point, find the square of its distance to the mean.
    // Step 3 : Sum the values from Step 2.
    // Step 4 : Divide by the number of data points.
    // Step 5: Take the square root.
    int n = data.size();
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / n;

    double squareDistSum = 0.0;
    for (double d : data) {
        squareDistSum += (d - mean) * (d - mean);
    }

    return sqrt(squareDistSum / n);
}

// Compare with image's feature in db and standard diviated feature, to find the closest feature's label
string classify::classifyObject(Feature &src, map<string, vector<Feature>> &db, Feature &stdDevFeature) {
    string res = "unknown";

    double minDist = 1000;

    // https://stackoverflow.com/questions/26281979/c-loop-through-map
    // map[first, second] -> [label, list of features]
    for (auto const &img : db) {
        double dist = 0.0;
        for (Feature cmpFeature : img.second) {
            dist += classify::euclideanDist(src, cmpFeature, stdDevFeature);
        }
        dist /= (double)img.second.size();

        if (dist < minDist) {
            res = img.first;
            minDist = dist;
        }
    }

    cout << "dist:\t\t  " << minDist << endl;

    return res;
}

// Calculate Euclidean Distance
double classify::euclideanDist(Feature &src, Feature &cmp, Feature &stdDevFeature) {
    double dist = 0.0;

    dist += fabs(src.fillRatio - cmp.fillRatio) / stdDevFeature.fillRatio;
    dist += fabs(src.bboxDimRatio - cmp.bboxDimRatio) / stdDevFeature.bboxDimRatio;
    dist += fabs(src.axisDimRatio - cmp.axisDimRatio) / stdDevFeature.axisDimRatio;

    // normalize 6 buckets of hu moments
    double sumSrc = 0.0, sumCmp = 0.0, sumStdDev = 0.0;
    for (int i = 0; i < 6; i++) {
        sumSrc += src.huMoments[i] * src.huMoments[i];
        sumCmp += cmp.huMoments[i] * cmp.huMoments[i];
        sumStdDev += stdDevFeature.huMoments[i] * stdDevFeature.huMoments[i];
    }
    dist += fabs(sqrt(sumSrc) - sqrt(sumCmp)) / (sqrt(sumStdDev));

    return dist;
}
