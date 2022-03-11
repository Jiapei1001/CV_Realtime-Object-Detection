#include "process.hpp"

#include <dirent.h>
#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "classify.hpp"
#include "image.hpp"

using namespace cv;
using namespace std;
using namespace image;

// print descriptions
void process::printModeDescriptions() {
    std::cout << "Keys for modes:" << std::endl;
    std::cout << "a)Baseline comparison \t\t\t\t\t -key '0' :" << std::endl;
    std::cout << "b)3D RGB Histogram comparison \t\t\t\t -key '1' :" << std::endl;
    std::cout << "c)Multiple part comparison \t\t\t\t -key '2' :" << std::endl;
    std::cout << "d)Sobel Texture + 3D RGB Histogram comparison  \t\t -key '3' :" << std::endl;
    std::cout << "e)Custom - Centeral Area + Sobel Texture + Hue Saturation-key '4' :" << std::endl;
    std::cout << "f)2D RG Histogram comparison \t\t\t\t -key '5' :" << std::endl;
    std::cout << "g)Sobel Texture + 2D RG Histogram \t\t\t -key '6' :" << std::endl;
    std::cout << "h)Major Shape Contour comparison \t\t\t -key '7' :" << std::endl;
    std::cout << "i)Gradient Orientation Texture + Hue Saturation \t -key '8' :" << std::endl;
}

// load images from a directory
void process::loadImages(vector<Mat> &images, const char *dirname, vector<string> &actualLabels) {
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;

    printf("Processing directory %s\n\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".jpeg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {
            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // image path
            // printf("full path name: %s\n", buffer);

            cv::Mat newImage;
            newImage = cv::imread(buffer);

            // check if new Mat is built
            if (newImage.data == NULL) {
                cout << "This new image" << buffer << "cannot be loaded into cv::Mat\n";
                exit(-1);
            }

            images.push_back(newImage);

            std::string s = dp->d_name;
            std::string delimiter = "_";
            std::string token = s.substr(0, s.find(delimiter));
            actualLabels.push_back(token);
        }
    }

    closedir(dirp);
}

// load training images from a directory; use the image name as the label
void process::loadTrainingImages(vector<Mat> &images, const char *dirname, vector<std::string> &labels) {
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;

    printf("Processing training images in the directory %s\n\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".jpeg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {
            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            cv::Mat newImage;
            newImage = cv::imread(buffer);

            // check if new Mat is built
            if (newImage.data == NULL) {
                cout << "This new image" << buffer << "cannot be loaded into cv::Mat\n";
                exit(-1);
            }

            images.push_back(newImage);

            // https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
            std::string s = dp->d_name;
            std::string delimiter = "_";
            std::string token = s.substr(0, s.find(delimiter));
            labels.push_back(token);
        }
    }

    closedir(dirp);
}

// display results in separate windows
void process::displayResults(vector<cv::Mat> &results) {
    float targetWidth = 600;
    float scale, targetHeight;

    for (int i = 0; i < results.size(); i++) {
        scale = targetWidth / results[i].cols;
        targetHeight = results[i].rows * scale;
        cv::resize(results[i], results[i], Size(targetWidth, targetHeight));

        string name = "top " + to_string(i);
        namedWindow(name, WINDOW_AUTOSIZE);
        cv::imshow(name, results[i]);
    }
}

// display results in one window
void process::displayResultsInOneWindow(vector<cv::Mat> &results) {
    int numR, numC;

    int resSq = (int)sqrt(results.size());

    // get one picture's dimension
    int singleW = results[0].cols;
    int singleH = results[0].rows;

    numC = resSq;
    // account for extra row(s) for remaining image(s), and round up
    numR = ceil((float)results.size() / (float)numC);

    cv::Mat dstMat(Size(numC * singleW, numR * singleH), CV_8UC3, Scalar(120, 120, 120));

    // assign result to mat
    int currIdx = 0;
    for (int i = 0; i < numR; i++) {
        for (int j = 0; j < numC; j++) {
            if (currIdx == results.size()) {
                break;
            }
            results[currIdx].copyTo(dstMat(Rect(j * singleW, i * singleH, singleW, singleH)));
            currIdx++;
        }
    }

    // scale
    float targetWidth = 1200;
    float scale, targetHeight;
    scale = targetWidth / dstMat.cols;
    targetHeight = dstMat.rows * scale;

    cv::resize(dstMat, dstMat, Size(targetWidth, targetHeight));

    string window_name = "top matched results";
    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow(window_name, dstMat);
}

// Display the features besides the original image
void process::displayResultsWithFeaturesAsImage(string displayName, ImgData &imgData) {
    Mat temp = imgData.thresholded;
    // make 1D channel to 3D
    // https://stackoverflow.com/questions/9970660/convert-1-channel-image-to-3-channel
    cv::Mat thresholded;
    cv::Mat in[] = {temp, temp, temp};
    cv::merge(in, 3, thresholded);

    // draw countours
    cv::drawContours(thresholded, imgData.contours, 0, Scalar(120, 80, 255), 6);

    // draw bounding box
    Point2f corners[4];
    imgData.bbox.points(corners);
    for (int j = 0; j < 4; j++) {
        // cv::line(thresholded, corners[j], corners[(j + 1) % 4], Scalar(220, 230, 80), 4);
        cv::line(thresholded, corners[j], corners[(j + 1) % 4], Scalar(255, 0, 0), 2);
    }

    // draw axes
    line(thresholded, imgData.axisEndPoints[0], imgData.axisEndPoints[2], Scalar(0, 220, 200), 3);
    line(thresholded, imgData.axisEndPoints[1], imgData.axisEndPoints[3], Scalar(0, 220, 200), 3);

    // draw label
    // thickness as 4, linetype as 16 - antiaxis
    Rect rec = imgData.bbox.boundingRect();
    // put text aside boundingbox
    // https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
    cv::putText(thresholded, imgData.label, Point(rec.x, rec.y - 10), FONT_HERSHEY_COMPLEX, 2, Scalar(150, 150, 150), 4, 16);

    float sw = 1024;
    float scale, sh;
    scale = sw / imgData.original.cols;
    sh = scale * imgData.original.rows;
    cv::resize(imgData.original, imgData.original, Size(sw, sh));
    cv::resize(thresholded, thresholded, Size(sw, sh));

    Mat result(Size(sw * 2, sh), CV_8UC3, Scalar(100, 100, 100));
    imgData.original.copyTo(result(Rect(0, 0, sw, sh)));
    thresholded.copyTo(result(Rect(sw, 0, sw, sh)));

    cv::namedWindow(displayName, WINDOW_AUTOSIZE);
    cv::imshow(displayName, result);
}

// Build a confusion matrix table and save it as a .csv file
void process::buildMatrixTable(vector<string> &actualLabels, vector<string> &detectedLabels) {
    ofstream file;
    file.open("../data/csv/matrix.csv");

    vector<string> labelSet(actualLabels);
    // get unique labels
    // https://stackoverflow.com/questions/26824260/c-unique-values-in-a-vector
    sort(labelSet.begin(), labelSet.end());
    vector<string>::iterator it;
    it = unique(labelSet.begin(), labelSet.end());
    labelSet.resize(distance(labelSet.begin(), it));
    // number of unique actual labels from the testing images
    int n = labelSet.size();
    cout << "label number: " << n << "\n";

    // header of the matrix
    file << "Confusion Matrix";
    // map to each label's column index
    map<string, int> map2Idx;
    for (int i = 0; i < n; i++) {
        file << "," << labelSet[i];
        map2Idx[labelSet[i]] = i;
    }
    file << "\n";

    // initiate matrix
    // https://stackoverflow.com/questions/15520880/initializing-entire-2d-array-with-one-value
    int matrix[n][n];
    memset(matrix, 0, n * n * sizeof(int));

    for (int i = 0; i < n; i++) {
        matrix[0][i] = 0;
    }
    // rows are detected labels, cols are actual labels
    for (int i = 0; i < actualLabels.size(); i++) {
        int c = map2Idx[actualLabels[i]];
        int r = map2Idx[detectedLabels[i]];
        matrix[r][c] += 1;
    }

    // save csv file
    for (int i = 0; i < n; i++) {
        // each label's name
        file << labelSet[i];
        for (int j = 0; j < n; j++) {
            file << "," << to_string(matrix[i][j]);
        }
        file << "\n";
    }

    file.close();
}

// Process object detection by video mode
// Reference Assignment #1
// int process::classifyObjectByVideo(map<string, vector<Feature>> &db, Feature &standardFeature) {
//     cv::VideoCapture *capdev;

//     // open the video device
//     capdev = new cv::VideoCapture(0);
//     if (!capdev->isOpened()) {
//         printf("Unable to open video device\n");
//         return (-1);
//     }

//     // get some properties of the image
//     cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
//                   (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
//     printf("Expected size: %d %d\n", refS.width, refS.height);

//     cv::namedWindow("Video", 1);  // identifies a window

//     cv::Mat frame;
//     for (;;) {
//         *capdev >> frame;  // get a new frame from the camera, treat as a stream
//         if (frame.empty()) {
//             printf("frame is empty\n");
//             break;
//         }

//         ImgData imgData = image::calculateImgData(frame);
//         imgData.label = classify::classifyObjectByKNN(frame.features, db, standardFeature);

//         // draw countours
//         cv::drawContours(frame, imgData.contours, 0, Scalar(120, 80, 255), 6);

//         // draw bounding box
//         Point2f corners[4];
//         imgData.bbox.points(corners);
//         for (int j = 0; j < 4; j++) {
//             // cv::line(thresholded, corners[j], corners[(j + 1) % 4], Scalar(220, 230, 80), 4);
//             cv::line(frame, corners[j], corners[(j + 1) % 4], Scalar(255, 0, 0), 2);
//         }

//         // draw axes
//         line(frame, imgData.axisEndPoints[0], imgData.axisEndPoints[2], Scalar(0, 220, 200), 3);
//         line(frame, imgData.axisEndPoints[1], imgData.axisEndPoints[3], Scalar(0, 220, 200), 3);

//         // draw label
//         // thickness as 4, linetype as 16 - antiaxis
//         Rect rec = imgData.bbox.boundingRect();
//         // put text aside boundingbox
//         // https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
//         cv::putText(frame, imgData.label, Point(rec.x, rec.y - 10), FONT_HERSHEY_COMPLEX, 2, Scalar(150, 150, 150), 4, 16);

//         cv::imshow("Video", frame);
//         if (cv::waitKey(10) == 'q') {
//             break;
//         }
//     }

//     delete capdev;
//     return (0);
// }

// Display features in video frame
int process::displayResultsWithFeaturesInVideoFrame(cv::Mat &frame, ImgData &imgData) {
    // draw countours
    cv::drawContours(frame, imgData.contours, 0, Scalar(120, 80, 255), 6);

    // draw bounding box
    Point2f corners[4];
    imgData.bbox.points(corners);
    for (int j = 0; j < 4; j++) {
        // cv::line(thresholded, corners[j], corners[(j + 1) % 4], Scalar(220, 230, 80), 4);
        cv::line(frame, corners[j], corners[(j + 1) % 4], Scalar(255, 0, 0), 1);
    }

    // draw axes
    line(frame, imgData.axisEndPoints[0], imgData.axisEndPoints[2], Scalar(0, 220, 200), 1);
    line(frame, imgData.axisEndPoints[1], imgData.axisEndPoints[3], Scalar(0, 220, 200), 1);

    // draw label
    // thickness as 4, linetype as 16 - antiaxis
    Rect rec = imgData.bbox.boundingRect();
    // put text aside boundingbox
    // https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
    cv::putText(frame, imgData.label, Point(rec.x, rec.y - 10), FONT_HERSHEY_COMPLEX, 2, Scalar(0, 128, 255), 3, 16);

    return (0);
}
