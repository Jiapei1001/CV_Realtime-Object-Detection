#include "cascade.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"

using namespace cascade;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

// process the video stream
// reference OpenCV: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
int cascade::cascadeVideoStream() {
    string face = samples::findFile("../data/haarcascades/haarcascade_frontalface_alt.xml");
    string eyes = samples::findFile("../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

    if (!face_cascade.load(face)) {
        cout << "face cascade cannot be loaded\n";
        return -1;
    }
    if (!eyes_cascade.load(eyes)) {
        cout << "eyes cascade cannot be loaded\n";
        return -1;
    }

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

    cv::Mat frame;
    for (;;) {
        *capdev >> frame;  // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        cascade::detectAndDisplay(frame);

        if (cv::waitKey(10) == 'q') {
            break;
        }
    }

    delete capdev;

    return 0;
}

// apply Haar Cascade detection to the identify face and eyes in the frame
// reference OpenCV: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
void cascade::detectAndDisplay(Mat &frame) {
    Mat gray;
    cv::cvtColor(frame, gray, COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // faces
    vector<Rect> faces;
    face_cascade.detectMultiScale(gray, faces);

    // draw
    for (size_t i = 0; i < faces.size(); i++) {
        // find the center of face
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        // draw an ellipse on top of the face frame
        cv::ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 153, 51), 2);

        Mat faceArea = gray(faces[i]);

        // eyes
        vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceArea, eyes);

        // draw
        for (size_t j = 0; j < eyes.size(); j++) {
            // find the center of eyes
            Point center_of_eyes(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            // radius
            int r = cvRound((eyes[j].width + eyes[j].height) / 4.0);
            // draw circle
            cv::circle(frame, center_of_eyes, r, Scalar(0, 255, 255), 2);
        }
    }

    cv::namedWindow("Video", 1);  // identifies a window
    cv::imshow("Video", frame);
}