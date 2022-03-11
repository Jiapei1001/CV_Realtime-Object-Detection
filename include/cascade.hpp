#ifndef cascade_hpp
#define cascade_hpp

#include <iostream>
#include <map>
#include <opencv2/core/mat.hpp>
#include <vector>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;

namespace cascade {

int cascadeVideoStream();
void detectAndDisplay(Mat &frame);

}  // namespace cascade

#endif /* cascade_hpp */