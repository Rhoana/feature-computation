#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
using namespace cv;
using namespace std;

void write_feature(H5::H5File h5file, const Mat &image, const char *name);

string membrane_feature_name(int windowsize, int membranewidth, int angle)
{
    ostringstream s;
    s << "membrane_" << windowsize << "_" << membranewidth << "_" << angle;
    return string(s.str());
}

void find_membranes(Mat &image_in, int windowsize, int membranewidth, H5::H5File &h5f)
{
  Mat tmplate = Mat::zeros(windowsize, windowsize, CV_8U);
  
  // OpenCV's template matching returns only the "valid" region (if compared to normxcorr2 in matlab).  We pre-pad
  // the image to result in "same" behavior.
  Mat image;
  copyMakeBorder(image_in, image,
                 windowsize / 2, windowsize / 2,
                 windowsize / 2, windowsize / 2,
                 BORDER_REFLECT_101);
  
  for (int i = (windowsize - membranewidth) / 2, ii = 0; ii < membranewidth; i++, ii++)
    for (int j = 0; j < windowsize; j++)
      tmplate.at<uchar>(i, j) = 255;
  
  Mat match;
  matchTemplate(image, tmplate, match, CV_TM_CCORR_NORMED);
  write_feature(h5f, match,
                membrane_feature_name(windowsize, membranewidth, 0).c_str());

  matchTemplate(image, tmplate.t(), match, CV_TM_CCORR_NORMED);
  write_feature(h5f, match, 
                membrane_feature_name(windowsize, membranewidth, 90).c_str());

  Mat rot_tmplate;
  Mat rot_mat(2, 3, CV_32FC1);
  for (int step = 1; step < 10; step++) {
    double angle = step * 90.0 / 10;
    Point center = Point(windowsize/2, windowsize/2);
    rot_mat = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(tmplate, rot_tmplate, rot_mat, tmplate.size());
    matchTemplate(image, rot_tmplate, match, CV_TM_CCORR_NORMED);
    write_feature(h5f, match,
                  membrane_feature_name(windowsize, membranewidth, angle).c_str());

    warpAffine(tmplate.t(), rot_tmplate, rot_mat, tmplate.size());
    matchTemplate(image, rot_tmplate, match, CV_TM_CCORR_NORMED);
    write_feature(h5f, match,
                  membrane_feature_name(windowsize, membranewidth, 90 + angle).c_str());
  }
}

