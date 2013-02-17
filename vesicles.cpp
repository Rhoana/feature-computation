#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>

#include "quickmedian.h"

using namespace cv;
using namespace std;

void write_feature(H5::H5File h5file, const Mat &image, const char *name);

#define MIN_RADIUS 2
#define MAX_RADIUS 6

string vesicle_feature_name(int radius)
{
    ostringstream s;
    s << "vesicle_" << radius;
    return string(s.str());
}

void vesicles(const Mat &image_in, H5::H5File &h5f)
{
    Mat maxv = Mat::zeros(image_in.size(), CV_32F);
    for (int r = MIN_RADIUS; r <= MAX_RADIUS; r++) {
        int windowsize = 2 * r + 3;
        Mat tmplate = Mat::zeros(windowsize, windowsize, CV_8U);
        circle(tmplate, Point(r + 1, r + 1), r, Scalar(255));

        // OpenCV's template matching returns only the "valid" region (if compared to normxcorr2 in matlab).  We pre-pad
        // the image to result in "same" behavior.
        Mat image;
        copyMakeBorder(image_in, image,
                       windowsize / 2, windowsize / 2,
                       windowsize / 2, windowsize / 2,
                       BORDER_REFLECT_101);

        // allocate space for all matches, to allow min/max/mean/variance/median computation per-pixel
        Mat match;
        matchTemplate(image, tmplate, match, CV_TM_CCORR_NORMED);
        write_feature(h5f, match, vesicle_feature_name(r).c_str());
        maxv = max(maxv, match);
    }
    write_feature(h5f, maxv, "vesicle_max");
}

