#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    Mat image;
    image = imread( argv[1], 1 );
    image.convertTo(image, CV_8U);
    cvtColor(image, image, CV_RGB2GRAY);

    int sz = atoi(argv[2]);
    int width = atoi(argv[3]);
    Mat tmplate = Mat::zeros(sz, sz, CV_8U);
    
    for (int i = (sz - width) / 2, ii = 0; ii < width; i++, ii++)
        for (int j = 0; j < sz; j++) 
            tmplate.at<uchar>(i, j) = 255;

    Mat match;
    matchTemplate(image, tmplate, match, CV_TM_CCORR_NORMED);
    normalize(match, match, 0, 1, CV_MINMAX);
    imshow( "Horizontal", match );

    matchTemplate(image, tmplate.t(), match, CV_TM_CCORR_NORMED);
    normalize(match, match, 1, 0, CV_MINMAX);
    imshow( "Vertical", match );
    waitKey(0);
    Mat rot_tmplate;
    Mat rot_mat(2, 3, CV_32FC1);
    for (int step = 1; step < 10; step++) {
        double angle = step * 90.0 / 10;
        Point center = Point(sz/2, sz/2);
        rot_mat = getRotationMatrix2D( center, angle, 1.0);
        warpAffine(tmplate, rot_tmplate, rot_mat, tmplate.size());
        matchTemplate(image, rot_tmplate, match, CV_TM_CCORR_NORMED);
        normalize(match, match, 1, 0, CV_MINMAX);
        imshow( "Horizontal", match );
        
        warpAffine(tmplate.t(), rot_tmplate, rot_mat, tmplate.size());
        matchTemplate(image, rot_tmplate, match, CV_TM_CCORR_NORMED);
        normalize(match, match, 1, 0, CV_MINMAX);
        imshow( "Vertical", match );
        waitKey(0);
    }


    waitKey(0);
    return 0;
}
