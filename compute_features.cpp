#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
#include <iostream>
#include <getopt.h>
#include <assert.h>
using namespace cv;
using namespace std;

H5::DataSet create_dataset(Mat &image, char *filename);
void write_feature(Mat &image, H5::DataSet &dataset);

void find_membranes(Mat &image_in, int windowsize, int membranewidth, H5::DataSet &dataset)
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
  write_feature(match, dataset);
  
  matchTemplate(image, tmplate.t(), match, CV_TM_CCORR_NORMED);
  write_feature(match, dataset);

  Mat rot_tmplate;
  Mat rot_mat(2, 3, CV_32FC1);
  for (int step = 1; step < 10; step++) {
    double angle = step * 90.0 / 10;
    Point center = Point(windowsize/2, windowsize/2);
    rot_mat = getRotationMatrix2D( center, angle, 1.0);
    warpAffine(tmplate, rot_tmplate, rot_mat, tmplate.size());
    matchTemplate(image, rot_tmplate, match, CV_TM_CCORR_NORMED);
    write_feature(match, dataset);

    warpAffine(tmplate.t(), rot_tmplate, rot_mat, tmplate.size());
    matchTemplate(image, rot_tmplate, match, CV_TM_CCORR_NORMED);
    write_feature(match, dataset);
  }
}

static int verbose;

/* The options we understand. */
static struct option long_options[] = {
  /* These options set a flag. */
  {"verbose", no_argument, &verbose, 1},
  /* These options don't set a flag.
   We distinguish them by their indices. */
  {"windowsize",  required_argument, 0, 'w'},
  {"membranewidth",  required_argument, 0, 'm'},
  {0, 0, 0, 0}
};

int main(int argc, char** argv) {
  /* Default values. */
  int windowsize = 19;
  int membranewidth = 3;
  
  while (1) {
    int option_index = 0;
    int c = getopt_long (argc, argv, "w:m:", long_options, &option_index);
    
    /* Detect the end of the options. */
    if (c == -1)
      break;
    switch (c) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
          break;
        break;
        
      case 'w':
        windowsize = atoi(optarg);
        assert ((windowsize % 2) == 1);
        break;
        
      case 'm':
        membranewidth = atoi(optarg);
        assert ((membranewidth % 2) == 1);
        break;
        
      case '?':
        /* getopt_long already printed an error message. */
        break;
        
      default:
        abort ();
    }
  }
  
  assert (argc - optind == 2);  /* 2 required arguments */
  char *input_image = argv[optind];
  char *output_hdf5 = argv[optind + 1];
  
  /* Read input, convert to grayscale */
  Mat image;
  image = imread( input_image, 1 );
  image.convertTo(image, CV_8U);
  cvtColor(image, image, CV_RGB2GRAY);
  
  /* create the dataset */
  H5::DataSet dataset = create_dataset(image, output_hdf5);

  /* compute and write features */
  
  /* FEATURE: Original image */
  write_feature(image, dataset);
  
  /* normalize image */
  // image = adaptive_histeq(image);
  
  /* FEATURE: normalized cross-correlation with membrane template */
  find_membranes(image, windowsize, membranewidth, dataset);
}



