#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
#include <iostream>
#include <getopt.h>
#include <assert.h>
using namespace cv;
using namespace std;

H5::H5File create_feature_file(char *filename, const Mat &image);
void write_feature(H5::H5File h5file, const Mat &image, const char *name);

void find_membranes(Mat &image_in, int windowsize, int membranewidth, H5::H5File &h5f);

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
  
  /* create the feature file */
  H5::H5File h5f = create_feature_file(output_hdf5, image);

  /* compute and write features */
  
  /* FEATURE: Original image */
  write_feature(h5f, image, "original");
  
  /* normalize image */
  // image = adaptive_histeq(image);
  
  /* FEATURE: normalized cross-correlation with membrane template */
  find_membranes(image, windowsize, membranewidth, h5f);
}



