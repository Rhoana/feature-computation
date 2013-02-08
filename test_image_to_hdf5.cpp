#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
using namespace cv;

H5::DataSet create_dataset(Mat &image, char *filename);
void write_feature(Mat &image, H5::DataSet &dataset);

int main(int argc, char** argv) {
  char *input_image = argv[1];
  char *output_hdf5 = argv[2];
  char *suboutput_hdf5 = argv[3];
  
  /* Read input, convert to grayscale */
  Mat image;
  image = imread( input_image, 1 );
  image.convertTo(image, CV_8U);
  cvtColor(image, image, CV_RGB2GRAY);
  
  /* create the dataset */
  H5::DataSet dataset = create_dataset(image, output_hdf5);
  
  /* write the image */
  write_feature(image, dataset);
  
  image = image(Rect(10, 10, image.size().width - 20, image.size().height - 20));
  /* create a dataset for a subimage */
  dataset = create_dataset(image, suboutput_hdf5);
  write_feature(image, dataset);  
}



