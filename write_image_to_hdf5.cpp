#include <opencv/cv.h>
#include <H5Cpp.h>
#include <assert.h>
using namespace cv;
using namespace H5;

DataSet create_dataset(Mat &image, char *filename)
{
  /* Open output, create dataset */
  H5File hdf5_f(filename, H5F_ACC_TRUNC);
  DSetCreatPropList cparms;
  hsize_t chunk_dims[3] = {256, 256, 1};
  hsize_t dims[3], maxdims[3];  // dataset dimensions at creation, and maximums
  float fill_val = 0;
  cparms.setChunk(3, chunk_dims);
  cparms.setFillValue(PredType::NATIVE_FLOAT, &fill_val);
  cparms.setShuffle();
  dims[0] = maxdims[0] = image.size().height;
  dims[1] = maxdims[1] = image.size().width;
  dims[2] = 0;
  maxdims[2] = H5S_UNLIMITED;
  
  return hdf5_f.createDataSet("features", PredType::NATIVE_FLOAT,
                              DataSpace(3, dims, maxdims),
                              cparms);

}

void write_feature(Mat &image, DataSet &dataset)
{
  hsize_t dims[3];
  DataSpace dsp = dataset.getSpace();
  dsp.getSimpleExtentDims(dims, NULL);
  assert (image.size().height == dims[0]);
  assert (image.size().width == dims[1]);
  // extend by one image
  dims[2] += 1;
  dataset.extend(dims);
  // make sure the image is in native float
  image.convertTo(image, CV_32F);
  
  // Select the last plane of the dataset for writing
  hsize_t offset[3], count[3];
  offset[0] = offset[1] = 0; offset[2] = dims[2] - 1;
  count[0] = dims[0]; count[1] = dims[1]; count[2] = 1;
  dsp = dataset.getSpace();  // necessary after extent
  dsp.selectHyperslab(H5S_SELECT_SET, count, offset);
  
  DataSpace imspace;
  float *imdata;
  if (image.isContinuous()) {
    imspace.setExtentSimple(2, count); // borrow the sizes from dsp
    imspace.selectAll();
    imdata = image.ptr<float>();
  } else {
    // we are working with an ROI
    assert (image.isSubmatrix());
    Size parent_size; Point parent_ofs;
    image.locateROI(parent_size, parent_ofs);
    hsize_t parent_count[2];
    parent_count[0] = parent_size.height; parent_count[1] = parent_size.width;
    imspace.setExtentSimple(2, parent_count);
    hsize_t im_offset[2];
    im_offset[0] = parent_ofs.y; im_offset[1] = parent_ofs.x;
    imspace.selectHyperslab(H5S_SELECT_SET, dims, im_offset);
    imdata = image.ptr<float>() - parent_ofs.x - parent_ofs.y * parent_size.width;
  }
  dataset.write(imdata, PredType::NATIVE_FLOAT, imspace, dsp);
}
