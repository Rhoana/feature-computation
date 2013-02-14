#include <opencv/cv.h>
#include <H5Cpp.h>
#include <assert.h>
using namespace cv;
using namespace H5;
using namespace std;


static Size imsize;
H5File create_feature_file(char *filename, const Mat &base_image)
{
    imsize = base_image.size();
    return H5File(filename, H5F_ACC_TRUNC);
}
    

static DataSet create_dataset(H5File h5f, const char *name)
{
    DSetCreatPropList cparms;
    hsize_t chunk_dims[2] = {256, 256};
    hsize_t dims[2];
    cparms.setChunk(2, chunk_dims);
    cparms.setShuffle();
    cparms.setDeflate(5);
    dims[0] = imsize.height;
    dims[1] = imsize.width;
  
    return h5f.createDataSet(name, PredType::NATIVE_FLOAT,
                             DataSpace(2, dims, dims),
                             cparms);
}

void write_feature(H5File h5f, const Mat &image_in, const char *name)
{
    // make sure the sizes match
    assert (imsize == image_in.size());

    // make sure the image is in native float
    Mat image;
    if (image_in.type() !=  CV_32F)
        image_in.convertTo(image, CV_32F);
    else
        image = image_in;
    
    DataSet dataset = create_dataset(h5f, name);

    DataSpace imspace;
    float *imdata;
    if (image.isContinuous()) {
        hsize_t count[2];
        imspace = dataset.getSpace(); // same size as 
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
        hsize_t im_offset[2], im_size[2];
        im_offset[0] = parent_ofs.y; im_offset[1] = parent_ofs.x;
        im_size[0] = image.size().height; im_size[1] = image.size().width;
        imspace.selectHyperslab(H5S_SELECT_SET, im_size, im_offset);
        imdata = image.ptr<float>() - parent_ofs.x - parent_ofs.y * parent_size.width;
    }
    dataset.write(imdata, PredType::NATIVE_FLOAT, imspace);
}
