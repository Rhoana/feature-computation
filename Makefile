OPENCV=/Users/thouis/homebrew//Cellar/opencv/2.4.2
HDF5_DIR=/Users/thouis/homebrew/Cellar/hdf5/1
CXXFLAGS=-I$(OPENCV)/include -I$(HDF5_DIR)/include -g
LDFLAGS=-L$(OPENCV)/lib -lopencv_highgui -lopencv_imgproc -lopencv_core -L$(HDF5_DIR)/lib -lhdf5_cpp -lhdf5

all: compute_features test_image_to_hdf5

membrane.o: quickmedian.h

compute_features: opencv_hdf5.o compute_features.o membrane.o adapthisteq.o local_statistics.o tensor_gradient_features.o drawhist.o 
	g++ -o $@ $^ $(LDFLAGS)

test_image_to_hdf5: opencv_hdf5.o test_image_to_hdf5.o 
	g++ -o $@ $^ $(LDFLAGS)

