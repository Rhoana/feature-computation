OPENCV=/Users/thouis/homebrew//Cellar/opencv/2.4.2
HDF5_DIR=/Users/thouis/homebrew/Cellar/hdf5/1
CXXFLAGS=-I$(OPENCV)/include -I$(HDF5_DIR)/include
LDFLAGS=-L$(OPENCV)/lib -lopencv_highgui -lopencv_imgproc -lopencv_core -L$(HDF5_DIR)/lib -lhdf5_cpp -lhdf5

all: compute_features test_image_to_hdf5

compute_features: write_image_to_hdf5.o compute_features.o 
	g++ -o $@ $^ $(LDFLAGS)

test_image_to_hdf5: write_image_to_hdf5.o test_image_to_hdf5.o 
	g++ -o $@ $^ $(LDFLAGS)

