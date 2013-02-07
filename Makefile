CXXFLAGS=-I$(OPENCV)/include
LDFLAGS=-L$(OPENCV)/lib -lopencv_highgui -lopencv_imgproc -lopencv_core

testopencv: testopencv.o
	g++ -o $@ $^ $(LDFLAGS)
