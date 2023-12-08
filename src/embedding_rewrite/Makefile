CXX = g++
CXXFLAGS = -std=c++17
LDFLAGS = -L/home/grads/s/sls7161/Documents/cpp_installs/vcpkg/packages/jsoncpp_x64-linux/lib/
LDLIBS = -ljsoncpp

EXECUTABLE2 = embeddings 

all: $(EXECUTABLE2)

$(EXECUTABLE2): embeddings_rewrite.o npy_reader.o utils.o 
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

npy_reader.o: npy_reader.cpp npy_reader.h
	$(CXX) $(CXXFLAGS) -c npy_reader.cpp -o npy_reader.o

utils.o: utils.cpp utils.h
	$(CXX) $(CXXFLAGS) -c utils.cpp -o utils.o

embeddings_rewrite.o: embeddings_rewrite.cpp npy_reader.h utils.h
	$(CXX) $(CXXFLAGS) -c embeddings_rewrite.cpp -o embeddings_rewrite.o

clean:
	rm -f $(EXECUTABLE2) *.o