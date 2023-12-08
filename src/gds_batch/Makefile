################################################################################
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Linux Platforms)
#
################################################################################

# Common includes and paths for CUDA
#CUDA_PATH   := /usr/local/cuda
CUDA_PATH   := /home/grads/s/sls7161/anaconda3/envs/matchmaker/

#CUFILE_PATH ?= /usr/local/cuda/targets/x86_64-linux/lib/
CUFILE_PATH ?= /home/grads/s/sls7161/anaconda3/lib/
#CUFILE_PATH ?= /home/grads/s/sls7161/anaconda3/envs/matchmaker/lib/

#CUFILE_INCLUDE_PATH ?= /usr/local/cuda/targets/x86_64-linux/include/
CUFILE_INCLUDE_PATH ?= /home/grads/s/sls7161/anaconda3/envs/matchmaker/include/

#CXXFLAGS    := -Wall
CXXFLAGS    += -I $(CUDA_PATH)/include/ 
CXXFLAGS    += -I $(CUFILE_INCLUDE_PATH)
###########################
# Enable the following line for code coverage
ifneq ($(CONFIG_CODE_COVERAGE),)
CXXFLAGS    += -ftest-coverage -fprofile-arcs
endif
CXXFLAGS += -std=c++17
###########################
CUDART_STATIC := -Bstatic -L $(CUDA_PATH)/lib64/ -lcudart_static -lrt -lpthread -ldl
LDFLAGS     :=  $(CUFILE_LIB) $(CUDART_STATIC) -lcrypto -lssl
CUFILE_LIB  := -L $(CUFILE_PATH) -lcufile

CUFILE_LIB_STATIC  := -L $(CUFILE_PATH) -lcufile_static
#CUFILE_LIB := -Bstatic -L $(CUFILE_PATH) -lcufile_static
LDFLAGS     :=  $(CUFILE_LIB) -L $(CUDA_PATH)/lib64/stubs -lcuda $(CUDART_STATIC) -Bdynamic -lrt -ldl
LDFLAGS_STATIC     :=  $(CUFILE_LIB_STATIC) -L $(CUDA_PATH)/lib64/stubs -lcuda $(CUDART_STATIC) -Bdynamic -lrt -ldl
INSTALL_GDSSAMPLES_PREFIX = /usr/local/gds/samples
NVCC          :=$(CUDA_PATH)//bin/nvcc

LDFLAGS += -L/home/grads/s/sls7161/Documents/cpp_installs/vcpkg/packages/jsoncpp_x64-linux/lib/
LDLIBS = -ljsoncpp
################################################################################
CC:=g++
DOCKER_BUILD_REPO=gds_build:manylinux2014-mofed-5-1-cuda-11-1
DOCKERFILE_BUILD_PATH=../../docker/manylinux2014-docker/Dockerfile-manylinux2014-cuda-11-1-MOFED-5-1
# Target rules
all: build
release_samples:
	docker build -t $(DOCKER_BUILD_REPO) -f $(DOCKERFILE_BUILD_PATH) ./
	docker run -v `pwd`/../../:/nvme-direct:rw --rm -it $(DOCKER_BUILD_REPO) sh -c "cd /nvme-direct/tools/samples;$(MAKE) build"
	@echo "Release samples Built"

build: GDS



#%: %.cc $(CUFILE_INCLUDE_PATH)/cufile.h
#	$(CC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
#%_static: %.cc $(CUFILE_INCLUDE_PATH)/cufile.h
#	$(CC) $(CXXFLAGS) $< -o $@ $(LDFLAGS_STATIC)

GDS: GPUDirectbatch.o utils.o #GPUDirect_IO.o #$(CUFILE_INCLUDE_PATH)/cufile.h
	$(CC) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@ 

GPUDirectbatch.o: GPUDirectbatch.cu $(CUFILE_INCLUDE_PATH)/cufile.h 
	$(NVCC) $(CXXFLAGS) -c GPUDirectbatch.cu -o GPUDirectbatch.o

utils.o: utils.cpp utils.h
	$(CC) $(CXXFLAGS) -c utils.cpp -o utils.o

GPUDirect_IO.o: GPUDirect_IO.cc GPUDirect_IO.h $(CUFILE_INCLUDE_PATH)/cufile.h 
	$(CC) $(CXXFLAGS) -c GPUDirect_IO.cc -o GPUDirect_IO.o

install:
	cp Makefile README $(INSTALL_GDSSAMPLES_PREFIX)

clean:
	find . -type f -executable -delete
	rm -f *.o

.PHONY : build install clean
