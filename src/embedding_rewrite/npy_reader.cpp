#include "npy_reader.h"
#include <stdexcept>
#include <cstring>
#include <regex>
#include <fcntl.h>
#include <unistd.h>
#include <cassert>
#include <iostream>

int parse_npy_header_x(int fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order) {
    char buffer[256];
    char* newlinePos = nullptr;

    ssize_t res = pread(fp, buffer, 11, 0);
    if (res != 11)
        throw std::runtime_error("parse_npy_header: failed fread");

    ssize_t bytesRead = pread(fp, buffer, 256, res);
    newlinePos = strchr(buffer, '\n');

    ssize_t bytesToProcess = newlinePos - buffer + 1;
    std::string fullHeader(buffer, bytesToProcess);
    std::string header = fullHeader.substr(0, fullHeader.size() - 1);

    size_t loc1, loc2;

    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    loc1 = header.find("descr");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());

    return res + bytesToProcess;
}

meta read_npy_metadata(std::string filename){
    int file = open(filename.c_str(), O_RDONLY);
    if (file == -1) {
        throw std::runtime_error("parse_npy_header: failed to open file");
    }
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;

    int bytesRead = parse_npy_header_x(file ,word_size,shape,fortran_order);
    //std::cout << "Bytes read from header = " << bytesRead << std::endl;

    // Iterate over the vector and print each element
    /*
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;    
    
    */

    int num_bytes = word_size*shape[0]*shape[1];

    //std::cout<< "word size = " <<  word_size << std::endl;
    //std::cout<< "fortran_order = " <<  fortran_order << std::endl;
    //std::cout<< "Number of bytes to read = " << num_bytes <<std::endl;

    off_t dataStart = bytesRead;

    meta metadata{file, shape, word_size, fortran_order, num_bytes, dataStart};

    return metadata;
}