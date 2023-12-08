#ifndef NPYPARSER_H
#define NPYPARSER_H

#include <vector>
#include <string>

struct meta {
    int file;
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    int num_bytes;
    off_t dataStart;
};

int parse_npy_header_x(int fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order);
meta read_npy_metadata(std::string filename);

#endif // NPYPARSER_H
