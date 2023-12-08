#ifndef UTILS_H
#define UTIILS_H

#include <filesystem>
#include <vector>
#include <string>

struct _embeddings_offset{
    int doc_id;
    unsigned long offset_size; //offset = offset_scale*4096
    int io_size; //1: 4096, 2:8192
};

bool numericSortComparator(const std::filesystem::directory_entry& entry1, const std::filesystem::directory_entry& entry2);

std::vector<std::string> getSortedNpyFiles(const std::string& directoryPath);

std::vector<int> readIntegerArrayFromJson(const std::string& filePath);

std::vector<std::string> getSortedJSONFiles(const std::string& directoryPath);

void saveStructArrayToJson(_embeddings_offset* array, int size, const std::string& filename);


double _print_throughput(clock_t start, clock_t end, clock_t reset_end, int batch_size, size_t size, int NUM_BATCH);

void generateOffsetArray(std::vector<int>& offsetArray, int arraySize, int blockSize);

#endif