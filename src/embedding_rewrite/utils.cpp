#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "utils.h"
#include "json/json.h"

bool numericSortComparator(const std::filesystem::directory_entry& entry1, const std::filesystem::directory_entry& entry2) {
    std::string filename1 = entry1.path().filename().stem().string();
    std::string filename2 = entry2.path().filename().stem().string();
    return std::stoi(filename1) < std::stoi(filename2);
}

std::vector<std::string> getSortedNpyFiles(const std::string& directoryPath) {
    std::vector<std::string> sortedFiles;

    if (std::filesystem::exists(directoryPath) && std::filesystem::is_directory(directoryPath)) {
        std::vector<std::filesystem::directory_entry> npyFiles;
        for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
            if (entry.path().extension() == ".npy") {
                npyFiles.push_back(entry);
            }
        }

        std::sort(npyFiles.begin(), npyFiles.end(), numericSortComparator);

        for (const auto& entry : npyFiles) {
            sortedFiles.push_back(entry.path().filename().string());
        }
    } else {
        std::cout << "Directory does not exist or is not a valid directory." << std::endl;
    }

    return sortedFiles;
}

std::vector<std::string> getSortedJSONFiles(const std::string& directoryPath) {
    std::vector<std::string> sortedFiles;

    if (std::filesystem::exists(directoryPath) && std::filesystem::is_directory(directoryPath)) {
        std::vector<std::filesystem::directory_entry> JSONFiles;
        for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
            if (entry.path().extension() == ".json") {
                JSONFiles.push_back(entry);
            }
        }

        std::sort(JSONFiles.begin(), JSONFiles.end(), numericSortComparator);

        for (const auto& entry : JSONFiles) {
            sortedFiles.push_back(entry.path().filename().string());
        }
    } else {
        std::cout << "Directory does not exist or is not a valid directory." << std::endl;
    }

    return sortedFiles;
}



std::vector<int> readIntegerArrayFromJson(const std::string& filePath) {
    std::vector<int> integerArray;

    // Open the JSON file
    std::ifstream inputFile(filePath);

    if (inputFile.is_open()) {
        // Read the entire contents of the file into a string
        std::string jsonString((std::istreambuf_iterator<char>(inputFile)),
                                std::istreambuf_iterator<char>());

        // Parse the JSON string
        Json::Value root;
        Json::CharReaderBuilder builder;
        Json::CharReader* reader = builder.newCharReader();
        std::string errors;
        bool parsingSuccessful = reader->parse(jsonString.c_str(), jsonString.c_str() + jsonString.size(), &root, &errors);
        delete reader;

        if (parsingSuccessful) {
            // Access the JSON array
            if (root.isArray()) {
                for (const Json::Value& element : root) {
                    if (element.isInt()) {
                        int value = element.asInt();
                        integerArray.push_back(value);
                    }
                }
            } else {
                std::cout << "Root is not an array." << std::endl;
            }
        } else {
            std::cout << "Failed to parse JSON: " << errors << std::endl;
        }
    } else {
        std::cout << "Failed to open JSON file." << std::endl;
    }

    return integerArray;
}

void saveStructArrayToJson(_embeddings_offset* array, int size, const std::string& filename) {
    // Create a JSON root object
    Json::Value root;

    // Convert each struct in the array to a JSON object and add it to the root
    for (int i = 0; i < size; ++i) {
        Json::Value structObject;
        structObject["doc_id"] = array[i].doc_id;
        structObject["offset_size"] = array[i].offset_size;
        structObject["io_size"] = array[i].io_size;
        root.append(structObject);
    }

    // Write the JSON root object to a file
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        outputFile << root;
        outputFile.close();
        std::cout << "JSON data saved to file: " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}


/*

int main() {
    std::string directoryPath = "/home/grads/s/sls7161/nvme/NeuralIR/ColBERTer/CLS/"; // Replace with the desired directory path

    std::vector<std::string> sortedNpyFiles = getSortedNpyFiles(directoryPath);

    for (const auto& filename : sortedNpyFiles) {
        std::cout << filename << std::endl;
    }
    std::string filePath = "/home/grads/s/sls7161/nvme/NeuralIR/ColBERTer/BOW/0.json";
    std::vector<int> result = readIntegerArrayFromJson(filePath);

    for (int i=0; i<10; i++){
        std::cout<<result[i]<<std::endl;
    }

    return 0;
}


*/
