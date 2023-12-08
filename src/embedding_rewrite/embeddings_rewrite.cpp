#include <string>
#include "npy_reader.h"
#include "utils.h"

#include <iostream>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <half.hpp>

int main( int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: program <cls_file_path> <bow_file_path> <save_dir>" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::string cls_npy_file = argv[1];
    std::string bow_npy_file = argv[2];
    std::string save_dir = argv[3];

    std::vector<std::string> CLSFiles = getSortedNpyFiles(cls_npy_file);
    std::vector<std::string> BOWFiles = getSortedNpyFiles(bow_npy_file);
    std::vector<std::string> JSONFiles = getSortedJSONFiles(bow_npy_file);

    //Open a file to write
    std::string embedding_file = save_dir + "Msmarco_v2.bin";
    const std::string embedding_metadata_file = save_dir + "Msmarco_v2.json";
    int embd_fd = open(embedding_file.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);

    //Declare varaibles
    half_float::half* buffer_4k = new half_float::half[2048]; //allocate 4KB memory
    half_float::half* buffer_8k = new half_float::half[4096]; //allocate 8KB memory
    half_float::half* buffer;

    ssize_t nread, nwrite;
    int buffer_size;
    int bow_bytes_to_read;
    int total_bytes_to_read;
    int cls_bytes_to_read;

    int result;
    int doc_id = 0;

    //calculate total number of documents
    int total_docs = 0 ;
    int num_files = CLSFiles.size();

    std::cout << "number of files = " << num_files << std::endl;
    for(int j = 0; j<num_files; j++){
        std::string json_file = bow_npy_file + JSONFiles[j];
        std::vector<int> bow_count = readIntegerArrayFromJson(json_file);
        total_docs += bow_count.size();
    }
    
    std::cout << "total docs = " << total_docs << std::endl;
    
    //define embedding_offset array
    _embeddings_offset* embd_metadata = new _embeddings_offset[total_docs]; // bow_count.size()

    /* loop through all the .npy files here   */
    
    unsigned long file_offset = 0;

    for(int j = 0; j< num_files; j++){

        std::string cls_file = cls_npy_file + CLSFiles[j];
        meta cls_vec =  read_npy_metadata(cls_file);

        std::string bow_file = bow_npy_file + BOWFiles[j];
        meta bow_vec =  read_npy_metadata(bow_file);

        std::string json_file = bow_npy_file + JSONFiles[j];
        std::vector<int> bow_count = readIntegerArrayFromJson(json_file);

        cls_bytes_to_read = cls_vec.word_size*cls_vec.shape[1];

        //initialize cls_offset, bow_offset, and file_offset
        unsigned long cls_offset = cls_vec.dataStart;
        unsigned long bow_offset = bow_vec.dataStart;
        
        int num_documents = bow_count.size();

        std::cout<< "Rewriting file " << cls_file << std::endl;
        // loop through documents ids in the opened numpy files
        for(int i =0; i<num_documents; i++){  
            bow_bytes_to_read = bow_vec.word_size*bow_vec.shape[1]*bow_count[i];
            total_bytes_to_read = cls_bytes_to_read + bow_bytes_to_read;

            if (total_bytes_to_read > 4096){
                buffer = buffer_8k;
                buffer_size = 8192;
            }
            else{
                buffer = buffer_4k;
                buffer_size = 4096;
            }
            
            memset(buffer, 0, buffer_size); //clear memory 
            nread = pread(cls_vec.file, buffer, cls_bytes_to_read, cls_offset); //read cls vector
            cls_offset += cls_bytes_to_read;
            
            if (total_bytes_to_read >=8192){
                nread = pread(bow_vec.file, buffer+128, 7936, bow_offset); //read bow vector
            }
            else{
                nread = pread(bow_vec.file, buffer+128, bow_bytes_to_read, bow_offset); //read bow vector  
            }
            bow_offset += bow_bytes_to_read;


            //write the 4k aligned embeddings data back to storage
            nwrite = pwrite(embd_fd, buffer, buffer_size, file_offset);
            embd_metadata[doc_id].doc_id = doc_id;
            embd_metadata[doc_id].io_size = buffer_size;
            embd_metadata[doc_id].offset_size = file_offset;

            
            //print offset and buffer size
            //std::cout<< "Doc_id = " << doc_id << " io_size = " << embd_metadata[i].io_size << " offset_size = " << embd_metadata[i].offset_size <<std::endl;
            
            //update file offset and doc id
            file_offset += buffer_size;
            doc_id++;
        }
    }

    //save the embedding_offset array as a json file
    saveStructArrayToJson(embd_metadata, total_docs, embedding_metadata_file);
    std::cout<< "Metadata saved." <<std::endl;

    close(embd_fd);
    free(buffer_4k);
    free(buffer_8k);
    free(embd_metadata);

    return 0;
}