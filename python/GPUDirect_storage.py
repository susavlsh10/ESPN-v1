import torch
import json
import numpy as np

# install GPUDirect before using module
import GPUDirect

class GPUDirect_Embeddings():
    def __init__(self, gpuid: int):
        self.GDS_8k = None
        self.offsets = None
        self.io_size = None

        #reserve GPU memory for GDS reads
        self.CLS_MAT_8k = None
        self.BOW_MAT_8k = None

        self.gpuid = gpuid

    def _gds_init(self, file_path:str, offset_path: str, io_size_path: str, batch_size_8k: int, num_batches_8k: int):

        cls_size = 128
        bow_size = 32
        self.GDS_8k = GPUDirect.GPUDirect_IO(self.gpuid, file_path, 8192, batch_size_8k, num_batches_8k, cls_size, bow_size)
        
        #load offsets
        self._load_gds_metadata(offset_path, io_size_path)
        print('GDS metadata loaded.')

        #prepare buffers
        self._create_buffers(batch_size_8k, num_batches_8k)
        print('GPU buffers created.')

    def _load_gds_metadata(self, offset_path: str, io_size_path: str):
        self.offsets = torch.load(offset_path)
        io = torch.load(io_size_path)
        self.io_size = io.to(torch.int)

    def _load_gds_metadata_from_json(self, metadata_path: str):
        # store offsets and io size
        # Read the JSON file
        with open(metadata_path, "r") as file:
            data = json.load(file)

        # allocate memory for offsets
        self.offsets = torch.zeros(len(data), dtype=int)        
        self.io_size = torch.zeros(len(data), dtype=bool)

        # allocate memory for io size 
        for item in data:
            doc_id = item["doc_id"]
            self.offsets[doc_id] = item["offset_size"]
            if item["io_size"] == 4096:
                self.io_size[doc_id] = True
            else:
                self.io_size[doc_id] = False 
    
    def _create_buffers(self, batch_size_8k, num_batches_8k):
        
        ''' If we need to use 4k io size
        max_documents_4k = batch_size_4k * num_batches_4k
        num_bow_vectors_4k =  int((4096-128*2)/(32*2))
        cls_shape_4k = (max_documents_4k, 128)
        bow_shape_4k = (max_documents_4k, num_bow_vectors_4k, 32)
        self.CLS_MAT_4k = torch.zeros(cls_shape_4k, dtype=dtype, device=device)
        self.BOW_MAT_4k = torch.zeros(bow_shape_4k, dtype=dtype, device=device)
        self.GDS_4k._prepare_batches()
        '''
        max_documents_8k = batch_size_8k * num_batches_8k
        num_bow_vectors_8k =  int((8192-128*2)/(32*2))
        cls_shape_8k = (max_documents_8k, 128)
        bow_shape_8k = (max_documents_8k, num_bow_vectors_8k, 32)

        dtype = torch.float16
        if self.gpuid == 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:1')
            
        self.CLS_MAT_8k = torch.zeros(cls_shape_8k, dtype=dtype, device=device)
        self.BOW_MAT_8k = torch.zeros(bow_shape_8k, dtype=dtype, device=device)
        self.GDS_8k._prepare_batches()

    def read_double(self, doc_ids):
        #check io size list and divide doc_ids into 4k and 8k list
        '''
        This method is not currently supported. # slow performance.
        '''
        if not torch.is_tensor(doc_ids):
            doc_ids = torch.tensor(doc_ids)

        doc_io_size = self.io_size[doc_ids]
        indices_4k = torch.where(doc_io_size == 1)[0]
        indices_8k = torch.where(doc_io_size == 0)[0]   
        
        doc_4k = doc_ids[indices_4k]
        doc_8k = doc_ids[indices_8k]

        doc_4k_len = doc_4k.shape[0]
        doc_8k_len = doc_8k.shape[0]

        offsets_4k = self.offsets[doc_4k]
        offsets_8k = self.offsets[doc_8k]
        
        #read from GDS_4k
        self.GDS_4k.read(offsets_4k, self.CLS_MAT_4k, self.BOW_MAT_4k)

        #read from GDS_8k if not empty

        if offsets_8k.shape[0] != 0:
            self.GDS_8k.read(offsets_8k, self.CLS_MAT_8k, self.BOW_MAT_8k)
        
        return (self.CLS_MAT_4k[0:doc_4k_len], self.BOW_MAT_4k[0:doc_4k_len], self.CLS_MAT_8k[0:doc_8k_len], self.BOW_MAT_8k[0:doc_8k_len]), (doc_4k, doc_8k)

    def read(self, doc_ids):

        if not torch.is_tensor(doc_ids):
            doc_ids = torch.tensor(doc_ids)
        
        num_ids = len(doc_ids)

        doc_io_size = self.io_size[doc_ids]
        doc_offsets= self.offsets[doc_ids]   #doc_offsets needs to be in CPU memory

        doc_io_size = doc_io_size.cuda()    #doc_io_size tensor needs to be in GPU before calling read_mixed

        #read from GDS_8k if not empty
        self.GDS_8k.read_mixed(doc_offsets, doc_io_size, self.CLS_MAT_8k, self.BOW_MAT_8k)
        
        return self.CLS_MAT_8k[0:num_ids], self.BOW_MAT_8k[0:num_ids]

    def _reset_batch(self):
        self.GDS_8k._reset_batch()
        self.CLS_MAT_8k.fill_(0)
        self.BOW_MAT_8k.fill_(0)

    def _reset_buffer(self):
        self.GDS_8k._reset_buffer()

    def _close(self):
        self.GDS_8k._close()




