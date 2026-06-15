import torch
from torch.nn.functional import pad

class NewINCACompressor(object):
    def __init__(self, params):
        self.device = params.get('device', 'cuda') 
        

        self.ds = params['d']
        self.original_size = params["size"]
        self.to_rescale = params.get("to_rescale", True)
        
        self.seed = params.get('seed', 42)
        self.nclients = params.get('nclients', 1)

        self.max_chunk_size = params.get("chunk_size", 0)
        self.compress_vec = dict()
        self.max_memory = dict()
        self.padded_tensor = dict()

        self.max_memory_chunk_size = params.get("max_chunk_size", self.max_chunk_size)

        # for name_idx in self.ds:
        #     dim = self.original_size[name_idx]
        #     self.padded_tensor[name_idx] = torch.zeros(((self.ds[name_idx] + self.max_chunk_size - 1) // self.max_chunk_size * self.max_chunk_size), dtype=torch.bfloat16, device=self.device)
        #     self.max_memory[name_idx] = torch.zeros(((self.ds[name_idx] + self.max_chunk_size - 1) // self.max_chunk_size), dtype=torch.bfloat16, device=self.device)


    def padding_tensor(self, tensor, name):
        orig_size = dim = padded_dim = self.original_size[name]
        padded_tensor = pad(tensor, (0, (self.max_chunk_size - orig_size % self.max_chunk_size) % self.max_chunk_size), mode='constant', value=0)
        padded_tensor = padded_tensor / self.nclients
        
        max_memory = torch.zeros((orig_size + self.max_memory_chunk_size - 1) // self.max_memory_chunk_size, dtype=torch.bfloat16, device=self.device)
        
        return padded_tensor, max_memory

    def max_memory_tensor(self, name):
        orig_size = self.original_size[name]
        
        max_memory = torch.zeros((orig_size + self.max_memory_chunk_size - 1) // self.max_memory_chunk_size, dtype=torch.bfloat16, device=self.device)
        return max_memory

    
