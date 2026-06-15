import torch.distributed as dist
import torch
from pathlib import Path


# error_rates = {4: 0.25, 5: 0.18, 8: 0.1, 2: 0.25} # shared error rate 
# error_rates = {4: 0.23, 5: 0.18, 8: 0.1, 2: 0.23} # llama
error_rates = {4: 0.20, 5: 0.18, 8: 0.08, 7: 0.1, 2: 0.27} # gemma
error_rates_scalars = {4: 0.2, 8: 0.08}


def correlated_rand_dir():
    return Path(__file__).resolve().parents[1] / "models" / "correlated_rand"


def load_correlated_rand_tensor(nclients, rank):
    path = correlated_rand_dir() / f"obj_{nclients}_{rank}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing correlated random tensor: {path}. "
            "Run `python simulations_llm/gen_correlated_rand.py` from the repository root first."
        )
    return torch.load(path)


# TODO: check. The quantized value range is += (1 << (nbit - 1)) (which is [-8, 8] for 4 bits, [-128, 128] for 8 bits). We cannot pack [-128, 128] into 8 bits
class Mee_Dynamic_Range(object):
    randomized_vec_pool = None
    def mapping(self, eps, l):
        return ((1 + 2 * eps * eps) ** l - 1) / (2 * eps * eps) * (1 + eps * eps) 

    def __init__(self, nbits, chunk_size, params):
        self.nbits = nbits
        self.chunk_size = chunk_size
        self.range = (1 << (self.nbits - 1)) - 1
        self.params = params
        self.cur_max_size = 0
        self.nclients = dist.get_world_size() if "world_size" not in params else params["world_size"]
        self.client_rank = dist.get_rank() if "rank" not in params else params["rank"]
        self.is_correlated = params["is_correlated"] if "is_correlated" in params else False

        for d_index in params["d"]:
            self.cur_max_size = max(self.cur_max_size, (params["d"][d_index] + chunk_size) // chunk_size)
        
        self.three_times_cur_max_size = (3 * self.cur_max_size) // self.nclients
        if Mee_Dynamic_Range.randomized_vec_pool is None:
            if self.is_correlated == False:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(self.client_rank)
                Mee_Dynamic_Range.randomized_vec_pool = torch.rand((self.three_times_cur_max_size, self.chunk_size), device="cuda", dtype=torch.bfloat16, generator=generator) * 65535.

            else:
                random_obj = load_correlated_rand_tensor(self.nclients, self.client_rank)
                random_obj = random_obj.to(torch.bfloat16)
                Mee_Dynamic_Range.randomized_vec_pool = (random_obj).to(device="cuda")[:self.three_times_cur_max_size * self.chunk_size].view(self.three_times_cur_max_size, self.chunk_size).mul_(65535.)

            # Mee_Dynamic_Range.randomized_vec_pool = torch.rand((self.three_times_cur_max_size, self.chunk_size), device="cuda", dtype=torch.bfloat16) * 65535.

        if self.is_correlated == False:
            self.rand_seed = (self.nbits * (self.client_rank + 1) * 998244353) & 0xFFFFFFFF
            self.rand_seed2 = (self.nbits * (self.client_rank + 1) * 1000007) & 0xFFFFFFFF
        else:
            self.rand_seed = [(self.nbits * (i + 1) * 998244353) & 0xFFFFFFFF for i in range(self.nclients)]
            self.rand_seed2 = (self.nbits * (self.client_rank + 1) * 1000007) & 0xFFFFFFFF

        nbit = self.nbits
        error_rate = error_rates[nbit]

        if "mee" in self.params["args"].aggregation_method:
            self.lookup_tab = torch.tensor([-self.mapping(error_rate, i) for i in range((1 << nbit), 0, -1)] + [self.mapping(error_rate, i) for i in range((1 << (nbit)) + 1)], device="cuda", dtype=torch.bfloat16) # 1 << nbit instead of 1 << (nbit - 1), so that "overflowed" values can still find an MEE lookup entry
        else:
            # combining mee with aee so here the mapping should be identity mapping
            self.lookup_tab = torch.tensor([-i for i in range((1 << nbit), 0, -1)] + [i for i in range((1 << (nbit)) + 1)], device="cuda", dtype=torch.bfloat16)

        # MEE related
        self.zero_val_index = 1 << nbit
        self.max_val_index = self.zero_val_index + (1 << (nbit - 1))

        self.max_val_scaling_factor = self.lookup_tab[self.max_val_index]

    def gen_next_rand(self, irank):
        if self.is_correlated == False:
            self.rand_seed = (self.rand_seed * 671431 + 1000000007) & 0xFFFFFFFF
            return self.rand_seed
        else:
            self.rand_seed[irank] = (self.rand_seed[irank] * 671431 + 1000000007) & 0xFFFFFFFF
            return self.rand_seed[irank]
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size

        if dtype == torch.uint8:
            padded_vec_size = padded_n # still use bfloat16
            padded_vec_size += (padded_n // self.chunk_size)

        else:
            padded_vec_size = padded_n
            padded_vec_size += (padded_n // self.chunk_size)

        return torch.zeros(padded_vec_size, dtype=torch.bfloat16, device="cuda") # check.
    
    def compress(self, interm_chunk, send_chunk, n, irank): # tmp chunk is of the same type as interm_chunk: float16
        
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size
        n_scale = padded_n // self.chunk_size
        
        two_dimen_interm_chunk = interm_chunk.view(-1, self.chunk_size)

        randl_mod = Mee_Dynamic_Range.randomized_vec_pool.numel() // self.chunk_size - two_dimen_interm_chunk.numel() // self.chunk_size
        
        randl = self.gen_next_rand(irank) % randl_mod
        rand_vec = Mee_Dynamic_Range.randomized_vec_pool[randl:randl + two_dimen_interm_chunk.numel() // self.chunk_size, :]
        
        tmp = torch.abs(two_dimen_interm_chunk)
        max_val = torch.max(tmp, dim=1, keepdim=True)[0]
        # max_val = torch.max(two_dimen_interm_chunk, dim=1, keepdim=True)[0]
        prob_pow = max_val.div_(self.max_val_scaling_factor).add_(1e-23)   # NOTE: 5.9e-8 is the smallest positive subnormal number in float16

        send_chunk_scales = send_chunk[padded_n:padded_n + n_scale]
        send_chunk_scales.copy_(prob_pow.view(-1))

        two_dimen_interm_chunk.div_(prob_pow)

        # MEE quantization
        indices = torch.searchsorted(self.lookup_tab, two_dimen_interm_chunk, right=True)
        v_left = torch.take(self.lookup_tab, indices - 1)
        v_right = torch.take(self.lookup_tab, indices)
        prob = (two_dimen_interm_chunk - v_left) / (v_right - v_left)

        indices = torch.where(rand_vec < prob * 65535, indices, indices - 1)

        indices -= (1 << self.nbits)
        indices = torch.clamp(indices, -self.range - 1, self.range + 1)

        send_chunk[:padded_n].copy_(indices.to(torch.bfloat16).view(-1))

    def compression(self, input_chunk, send_chunk, **kwargs): # compress input_chunk to send_chunk
        n = input_chunk.numel()

        self.compress(input_chunk, send_chunk, n, irank=kwargs["i"])
         
    def decompress(self, recv_chunk, interm_chunk, n):
        padded_n = (n + self.chunk_size - 1) // self.chunk_size * self.chunk_size
        n_scale = padded_n // self.chunk_size

        scale_chunk = recv_chunk[padded_n:padded_n + n_scale].view(-1, 1)  # shared memory

        interm_chunk.copy_(recv_chunk[:padded_n])

        data_chunk = interm_chunk.view(-1, self.chunk_size)
        data_chunk += (1 << self.nbits)
        data_chunk = torch.take(self.lookup_tab, data_chunk.long())

        data_chunk.mul_(scale_chunk)
        interm_chunk.copy_(data_chunk.view(-1))

    def decompression(self, recv_chunk, input_chunk, **kwargs): # recv_chunk, input_chunk, interm chunk are contiguous tensors. Decompress recv_chunk to input_chunk
        n = input_chunk.numel()
        self.decompress(recv_chunk, input_chunk, n)

    def dec_compression(self, recv_chunk, input_chunk, send_chunk, **kwargs): # recv_chunk, input_chunk, send_chunk, interm_chunk are contiguous tensors. Decompress recv_chunk to interm_chunk, add interm_chunk to  input_chunk, compress input_chunk to send_chunk
        n = input_chunk.numel()
        interm_chunk = input_chunk.clone()
        self.decompress(recv_chunk, interm_chunk[:n], n)
        
        input_chunk.add_(interm_chunk[:n])
        
        # input_chunk is not modifiable
        interm_chunk[:n].copy_(input_chunk)
        self.compress(interm_chunk[:n], send_chunk, n, irank=kwargs["i"])




class Mee_Dynamic_Range_Hierarchical(object):
    randomized_vec_pool = None

    def mapping(self, eps, l):
        return ((1 + 2 * eps * eps) ** l - 1) / (2 * eps * eps) * (1 + eps * eps)

    def __init__(self, nbits, chunk_size, params):
        self.nbits = nbits
        self.chunk_size = chunk_size
        self.supergroup = params["supergroup"]
        self.supergroup_size = self.supergroup * self.chunk_size
        self.chunk_quant = 8
        self.range = (1 << (self.nbits - 1)) - 1
        self.params = params
        self.cur_max_size = 0
        
        self.nclients = dist.get_world_size() if "world_size" not in params else params["world_size"]
        self.client_rank = dist.get_rank() if "rank" not in params else params["rank"]
        self.is_correlated = params["is_correlated"] if "is_correlated" in params else False

        for d_index in params["d"]:
            self.cur_max_size = max(self.cur_max_size, (params["d"][d_index] + chunk_size) // chunk_size)
        
        self.three_times_cur_max_size = (3 * self.cur_max_size) // self.nclients
        if Mee_Dynamic_Range_Hierarchical.randomized_vec_pool is None:
            if self.is_correlated == False:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(self.client_rank)
                Mee_Dynamic_Range_Hierarchical.randomized_vec_pool = torch.rand((self.three_times_cur_max_size, self.chunk_size), device="cuda", dtype=torch.bfloat16, generator=generator) * 65535.
            
        

            else:
                random_obj = load_correlated_rand_tensor(self.nclients, self.client_rank)
                random_obj = random_obj.to(torch.bfloat16)
                Mee_Dynamic_Range_Hierarchical.randomized_vec_pool = (random_obj).to(device="cuda")[:self.three_times_cur_max_size * self.chunk_size].view(self.three_times_cur_max_size, self.chunk_size).mul_(65535.)

                

        if self.is_correlated == False:
            self.rand_seed = (self.nbits * (self.client_rank + 1) * 998244353) & 0xFFFFFFFF
            self.rand_seed2 = (self.nbits * (self.client_rank + 1) * 1000007) & 0xFFFFFFFF
        else:
            self.rand_seed = [(self.nbits * (i + 1) * 998244353) & 0xFFFFFFFF for i in range(self.nclients)]
            self.rand_seed2 = (self.nbits * (self.client_rank + 1) * 1000007) & 0xFFFFFFFF

        
        nbit = self.nbits
        error_rate = error_rates[nbit]

        if "mee" in self.params["args"].aggregation_method:
            self.lookup_tab = torch.tensor([-self.mapping(error_rate, i) for i in range((1 << nbit), 0, -1)] + [self.mapping(error_rate, i) for i in range((1 << (nbit)) + 1)], device="cuda", dtype=torch.bfloat16)
        else:
            self.lookup_tab = torch.tensor([-i for i in range((1 << nbit), 0, -1)] + [i for i in range((1 << (nbit)) + 1)], device="cuda", dtype=torch.bfloat16)

        # MEE related
        self.zero_val_index = 1 << nbit
        self.max_val_index = self.zero_val_index + (1 << (nbit - 1))

        self.max_val_scaling_factor = self.lookup_tab[self.max_val_index]

    def gen_next_rand(self, irank):
        if self.is_correlated == False:
            self.rand_seed = (self.rand_seed * 671431 + 1000000007) & 0xFFFFFFFF
            return self.rand_seed
        else:
            self.rand_seed[irank] = (self.rand_seed[irank] * 671431 + 1000000007) & 0xFFFFFFFF
            return self.rand_seed[irank]
        

    def gen_next_rand2(self):
        self.rand_seed2 = (self.rand_seed2 * 536267 + 1000000007) & 0xFFFFFFFF
        return self.rand_seed2
    
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()
        padded_n = (n + self.supergroup_size - 1) // self.supergroup_size * self.supergroup_size

        if dtype == torch.uint8:
            padded_vec_size = padded_n # still use bfloat16. all numbers in [-128, 128] can be precisely recorded as bfloat16 numbers!
            padded_vec_size += (padded_n // self.chunk_size) # for the scalars of the first layer
            # we encode the first-layer scalars as bfloat16, so there is no need to encode the second-layer scalars here for simulation but we instead record the "encoded-and-then-decoded" first layer scalars
            # padded_vec_size += (padded_n // self.supergroup_size)

        else:
            padded_vec_size = padded_n
            padded_vec_size += (padded_n // self.chunk_size)

        return torch.zeros(padded_vec_size, dtype=torch.bfloat16, device="cuda") # check.
    
    def compress(self, interm_chunk, send_chunk, n, irank): # tmp chunk is of the same type as interm_chunk: float16
        padded_n = (n + self.supergroup_size - 1) // self.supergroup_size * self.supergroup_size
        n_scale_first = padded_n // self.chunk_size
        
        two_dimen_interm_chunk = interm_chunk.view(-1, self.chunk_size)

        randl_mod = Mee_Dynamic_Range_Hierarchical.randomized_vec_pool.numel() // self.chunk_size - two_dimen_interm_chunk.numel() // self.chunk_size
        
        randl = self.gen_next_rand(irank) % randl_mod
        rand_vec = Mee_Dynamic_Range_Hierarchical.randomized_vec_pool[randl:randl + two_dimen_interm_chunk.numel() // self.chunk_size, :]
        
        tmp = torch.abs(two_dimen_interm_chunk)
        max_val = torch.max(tmp, dim=1, keepdim=True)[0]
        # max_val = torch.max(two_dimen_interm_chunk, dim=1, keepdim=True)[0]
        prob_pow = max_val.div_(self.max_val_scaling_factor).add_(1e-23)   # NOTE: 5.9e-8 is the smallest positive subnormal number in float16

        two_dimen_interm_chunk.div_(prob_pow)

        # MEE quantization
        indices = torch.searchsorted(self.lookup_tab, two_dimen_interm_chunk, right=True)
        v_left = torch.take(self.lookup_tab, indices - 1)
        v_right = torch.take(self.lookup_tab, indices)
        prob = (two_dimen_interm_chunk - v_left) / (v_right - v_left)

        indices = torch.where(rand_vec < prob * 65535, indices, indices - 1)

        indices -= (1 << self.nbits)
        indices = torch.clamp(indices, -self.range - 1, self.range + 1)

        send_chunk[:padded_n].copy_(indices.to(torch.bfloat16).view(-1))

        # quantization for the first layer
        prob_pow = prob_pow.view(-1, self.supergroup)
        supergroup_max_val = torch.max(prob_pow, dim=1, keepdim=True)[0].add_(1e-23)
        nlevels = 1 << self.chunk_quant
        prob_pow = prob_pow / supergroup_max_val * (nlevels - 1) # quantize to [0, nlevels - 1]

        floored_prob_pow = torch.floor(prob_pow)
        prob = prob_pow - floored_prob_pow
        nrow = (prob_pow.numel() + self.chunk_size - 1) // self.chunk_size # chunk size for randomized_vec_pool
        randl_mod = Mee_Dynamic_Range_Hierarchical.randomized_vec_pool.numel() // self.chunk_size - nrow - 1
        rand_l = self.gen_next_rand2() % randl_mod
        rand_vec = Mee_Dynamic_Range_Hierarchical.randomized_vec_pool[rand_l:rand_l + nrow, :].view(-1)[:prob_pow.numel()].view(-1, self.supergroup)

        floored_prob_pow += (rand_vec < prob * 65535)

        prob_pow = (floored_prob_pow * supergroup_max_val / (nlevels - 1)).view(-1, 1) # encoded and then decoded first-layer scalars

        send_chunk_scales = send_chunk[padded_n:padded_n + n_scale_first]
        send_chunk_scales.copy_(prob_pow.view(-1))




    def compression(self, input_chunk, send_chunk, **kwargs): # compress input_chunk to send_chunk
        n = input_chunk.numel()
        
        assert("i" in kwargs)

        self.compress(input_chunk, send_chunk, n, irank=kwargs["i"])
        
    
    def decompress(self, recv_chunk, interm_chunk, n):
        padded_n = (n + self.supergroup_size - 1) // self.supergroup_size * self.supergroup_size

        n_scale = padded_n // self.chunk_size
        scale_chunk = recv_chunk[padded_n:padded_n + n_scale].view(torch.bfloat16).view(-1, 1)  # shared memory

        interm_chunk.copy_(recv_chunk[:padded_n])

        data_chunk = interm_chunk.view(-1, self.chunk_size)
        data_chunk += (1 << self.nbits)
        data_chunk = torch.take(self.lookup_tab, data_chunk.long())

        data_chunk.mul_(scale_chunk)
        interm_chunk.copy_(data_chunk.view(-1))

    def decompression(self, recv_chunk, input_chunk, **kwargs): # recv_chunk, input_chunk, interm chunk are contiguous tensors. Decompress recv_chunk to input_chunk
        n = input_chunk.numel()
        self.decompress(recv_chunk, input_chunk, n)

    def dec_compression(self, recv_chunk, input_chunk, send_chunk, **kwargs): # recv_chunk, input_chunk, send_chunk, interm_chunk are contiguous tensors. Decompress recv_chunk to interm_chunk, add interm_chunk to  input_chunk, compress input_chunk to send_chunk
        assert("i" in kwargs)
        n = input_chunk.numel()
        interm_chunk = input_chunk.clone()
        self.decompress(recv_chunk, interm_chunk[:n], n)
        
        input_chunk.add_(interm_chunk[:n])
        
        interm_chunk[:n].copy_(input_chunk)
        self.compress(interm_chunk[:n], send_chunk, n, irank=kwargs["i"])



class Direct_Summation(object): # inputs should be integers. For THC's direct summation approach.
    def __init__(self, params):
        self.params = params
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()

        return torch.zeros(n, dtype=torch.bfloat16, device="cuda") # check.
    
    def compress(self, interm_chunk, send_chunk, n): # interm_chunk: "original" summation of quantized values with overflowed values
        send_chunk[:n].copy_(interm_chunk[:n])

    def compression(self, input_chunk, send_chunk, **kwargs): # compress input_chunk to send_chunk
        n = input_chunk.numel()

        self.compress(input_chunk, send_chunk, n)
        
    
    def decompress(self, recv_chunk, interm_chunk, n):
        interm_chunk.copy_(recv_chunk[:n])

    def decompression(self, recv_chunk, input_chunk, **kwargs): # recv_chunk, input_chunk, interm chunk are contiguous tensors. Decompress recv_chunk to input_chunk
        n = input_chunk.numel()
        self.decompress(recv_chunk, input_chunk, n)

    def dec_compression(self, recv_chunk, input_chunk, send_chunk, **kwargs): # recv_chunk, input_chunk, send_chunk, interm_chunk are contiguous tensors. Decompress recv_chunk to interm_chunk, add interm_chunk to  input_chunk, compress input_chunk to send_chunk
        n = input_chunk.numel()
        input_chunk.add_(recv_chunk[:n])
        
        send_chunk[:n].copy_(input_chunk[:n])







class BFloat16_compression(object):
    def __init__(self, params):
        self.params = params
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()

        return torch.zeros(n, dtype=dtype, device="cuda")
    
    def compression(self, input_chunk, send_chunk, **kwargs): # send_chunk in float16; input_chunk in float32
        n = input_chunk.numel()
        send_chunk[:n].copy_(input_chunk)

    def decompression(self, recv_chunk, input_chunk, **kwargs):
        n = input_chunk.numel()
        input_chunk.copy_(recv_chunk[:n])
    
    def dec_compression(self, recv_chunk, input_chunk, send_chunk, **kwargs):
        n = input_chunk.numel()
        input_chunk.add_(recv_chunk[:n])
        send_chunk[:n].copy_(input_chunk)



class Float8_compression(object):
    def __init__(self, params, **kwargs):
        self.params = params
    
    def create_tensor(self, tensor, dtype):
        n = tensor.numel()

        return torch.zeros(n, dtype=dtype, device="cuda")
    
    def compression(self, input_chunk, send_chunk, **kwargs): # send_chunk in float8_e4m3fn or float8_e5m2
        n = input_chunk.numel()
        send_chunk[:n].copy_(input_chunk)
        # send_chunk[:n].copy_(input_chunk.to(torch.float8_e4m3fn).to(torch.bfloat16))

    def decompression(self, recv_chunk, input_chunk, **kwargs): # recv_chunk in float8_e4m3fn or float8_e5m2
        n = input_chunk.numel()
        input_chunk.copy_(recv_chunk[:n])
    
    def dec_compression(self, recv_chunk, input_chunk, send_chunk, **kwargs):
        n = input_chunk.numel()
        temp_chunk = input_chunk.to(torch.float8_e4m3fn) # input chunk should be 8 bits, but pytorch does not support addition for float8 yet...
        input_chunk.copy_(temp_chunk)
        input_chunk.add_(recv_chunk[:n].to(torch.bfloat16))
        send_chunk[:n].copy_(input_chunk)



# FP6
class Fpx_arithmetics(object):
    def __init__(self, num_exp, num_mantissa, params):
        self.num_exp = num_exp
        self.num_mantissa = num_mantissa
        self.max_exponent = (2 ** self.num_exp)
        self.fpx_max = 2 ** (2**(self.num_exp - 1)) * (2**(self.num_mantissa + 1) - 1) // (2 ** self.num_mantissa)
        self.params = params
        if num_exp == 2:
            self.rounding_add = 0.43
        else:
            self.rounding_add = 0.4
        

    def fpx_downcast(self, src_tensor: torch.Tensor):
        
        sign = torch.where(src_tensor >= 0, 1, -1)
        src_tensor = torch.abs(src_tensor)

        src_tensor = torch.clamp(src_tensor, -self.fpx_max, self.fpx_max)
        exponent = torch.where(src_tensor > self.fpx_max, self.max_exponent - 1, torch.floor(torch.log2(src_tensor) + (2 ** (self.num_exp - 1)) - 1))
        exponent = torch.where(exponent < 0, 0, exponent)

        # TODO
        # mantissa = (src_tensor / (2 ** (exponent - (-1 + 2 ** (self.num_exp - 1)))) * (2 ** self.num_mantissa) + 0.499).to(torch.int32)
        # mantissa = torch.where(exponent == 0, (src_tensor / (2 ** (exponent - (-1 + 2 ** (self.num_exp - 1)))) * (2 ** (self.num_mantissa - 1)) + 0.499).to(torch.int32) * 2, mantissa)
        mantissa = (src_tensor / (2 ** (exponent - (-1 + 2 ** (self.num_exp - 1)))) * (2 ** self.num_mantissa) + self.rounding_add).to(torch.int32)
        # mantissa = torch.where(exponent == 0, (src_tensor / (2 ** (exponent - (-1 + 2 ** (self.num_exp - 1)))) * (2 ** (self.num_mantissa - 1)) + self.rounding_add).to(torch.int32) * 2, mantissa)
        mantissa = torch.where(exponent == 0, (src_tensor / (2 ** (exponent - (-1 + 2 ** (self.num_exp - 1)))) * (2 ** (self.num_mantissa - 1))).to(torch.int32) * 2, mantissa)
        
        return sign, exponent, mantissa

    def fpx_upcast(self, sign, exponent, mantissa):
        return torch.where(exponent < self.max_exponent, sign * (mantissa.to(torch.float32) / (2 ** self.num_mantissa) * (2 ** (exponent - (-1 + 2 ** (self.num_exp - 1))).to(torch.float32))), self.fpx_max * sign)

    def fpx_compress_decompress(self, src_tensor: torch.Tensor):
        sign, exponent, mantissa = self.fpx_downcast(src_tensor)
        return self.fpx_upcast(sign, exponent, mantissa)

    def create_tensor(self, tensor, dtype):
        n = tensor.numel()

        return torch.zeros(n, dtype=torch.bfloat16, device="cuda")

    def compression(self, input_chunk, send_chunk, **kwargs):
        n = input_chunk.numel()
        results = self.fpx_compress_decompress(input_chunk[:n])
        send_chunk[:n].copy_(results.to(torch.bfloat16))
    
    def decompression(self, recv_chunk, input_chunk, **kwargs):
        n = input_chunk.numel()
        input_chunk[:n].copy_(recv_chunk[:n])
    
    def dec_compression(self, recv_chunk, input_chunk, send_chunk, **kwargs):
        n = input_chunk.numel()
        input_chunk.copy_(self.fpx_compress_decompress(input_chunk[:n]).to(torch.bfloat16))
        input_chunk.add_(recv_chunk[:n])
        self.compression(input_chunk, send_chunk, **kwargs)





