import torch
from torch.nn.functional import pad
import numpy as np
import math
from pathlib import Path
import torch.distributed as dist

from .utils_aggregation import Direct_Summation
from .utils import composable_allreduce_arbitrary_callback

HADAMARD_MAX_RANDOM_DIMENSION = 2 ** 28
hadamard_random_vec_pool = None

initial_seed = 0


def correlated_rand_dir():
    return Path(__file__).resolve().parents[1] / "models" / "correlated_rand"


def load_hadamard_random_vec():
    path = correlated_rand_dir() / "obj_hadamard.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Hadamard random vector: {path}. "
            "Run `python simulations_llm/gen_correlated_rand.py` from the repository root first."
        )
    return torch.load(path)


def _resolve_table_dir(params):
    default_table_dir = Path(__file__).resolve().parents[1] / "compression" / "new_tables"
    configured_table_dir = params.get("table_dir")
    if configured_table_dir is None:
        return default_table_dir

    configured_path = Path(configured_table_dir)
    candidates = [configured_path] if configured_path.is_absolute() else [
        Path.cwd() / configured_path,
        Path(__file__).resolve().parents[1] / configured_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default_table_dir


class Hadamard:
    def __init__(self, dim, seed, device, to_perm=False):
        global hadamard_random_vec_pool, initial_seed
        global HADAMARD_MAX_RANDOM_DIMENSION
        if hadamard_random_vec_pool == None:
            random_obj = load_hadamard_random_vec()
            random_obj = random_obj.to(torch.bfloat16).to(device)
            hadamard_random_vec_pool = random_obj

        self.d = dim
        self.device = device

        self.rand_seed = initial_seed + 1
        self.randl = 0
        self.randr = 0
        initial_seed += 1
        
        if self.d & (self.d - 1) != 0:
            raise Exception("input numel must be a power of 2")
            
    def hadamard(self, vec):
        
        d = vec.numel()
        assert(d == self.d)


        h = 2
        while h <= d:
            hf = h // 2
            vec = vec.view(d // h,h)
            vec[:, :hf]  = vec[:, :hf] + vec[:, hf:2*hf]
            vec[:, hf:2*hf] = vec[:, :hf] - 2*vec[:, hf:2*hf]
            h *= 2
        
        return vec.view(-1)

    def get_next_rand_seed(self):
        self.rand_seed = (self.rand_seed * 671431 + 1000000007) & 0xFFFFFFFF
        return self.rand_seed 


    def rht(self, vec: torch.Tensor):
        # event = torch.cuda.Event()

        self.randl = self.get_next_rand_seed() % (HADAMARD_MAX_RANDOM_DIMENSION - self.d + 1)
        self.randr = self.randl + self.d

        random_diagonal = hadamard_random_vec_pool[self.randl: self.randr]
        self.random_diagonal_randrange = (self.randl, self.randr)

        dim = vec.numel()
        
        vec = vec.mul_(random_diagonal)

        vec = self.hadamard(vec)
        
        vec = vec.div_(math.sqrt(dim))

        return vec
        
    def irht(self, vec: torch.Tensor, is_decompress=0):
        
        self.randl, self.randr = self.random_diagonal_randrange
        random_diagonal = hadamard_random_vec_pool[self.randl: self.randr]
        
        dim = vec.numel()

        vec = self.hadamard(vec)

        vec = vec.div_(math.sqrt(dim))
        
        vec = vec.mul_(random_diagonal)
        
        return vec

class OldINCACompressor(object):
    def __init__(self, params):
        self.device = params.get('device', 'cuda') 

        self.ds = params['d']
        self.original_size = params["size"]


        self.seed = params.get('seed', 42)
        
        self.prng = torch.Generator(device=self.device)
        self.prng.manual_seed(self.seed)
                
        
        self.quantization_levels = params.get('quantization_levels', 16)
        self.overflow_frequency = params.get('overflow_frequency', 1024)
        self.smaxval = params.get('max_val', 42)
        self.table_size = params.get('table_size', 10001)
        
        self.hadamards = dict()
        for name_idx in self.ds:
            self.hadamards[name_idx] = Hadamard(self.ds[name_idx], self.seed, self.device)


        self.sender_prng = torch.Generator(device=self.device)

        table_dir = _resolve_table_dir(params)
        fn = str(table_dir / '{}_tablesize_{}_maxval_{}_qlevels_{}_ofreq_'.format(
            self.table_size,
            self.smaxval,
            16,
            self.overflow_frequency,
        ))

        self.data = self.sender_table(fn)

        self.nclients = params.get('nclients', 10)
        self.max_norm_dict = dict()

    def sender_table(self, prefix):
        data = eval(open(prefix + 'data.txt').read())
        return data

    
            
    def rvec_compress(self, tensor, max_norm, dim):
        max_coordinate = self.data['T'] * max_norm / np.sqrt(dim)
        min_coordinate = -max_coordinate # hadamard-transformed tensor, the estimation of min_coordinate is always -max_coordinate

        delta = (max_coordinate - min_coordinate) / (self.quantization_levels - 1) + 1e-23

        tensor.sub_(min_coordinate).div_(delta)
        tensor = torch.clamp_(tensor, min=0, max=self.quantization_levels - 1)
        tensor2 = tensor.clone()
        tensor = tensor.floor_()
        p = tensor2.sub_(tensor)

        tensor = tensor.add_(p.bernoulli_(generator=self.sender_prng))
        
        
        return tensor.to(torch.bfloat16), tensor, min_coordinate, delta


    """compression."""
    def compress(self, tensor, name, max_norm): # name: (index_id, partition_id)
        """Returns the tensor unmodified."""

        orig_size = dim = padded_dim = self.original_size[name]
        self.max_norm_dict[name] = max_norm


        if not dim & (dim - 1) == 0:
            padded_dim = max(64, int(2**(np.ceil(np.log2(dim)))))
            paddings = pad(tensor, (0, padded_dim - dim), mode='constant', value=0)
            tensor = paddings

        
        temp1 = self.hadamards[name].rht(tensor)
        
        tensor, _, _, _ = self.rvec_compress(temp1, max_norm, padded_dim)
        return tensor

    """Uncompress the tensor."""
    def decompress(self, tensor, name, max_norm=None):
        """Returns the tensor unmodified."""
        # return tensor.float()
        if max_norm == None:
            max_norm = self.max_norm_dict[name]

        max_coordinate = self.data['T'] * max_norm / np.sqrt(self.ds[name])
        min_coordinate = -max_coordinate
        delta = (max_coordinate - min_coordinate) / (self.quantization_levels - 1) + 1e-23
        

        tensor = tensor.float()
        tensor.mul_(delta / self.nclients).add_(min_coordinate)

        tensor = self.hadamards[name].irht(tensor)

        return tensor[:self.original_size[name]]






def P2P_THC_compress_hook(
    state, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    INTEG_PARTITION_LAYER = 4
    CHUNK_SIZE_THRESHOLD = 1 << 23 # 8 MB
    def compress_and_reduce_and_decompress(fut, state, index, l, r, no_hadamard=False):
        fut.wait()
        try:

            for i in range(l, r):
                sl = state["start_idx"][(index, i)]
                sr = sl + state["params"]["size"][(index, i)]
                
                tensor = vec[sl:sr]

                norm_tensor = torch.norm(tensor, 2)
                dist.all_reduce(norm_tensor, async_op=False, op=dist.ReduceOp.MAX)
                reduced_max_norm = norm_tensor

                first_coordinate = tensor[0].clone().view(-1)
                dist.all_reduce(first_coordinate, async_op=False, op=dist.ReduceOp.SUM)
                first_coordinate /= dist.get_world_size()

                compressed_tensor = state["params"]["compressor"].compress(tensor, (index, i), reduced_max_norm)

                aggregated_tensor = composable_allreduce_arbitrary_callback(compressed_tensor, state["params"]["callback_comm"], state["params"], dtype=torch.bfloat16, tag=(index << 8))
                
                aggregated_tensor = state["params"]["compressor"].decompress(aggregated_tensor, (index, i), reduced_max_norm)
                aggregated_tensor[0] = first_coordinate[0]

                ret_tensor[sl:sr].copy_(aggregated_tensor.view(-1)[:sr - sl])
            
            return ret_tensor
                

        except Exception:
            raise
    
    def return_func(fut):
        return ret_tensor
    
    state = state

    if state["batch_idx"] == 0:
        return (
            dist.all_reduce(bucket.buffer() / dist.get_world_size(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )

    elif state["batch_idx"] == 1:
        
        index = bucket.index()
        total_size = bucket.buffer().numel()
        orig_total_size = total_size
        start_interm_idx = 0
        i = 0
        while True:
            if i >= INTEG_PARTITION_LAYER - 1 and total_size <= CHUNK_SIZE_THRESHOLD:
            # if i >= INTEG_PARTITION_LAYER - 1:
                cur_size = total_size
                cur_d = 1 << ((total_size - 1).bit_length())
            else:
                cur_size = min(CHUNK_SIZE_THRESHOLD, 1 << (total_size.bit_length() - 1))
                # cur_size = 1 << (total_size.bit_length() - 1)
                cur_d = cur_size

            state["params"]["d"][(index, i)] = cur_d
            state["params"]["size"][(index, i)] = cur_size
            state["start_idx"][(index, i)] = orig_total_size - total_size
            state["start_interm_idx"][(index, i)] = start_interm_idx
            
            start_interm_idx += cur_d

            total_size -= cur_size
            i += 1
            if total_size == 0:
                break

        state["partition_len"][index] = i

        state["params"]["chunk_size"] = 64
        
        if bucket.is_last():
            state["params"]["callback_comm"] = Direct_Summation(state["params"])

            state["params"]["compressor"] = OldINCACompressor(state["params"])

        return (
            dist.all_reduce(bucket.buffer() / dist.get_world_size(), async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
        )
        
    else:
        index = bucket.index()
        
        vec = bucket.buffer()
        total_size = bucket.buffer().numel()
        ret_tensor = bucket.buffer()


        init_future = torch.futures.Future()
        init_future.set_result(0)

        l = 0
        r = state["partition_len"][index]

        reduce_future = init_future.then(lambda fut: compress_and_reduce_and_decompress(fut, state, index, l, r, True)) #bucket.is_last()))

        return reduce_future
