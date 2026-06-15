
import torch
import numpy as np
import warnings
import math
import pdb

from scipy.stats import norm

##############################################################################
##############################################################################

class Baseline:
    
    def __init__(self, params=None):
        pass
    
    def roundtrip(self, client_grad_vecs):        
        return torch.mean(torch.stack((client_grad_vecs)), dim=0)

##############################################################################
##############################################################################

class TopK:
    
    def __init__(self, params):
        
        self.kp = params['kp']
        self.ef = params['ef']
        self.d = params['d']
        
        if self.ef:
            
            self.errors = {}
                          
    def roundtrip(self, client_grad_vecs):
        
        device = client_grad_vecs[0].device
        nclients = len(client_grad_vecs)
        
        res = torch.zeros(size=(self.d,), device=device)
               
        for i in range(nclients):
            
            if self.ef:
                                
                self.errors[i] = client_grad_vecs[i] + self.errors.get(i,0)
                _, idx = self.errors[i].abs().topk(int(self.kp * self.d))
                res[idx] += self.errors[i][idx]

                self.errors[i][idx] = 0
                
            else:
                                
                _, idx = client_grad_vecs[i].abs().topk(int(self.kp * self.d))
                res[idx] += client_grad_vecs[i][idx]
            
        return res/nclients

##############################################################################
##############################################################################

class Hadamard:
    
    def __init__(self, dim, seed, device):
        
        self.device = device
        self.d = dim
        self.prng = torch.Generator(device=device)
        self.prng.manual_seed(seed)
        self.random_diagonal = 2 * torch.bernoulli(torch.ones(size=(int(2**(np.ceil(np.log2(dim)))),), device=device) / 2, generator=self.prng) - 1
            
    def hadamard(self, vec):
        
        d = vec.numel()
        if d & (d-1) != 0:
            raise Exception("input numel must be a power of 2")
          
        h = 2
        while h <= d:        
            hf = h//2
            vec = vec.view(d//h,h)
            vec[:,:hf]  = vec[:,:hf] + vec[:,hf:2*hf]
            vec[:,hf:2*hf] = vec[:,:hf] - 2*vec[:,hf:2*hf]
            h *= 2   
        vec /= np.sqrt(d)
        
        return vec.view(-1)

    def rht(self, vec):

        dim = vec.numel()
        
        if not dim & (dim - 1) == 0:
            
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = vec
            
            padded_vec = padded_vec * self.random_diagonal
            padded_vec = self.hadamard(padded_vec)
            
            return padded_vec
        
        else:   
            
            vec = vec * self.random_diagonal
            vec = self.hadamard(vec)
            
            return vec
        
    def irht(self, vec):
        
        vec = self.hadamard(vec)
        vec = vec * self.random_diagonal
        
        return vec[:self.d]
    
##############################################################################
##############################################################################

class INCA_old:
    
    def __init__(self, params):
        
        self.device = params['device']  

        self.d = params['d']
        self.seed = params['seed']
        
        self.prng = torch.Generator(device=params['device'])
        self.prng.manual_seed(params['seed'])
                        
        self.ef = params['ef']
        self.rotation = params['rotation']
        
        self.quantization_levels = params['quantization_levels']

        if self.ef:
            self.errors = {}
        
        self.norm_normalization = params.get('norm_normalization', True)
        if self.norm_normalization:
            self.per_coordinate_overflow_prob = params.get('per_coordinate_overflow_prob', 1 / (2 * params['overflow_frequency']))
        
        self.hadamard = Hadamard(self.d, self.seed, self.device) 
        
    def min_max_stochastic_quantization(self, vecs):
        
        cloned_vecs = [vec.clone() for vec in vecs]
        
        min_coordinate = min([vec.min() for vec in cloned_vecs])
        max_coordinate = max([vec.max() for vec in cloned_vecs])
        
        delta = (max_coordinate - min_coordinate) / (self.quantization_levels - 1)
        
        for i in range(len(cloned_vecs)):
            
            cloned_vecs[i] = (cloned_vecs[i] - min_coordinate) / delta
            cloned_vecs[i] = torch.floor(cloned_vecs[i]) + torch.bernoulli(cloned_vecs[i]-torch.floor(cloned_vecs[i]), generator=self.prng)   
        
        return cloned_vecs, min_coordinate, delta
    
    def norm_stochastic_quantization(self, vecs):
        
        cloned_vecs = [vec.clone() for vec in vecs]
        
        dim = cloned_vecs[0].numel() ### might be padded -> self.d might be wrong
        max_norm = max([vec.norm(2) for vec in cloned_vecs])
        
        max_coordinate = norm.isf(self.per_coordinate_overflow_prob , scale=(max_norm/np.sqrt(dim)).cpu())
        min_coordinate = -max_coordinate
                
        delta = (max_coordinate - min_coordinate) / (self.quantization_levels - 1)
        
        for i in range(len(cloned_vecs)):
            
            overflow_p = ((cloned_vecs[i] > max_coordinate).sum() + (cloned_vecs[i] < min_coordinate).sum()) / float(dim)
            if not self.ef and overflow_p > 0:
                warnings.warn('quantization overflow with no error feedback detected: {}% overflow'.format(overflow_p))

            cloned_vecs[i] = (cloned_vecs[i] - min_coordinate) / delta
            cloned_vecs[i] = torch.clamp(cloned_vecs[i], min=0, max=self.quantization_levels-1)
            cloned_vecs[i] = torch.floor(cloned_vecs[i]) + torch.bernoulli(cloned_vecs[i]-torch.floor(cloned_vecs[i]), generator=self.prng)   
                                    
        return cloned_vecs, min_coordinate, delta
   
    def roundtrip(self, client_grad_vecs): 
        
        if self.norm_normalization:
            sq_func = self.norm_stochastic_quantization
        else:
            sq_func = self.min_max_stochastic_quantization
                    
        nclients = len(client_grad_vecs)
        
        if self.rotation:
            padded_dim = int(2**(np.ceil(np.log2(self.d))))
            res = torch.zeros(size=(padded_dim,), device=self.device)
        else:
            res = torch.zeros(size=(self.d,), device=self.device)
                           
        if self.ef:
            
            for i in range(nclients):
                self.errors[i] = client_grad_vecs[i] + self.errors.get(i,0)
            
            if self.rotation:    
                client_grad_vecs, min_coordinate, delta = sq_func([self.hadamard.rht(self.errors[i]) for i in sorted(self.errors)])
            else:
                client_grad_vecs, min_coordinate, delta = sq_func([self.errors[i] for i in sorted(self.errors)])
            
            for i in range(nclients):
                
                if self.rotation:  
                    self.errors[i] -= self.hadamard.irht(min_coordinate + client_grad_vecs[i] * delta) 
                else:
                    self.errors[i] -= (min_coordinate + client_grad_vecs[i] * delta) 
                    
                res += client_grad_vecs[i] ### switch aggregation
                                        
        else:
            
            if self.rotation: 
                client_grad_vecs, min_coordinate, delta = sq_func([self.hadamard.rht(cgv) for cgv in client_grad_vecs])
            else:
                client_grad_vecs, min_coordinate, delta = sq_func(client_grad_vecs)
                
            for i in range(nclients):
                res += client_grad_vecs[i] ### switch aggregation
                
        res =  min_coordinate + (res / nclients) * delta  ### parallel at the workers 

        if self.rotation:
            res = self.hadamard.irht(res)
                    
        return res        
        
##############################################################################
##############################################################################

import pathlib
path = pathlib.Path(__file__).parent.resolve()

import sys
sys.path.insert(0, str(path) + "/../")

class INCA:

    def __init__(self, params):
        
        self.device = params['device']  

        self.d = params['d']
        self.seed = params['seed']
        
        self.prng = torch.Generator(device=params['device'])
        self.prng.manual_seed(params['seed'])
                        
        self.ef = params['ef']
        
        self.quantization_levels = params['quantization_levels']
        self.overflow_frequency = params['overflow_frequency']
        self.smaxval = params['max_val'] 
        
        self.tablesize = params['tablesize']
                
        ######################################################################
        
        self.hadamard = Hadamard(self.d, self.seed, self.device)
        
        ######################################################################

        if self.ef:
            self.errors = {}
        
        self.fn_prefix = []

        self.fn_prefix.append(str(path))
        self.fn_prefix.append('new_tables')
        self.fn_prefix.append('{}_tablesize_{}_maxval_{}_qlevels_{}_ofreq_'.format(self.tablesize,
                                                                                   self.smaxval,
                                                                                   self.quantization_levels, 
                                                                                   self.overflow_frequency * 2))
        
        fn = "/".join(self.fn_prefix)
        
        ### sender ###########################################################      
        self.sender_prng = torch.Generator(device=self.device)
        self.sender_table_X, self.sender_table_p, self.data = self.sender_table(fn, self.device)
        self.half_table_size = (self.sender_table_X.numel() - 1) // 2
        
        ### receiver #########################################################
        self.receiver_prng = torch.Generator(device=self.device)
        self.recv_table = self.receiver_table(fn, self.device)
        
    ##########################################################################
       
    def sender_table(self, prefix, device):
    
        sender_table_X = torch.load(prefix + 'sender_table_X.pt').to(device)
        sender_table_p = torch.load(prefix + 'sender_table_p.pt').to(device)
        
        data = eval(open(prefix + 'data.txt').read())
    
        return sender_table_X, sender_table_p, data
            
    ##########################################################################
        
    def receiver_table(self, prefix, device):
        
        recv_table = torch.load(prefix +'recv_table.pt').to(device)
        
        return recv_table

    ##########################################################################

    def rvec_roundtip(self, vecs):
        # pdb.set_trace()
        cloned_vecs = [vec.clone() for vec in vecs]
        
        # this happens at the preliminary round where all workers share their norm and learn the max
        dim = cloned_vecs[0].numel() ### might be padded -> self.d might be wrong
        max_norm = max([vec.norm(2) for vec in cloned_vecs])
        
        max_coordinate = self.data['T']
        min_coordinate = -max_coordinate
        
        for i in range(len(cloned_vecs)):

            ######
            # worker logic
            
            scale = np.sqrt(dim) / max_norm
            
            cloned_vecs[i] *= scale
            
            overflow_p = ((cloned_vecs[i] > max_coordinate).sum() + (cloned_vecs[i] < min_coordinate).sum()) / float(dim)
            if not self.ef and overflow_p > 0:
                warnings.warn('quantization overflow with no error feedback detected: {}% overflow'.format(overflow_p))

            
            cloned_vecs[i] = torch.clamp(cloned_vecs[i], min=min_coordinate, max=max_coordinate)
            
            cloned_vecs[i] /= self.data['delta']
            
            p = cloned_vecs[i] - cloned_vecs[i].floor()
            cloned_vecs[i] = cloned_vecs[i].floor() + torch.bernoulli(p, generator=self.sender_prng)
            
            X = torch.take(self.sender_table_X, (cloned_vecs[i] + self.half_table_size).long())
            p_X = torch.take(self.sender_table_p, (cloned_vecs[i] + self.half_table_size).long())
            
            X += torch.bernoulli(p_X).int()
        
            ######
            # switch lookup

            cloned_vecs[i] = torch.take(self.recv_table, X.long())
        
        return cloned_vecs, scale
       
    ##########################################################################

    def roundtrip(self, client_grad_vecs):
        nclients = len(client_grad_vecs)

        if self.ef:
                        
            for i in range(nclients):
                self.errors[i] = client_grad_vecs[i] + self.errors.get(i,0)

            client_grad_vecs, scale = self.rvec_roundtip([self.hadamard.rht(self.errors[i]) for i in sorted(self.errors)])
            
            for i in range(nclients):
                temp = client_grad_vecs[i].float() - self.smaxval / 2 
                self.errors[i] -= self.hadamard.irht(temp / scale * self.data['inc'])
                
        else:
            
            client_grad_vecs, scale = self.rvec_roundtip([self.hadamard.rht(cgv) for cgv in client_grad_vecs])

        res = sum(client_grad_vecs) ### switch aggregation
        
        ### parallel at the workers
        res = res.float()
        res /=  nclients

        res = res - self.smaxval / 2
        res = res / scale * self.data['inc']

        res = self.hadamard.irht(res)
                           
        return res[:]        
             
##############################################################################
##############################################################################




"""
Explanations of the variables.
T: the threshold t_p of the normal distribution such that P(X <= -T || X >= T) = freq
quantization_levels: b bits after applying the lookup table
maxval: g. The precision after applying the inv-lookup table at PS.
inc: maxval = 1/inc * T
"""


class INCA_Ring(object):
    def __init__(self, params):
        self.device = params['device']
        self.nclients = params["nclients"]
        self.option = params["option"]

        self.d = params['d']
        self.seed = params['seed']
        
        self.prng = torch.Generator(device=params['device'])
        self.prng.manual_seed(params['seed'])
                        
        self.ef = params['ef']
        
        self.quantization_levels = params['quantization_levels']
        self.overflow_frequency = params['overflow_frequency']
        self.smaxval = params['max_val'] 
        
        self.tablesize = params['tablesize']
                
        ######################################################################
        
        self.hadamard = Hadamard(self.d, self.seed, self.device)
        
        ######################################################################

        if self.ef:
            self.errors = {}
        
        def gen_prefix_fn(extra_bits):
            fn_prefix = []
            fn_prefix.append(str(path))
            fn_prefix.append('new_tables')
            if (self.quantization_levels << extra_bits) == 4:
                fn_prefix.append('{}_tablesize_{}_maxval_{}_qlevels_{}_ofreq_'.format(self.tablesize,
                                                                                    34,
                                                                                    self.quantization_levels << extra_bits, 
                                                                                    self.overflow_frequency * 2))
            else:
                fn_prefix.append('{}_tablesize_{}_maxval_{}_qlevels_{}_ofreq_'.format(self.tablesize,
                                                                                    self.smaxval[extra_bits],
                                                                                    self.quantization_levels << extra_bits, 
                                                                                    self.overflow_frequency * 2))
            
            fn = "/".join(fn_prefix)
            return fn

        
        ### sender ###########################################################      
        self.sender_prng = torch.Generator(device=self.device)
        self.sender_tables_X, self.sender_tables_p, self.datas = self.sender_table(gen_prefix_fn, self.device)
        self.half_table_size = (self.sender_tables_X[0].numel() - 1) // 2
        
        ### receiver #########################################################
        self.receiver_prng = torch.Generator(device=self.device)
        self.recv_tables = self.receiver_table(gen_prefix_fn, self.device)
        
    ##########################################################################
       
    def sender_table(self, gen_prefix_fn, device):
        sender_tables_X = []
        sender_tables_p = []
        datas = []

        bound = 1 if self.nclients <= 2 else math.ceil(math.log2(self.nclients - 1) / 2) + 1

        for i in range(bound):
            if self.option == "fixed":
                prefix = gen_prefix_fn(0)# (i)
            else:
                prefix = gen_prefix_fn(i)
            sender_tables_X.append(torch.load(prefix + 'sender_table_X.pt').to(device))
            sender_tables_p.append(torch.load(prefix + 'sender_table_p.pt').to(device))
        
            datas.append(eval(open(prefix + 'data.txt').read()))
    
        return sender_tables_X, sender_tables_p, datas
            
    ##########################################################################
        
    def receiver_table(self, gen_prefix_fn, device):
        recv_tables = []
        bound = 1 if self.nclients <= 2 else math.ceil(math.log2(self.nclients - 1) / 2) + 1

        for i in range(bound):
            if self.option == "fixed":
                prefix = gen_prefix_fn(0)# (i)
            else:
                prefix = gen_prefix_fn(i)
            recv_tables.append(torch.load(prefix +'recv_table.pt').to(device))
        
        return recv_tables

    ##########################################################################

    def rvec_roundtip(self, vecs):
        # pdb.set_trace()
        cloned_vecs = [vec.clone() for vec in vecs]
        vecs_wo_lookup = []
        
        # this happens at the preliminary round where all workers share their norm and learn the max
        dim = cloned_vecs[0].numel() ### might be padded -> self.d might be wrong
        max_norm = max([vec.norm(2) for vec in cloned_vecs])

        sum_vec = torch.zeros(vecs[0].shape).to(self.device)
        
        for i in range(len(cloned_vecs)):
            ######
            # worker logic
            lookup_index = math.ceil(math.log2(i + 1) / 2)
            if i < len(cloned_vecs) - 1:
                max_coordinate = self.datas[lookup_index]['T']
            else:
                max_coordinate = self.datas[-1]['T']
            min_coordinate = -max_coordinate
            scale = np.sqrt(dim) / max_norm
            
            cloned_vecs[i] *= scale
            
            overflow_p = ((cloned_vecs[i] > max_coordinate).sum() + (cloned_vecs[i] < min_coordinate).sum()) / float(dim)
            if not self.ef and overflow_p > 0:
                warnings.warn('quantization overflow with no error feedback detected: {}% overflow'.format(overflow_p))

            
            cloned_vecs[i] = torch.clamp(cloned_vecs[i], min=min_coordinate, max=max_coordinate)
            vecs_wo_lookup.append(cloned_vecs[i].clone())

            sum_vec += cloned_vecs[i]
            if i < len(cloned_vecs) - 1:
            
                sum_vec /= self.datas[lookup_index]['delta'] * (i + 1)
                sum_vec = torch.clamp(sum_vec, min=-(len(self.sender_tables_X[lookup_index]) - 1) / 2, max=(len(self.sender_tables_X[lookup_index]) - 1) / 2)
                
                p = sum_vec - sum_vec.floor()
                sum_vec = sum_vec.floor() + torch.bernoulli(p, generator=self.sender_prng)

                
                X = torch.take(self.sender_tables_X[lookup_index], (sum_vec + self.half_table_size).long())
                p_X = torch.take(self.sender_tables_p[lookup_index], (sum_vec + self.half_table_size).long())
                
                X += torch.bernoulli(p_X).int()
            
                ######
                # switch lookup

                sum_vec = torch.take(self.recv_tables[lookup_index], X.long())
                sum_vec = (sum_vec - self.smaxval[lookup_index] / 2) * self.datas[lookup_index]["inc"] * (i + 1)
            
            else:
                sum_vec /= self.datas[-1]["inc"] / (i + 1)
                p = sum_vec - sum_vec.floor()
                sum_vec = sum_vec.floor() + torch.bernoulli(p, generator=self.sender_prng)
                sum_vec *= self.datas[-1]["inc"] / (i + 1)
        
        return vecs_wo_lookup, sum_vec, scale
       
    ##########################################################################

    def roundtrip(self, client_grad_vecs):
        nclients = len(client_grad_vecs)
        # pdb.set_trace()
        if self.ef:
                        
            for i in range(nclients):
                self.errors[i] = client_grad_vecs[i] + self.errors.get(i,0)

            vecs_wo_lookup, sum_client_grad_vecs, scale = self.rvec_roundtip([self.hadamard.rht(self.errors[i]) for i in sorted(self.errors)])
            
            for i in range(nclients):
                temp = vecs_wo_lookup[i].float()
                self.errors[i] -= self.hadamard.irht(temp / scale)
                
        else:
            vecs_wo_lookup, sum_client_grad_vecs, scale = self.rvec_roundtip([self.hadamard.rht(cgv) for cgv in client_grad_vecs])

        # res = sum(client_grad_vecs) ### switch aggregation
        res = sum_client_grad_vecs
        ### parallel at the workers
        res = res.float()
        res /=  nclients

        res = res
        res = res / scale

        res = self.hadamard.irht(res)
                           
        return res[:]        
             
##############################################################################
##############################################################################

