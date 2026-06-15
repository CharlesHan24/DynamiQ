import torch
import numpy as np
from scipy.stats import truncnorm
import scipy

np.random.seed(42)

# Xs = [2, 3, 4]
# invps = [2**i for i in range(4,11)]

Xs = [2]
invps = [1024]

seen = set()

for Xval in Xs:
    #print(Xval)
    X_range = 2**(Xval-1) - 1
    for invp in invps:
        print(Xval,invp)
        lines = open('optimal_INCA_'+str(Xval)+'_X_bits_invp_'+str(invp)+'.txt').readlines()
        print('optimal_INCA_'+str(Xval)+'_X_bits_invp_'+str(invp)+'.txt')
        opt_dict = eval(lines[0])
        for maxval in opt_dict:
            if (maxval <= 2**(Xval-1)):
                continue
            # import pdb
            # pdb.set_trace()
            #recv_table = [-x+opt_dict[maxval][1][-1] for x in reversed(opt_dict[maxval][1]) if x>0] + [x+opt_dict[maxval][1][-1] for x in opt_dict[maxval][1]]
            recv_table = [-x+opt_dict[maxval][1][-1] for x in reversed(opt_dict[maxval][1])] + [x+opt_dict[maxval][1][-1] for x in opt_dict[maxval][1]]
            print(maxval,recv_table, len(recv_table))
            if ((Xval, invp, recv_table[-1]) in seen):
                continue
            else:
                seen.add((Xval, invp, recv_table[-1]))
            T = opt_dict[maxval][-1][-1]


            points = 10001  
            start = -T
            stop = T

            t = np.linspace(start, stop, num=points)

            sender_msg_matrix = []      #t -> (X \in {0,2*X_range-1}, p\in [0,1])

            Est_array = opt_dict[maxval][1]
            #print('Est_array = ', Est_array)
            x = T / Est_array[-1]
            delta = t[1] - t[0]

            data = {'delta': delta, 'T':T, 'inc': x}            
            def encode_coordinate(ti):
                p = -1
                if (ti == T):
                    return (2*X_range,1)
                if (ti < 0):
                    Xm,pm = encode_coordinate(-ti)
                    return 2*X_range-Xm, 1-pm
                for X in range(X_range+1):
                    if Est_array[X]*x > ti:
                        if (X > 0):
                            p = (ti - Est_array[X-1]*x) / (Est_array[X]*x - Est_array[X-1]*x)
                        else:
                            p = (ti - (-Est_array[0])*x) / (2*Est_array[0]*x)
                        break    
                #print(ti, X+X_range, Est_array[X], Est_array[X]*x, p)
                #return X+X_range - (0 in Est_array),p
                return X+X_range,p

            for i,ti in enumerate(t):
                X, p = encode_coordinate(ti)
                #print (ti, X, p)
                sender_msg_matrix.append((X,p))            
            #print(sender_msg_matrix[-1])
            #print('X_range',X_range)
            #exit()            
            
            #print(maxval,opt_dict[maxval], recv_table)
            prefix = '{}_tablesize_{}_maxval_{}_qlevels_{}_ofreq_'.format(points,recv_table[-1], 2**Xval, invp)

            sender_table_X,sender_table_p = zip(*sender_msg_matrix)
            #print(sender_table_X)
            #print(sender_table_p)
            #print(recv_table)  

            sender_table_X = torch.Tensor(sender_table_X).int()
            sender_table_p = torch.Tensor(sender_table_p)
            recv_table = torch.Tensor(recv_table).int()

            np.savetxt(prefix + 'sender_table_X.txt', sender_table_X.numpy())
            np.savetxt(prefix + 'sender_table_p.txt', sender_table_p.numpy())
            np.savetxt(prefix + 'recv_table.txt', recv_table.numpy())
            open(prefix + 'data.txt','w').write(str(data))

            torch.save(sender_table_X, prefix + 'sender_table_X.pt')
            torch.save(sender_table_p, prefix + 'sender_table_p.pt')
            torch.save(recv_table, prefix + 'recv_table.pt')
            #print(sender_table_p)
exit()


lines =  open(str(maxval) + '_max_int_val_4_X_bits_0_h_bits_256_quantiles_solver_1_win32_invp_512.txt').readlines()



Est_array = {}
for line in lines:
    if 'Est_array' in line:
        Est_array[int(line.split('[')[1].split(']')[0].split(',')[0])] = float(line.split('=')[-1])
    if 'x = ' in line:
        x = float(line.split('=')[-1])



X_range = max(Est_array)

T = x * Est_array[X_range]

print (Est_array, x, T)

points = 10001  
start = -T
stop = T

t = np.linspace(start, stop, num=points)

sender_msg_matrix = []      #t -> (X \in {0,2*X_range-1}, p\in [0,1])
receiver_est_matrix = {}    #X -> t_hat

#print(X_range)

print(Est_array)
def encode_coordinate(t):
    p = -1
    if (t < 0):
        Xm,pm = encode_coordinate(-t)
        return 2*X_range-Xm, 1-pm
    for X in range(X_range+1):
        if Est_array[X]*x >= t:
            if (X > 0):
                p = (t - Est_array[X-1]*x) / (Est_array[X]*x - Est_array[X-1]*x)
            else:
                p = (t - (-Est_array[0])*x) / (2*Est_array[0]*x)
            break    
    #print(t, X+X_range, Est_array[X], Est_array[X]*x, p)
    return X+X_range,p

for i,ti in enumerate(t):
    X, p = encode_coordinate(ti)
    sender_msg_matrix.append((X,p))
    
recv_table = []
h_range = 0
for X in range(X_range+1):
    recv_table.append(int(Est_array[X_range]-Est_array[X_range-X]))
for X in range(X_range+1):
    recv_table.append(int(Est_array[X_range]+Est_array[X]))
  

delta = t[1] - t[0]

data = {'delta': delta, 'T':T, 'inc': x}



prefix = '{}_maxval_{}_qlevels_{}_quantiles_512_ofreq_'.format(2*maxval, 2*(X_range+1), 256)

sender_table_X,sender_table_p = zip(*sender_msg_matrix)
#print(sender_table_X)
#print(sender_table_p)
print(recv_table)  

sender_table_X = torch.Tensor(sender_table_X).int()
sender_table_p = torch.Tensor(sender_table_p)
recv_table = torch.Tensor(recv_table).int()

np.savetxt(prefix + 'sender_table_X.txt', sender_table_X.numpy())
np.savetxt(prefix + 'sender_table_p.txt', sender_table_p.numpy())
np.savetxt(prefix + 'recv_table.txt', recv_table.numpy())
open(prefix + 'data.txt','w').write(str(data))

torch.save(sender_table_X, prefix + 'sender_table_X.pt')
torch.save(sender_table_p, prefix + 'sender_table_p.pt')
torch.save(recv_table, prefix + 'recv_table.pt')

exit(0)
    
def receiver_table(fn='temp'):
    
    recv_table = torch.Tensor(eval(open('recv_table').read()))
    
    return recv_table
    
sender_table_X, sender_table_p, data = sender_table()
#recv_table = receiver_table()

recv_table = torch.Tensor(recv_table)    


    

exit(0)


for h_file in range(0,1):
    for p_idx, p in enumerate([2**-i for i in range(6,7)]):
        try:
            lines = open('1_X_bits_'+str(h_file)+'_h_bits_256_quantiles_solver_1_win32_invp_'+str(int(1/p))+'.txt').readlines()
        except:
            lines = open('1_X_bits_'+str(h_file)+'_h_bits_256_quantiles_solver_3_linux_invp_'+str(int(1/p))+'.txt').readlines()
        Est_array = {}
        for line in lines:
            if 'Est_array' in line:
                Est_array[int(line.split('[')[1].split(']')[0].split(',')[0]),int(line.split('[')[1].split(']')[0].split(',')[1])] = float(line.split('=')[-1])




        prefix = '1_X_'+str(h_file)+'_h_256_q_invp_'+str(int(1/p))
        #print(prefix,Est_array)
        #continue




        h_range = max([y for (x,y) in Est_array])
        X_range = max([x for (x,y) in Est_array])

        T = sum([Est_array[X_range,h] for h in range(h_range+1)])/(h_range+1)

        points = 10001
        start = -T
        stop = T

        t = np.linspace(start, stop, num=points)

        sender_msg_matrix = {}      #(t, h) -> (X \in {-1,...,X_range-1}, p\in [0,1])
        receiver_est_matrix = {}    #(X, h) -> t_hat

        def calculate_stochastic_h_ranges():
            if (h_range > 1):
                stochastic_h_ranges = []
                for stochastic_h in range(h_range//2,h_range):
                    distance = stochastic_h - h_range//2
                    min_t = sum([Est_array[0,h] for h in range(h_range//2-distance+1, h_range//2+distance+1)]) / (h_range+1)
                    stochastic_h_ranges.append((min_t,X_range+1,stochastic_h))
                stochastic_h_ranges.append((sum([Est_array[0,h] for h in range(h_range+1)]) / (h_range+1),X_range+1,h_range))
                for X in range(1,X_range+1):
                    for stochastic_h in range(1,h_range+2):
                        min_t = (sum([Est_array[X-1,h] for h in range(stochastic_h, h_range+1)]) + sum([Est_array[X,h] for h in range(stochastic_h)])) / (h_range+1)
                        stochastic_h_ranges.append((min_t,X_range+X+1,stochastic_h-1))
                #stochastic_h_ranges.append((T,2*X_range+1,h_range))
            else:
                stochastic_h_ranges = [(-Est_array[0,0],X_range,0)]
                for stochastic_h in range(h_range//2,h_range):
                    distance = stochastic_h - h_range//2
                    min_t = sum([Est_array[0,h] for h in range(h_range//2-distance+1, h_range//2+distance+1)]) / (h_range+1)
                    stochastic_h_ranges.append((min_t,X_range+1,stochastic_h))
                stochastic_h_ranges.append((sum([Est_array[0,h] for h in range(h_range+1)]) / (h_range+1),X_range+1,h_range))
                for X in range(1,X_range+1):
                    for stochastic_h in range(1,h_range+2):
                        min_t = (sum([Est_array[X-1,h] for h in range(stochastic_h, h_range+1)]) + sum([Est_array[X,h] for h in range(stochastic_h)])) / (h_range+1)
                        stochastic_h_ranges.append((min_t,X_range+X+1,stochastic_h-1))
                #stochastic_h_ranges.append((T,2*X_range+1,h_range))
            return stochastic_h_ranges

           
            
        def encode_coordinate(ti,hi,stochastic_h_ranges):
            if ti < 0 and X_range + h_range > 0:
                minus_ti_X, minus_ti_p = encode_coordinate(-ti,h_range-hi,stochastic_h_ranges)
                if (minus_ti_p == 0):
                    return 2*X_range-minus_ti_X+1,0
                return 2*X_range-minus_ti_X, 1-minus_ti_p
            if (X_range == 0) and (h_range == 0):
                range_min = -Est_array[0,h_range-hi]
                range_max = Est_array[0,hi]
                X = 1
            else:        
                for i in range(len(stochastic_h_ranges)):
                    if (ti < stochastic_h_ranges[i][0]):
                        break
                threshold, X, stochastic_h = stochastic_h_ranges[i]
                if hi != stochastic_h:
                    if hi<stochastic_h:
                        return X, 0 
                    else:
                        return X-1, 0
                range_min = stochastic_h_ranges[i-1][0]
                range_max = threshold
               # print(ti, i, threshold, X, stochastic_h)
            
            p = (ti - range_min) / (range_max - range_min)
            #print(ti, hi, range_min, range_max, p, X_range, h_range, X_range, X)
            #print(stochastic_h_ranges)
            #exit()
            assert(p >= -0.000000001 and p < 1.000000001)
            p = min(p, 1)
            p = max(p, 0)
            #exit()
            if (p == 1):
                return X, 0
            return (X-1), p

        est_t = np.zeros([points,])
        stochastic_h_ranges = calculate_stochastic_h_ranges()

        encode_coordinate(0,0,stochastic_h_ranges)

        for i,ti in enumerate(t):
            for h in range(h_range+1):
                sender_msg_matrix[i, h] = encode_coordinate(ti,h,stochastic_h_ranges)
               

        outfile = open('temp', 'w')

        output = []
        for i,ti in enumerate(t):
            output.append([])
            for h in range(h_range+1):
                #print(ti,h, sender_msg_matrix[i, h])
                output[-1].append((ti,sender_msg_matrix[i, h]))
               
        outfile.write(str(output))
        outfile.close()

        #read_list = eval(open('temp').read())
        #print(read_list)

        recv_table = []
        for X in range(X_range+1):
            recv_table.append([])
            for h in range(h_range+1):
                recv_table[-1].append(-Est_array[X_range-X,h_range-h])
        for X in range(X_range+1):
            recv_table.append([])
            for h in range(h_range+1):
                recv_table[-1].append(Est_array[X,h])
        print(recv_table)



        def sender_table(fn='temp'):

            read_list = eval(open(fn).read())
            
            T = read_list[-1][0][0]
            
            h_len = len(read_list[0])
            
            x_len =  len(read_list)
            delta = read_list[1][0][0] - read_list[0][0][0]
            
            sender_table_X = torch.zeros((x_len, h_len+1))
            sender_table_p = torch.zeros((x_len, h_len+1))
            
            for i in range(x_len):
                qi = read_list[i][0][0]
                sender_table_X[i,0] = qi
                sender_table_p[i,0] = qi
                for j in range(h_len):
                    sender_table_X[i,j+1] = read_list[i][j][1][0]
                    sender_table_p[i,j+1] = read_list[i][j][1][1]
                    if read_list[i][j][1][1] < 0:
                        print(read_list[i])
            
            return sender_table_X, sender_table_p, {'delta': delta, 'T':T, 'h_len': h_len, 'x_len': x_len}
                

        def receiver_table(fn='temp'):
            
            recv_table = torch.Tensor(eval(open('recv_table').read()))
            
            return recv_table
            
        sender_table_X, sender_table_p, data = sender_table()
        #recv_table = receiver_table()

        recv_table = torch.Tensor(recv_table)



        np.savetxt(prefix + 'sender_table_X.txt', sender_table_X.numpy())
        np.savetxt(prefix + 'sender_table_p.txt', sender_table_p.numpy())
        np.savetxt(prefix + 'recv_table.txt', recv_table.numpy())
        #open(prefix + 'recv_table.txt', 'w').write(str(recv_table))
        open(prefix + 'data.txt','w').write(str(data))

        torch.save(sender_table_X[:,1:], prefix + 'sender_table_X.pt')
        torch.save(sender_table_p[:,1:], prefix + 'sender_table_p.pt')
        torch.save(recv_table, prefix + 'recv_table.pt')

        #sender_table_X_read = torch.load(prefix + 'sender_table_X.pt')
        #print(sender_table_X_read)
