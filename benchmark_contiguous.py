import time
import torch
torch.set_num_threads(1)
from clebsch_gordan import get_real_clebsch_gordan, ClebschGordan
from sparse_accumulation_plain_torch import sparse_accumulation_loops, sparse_accumulation_index_add
import sparse_accumulation, sparse_accumulation_active_dim_first,  sparse_accumulation_active_dim_middle
import numpy as np
from torch.utils import cpp_extension

cpp_extension.load(
    name="sparse_accumulation_cuda",
    sources=["cuda/sparse_accumulation_cuda_kernel2D.cu"],
    is_python_module=False,
    extra_cuda_cflags=None,
    verbose=True,
)

L_MAX = 8
BATCH_SIZE = 1000
N_FEATURES = 200
print(F"{L_MAX=}; {BATCH_SIZE=}; {N_FEATURES=}")
print("preparing real life transformation rule")

clebsch = ClebschGordan(L_MAX).precomputed_
indices = get_real_clebsch_gordan(clebsch[L_MAX, L_MAX, L_MAX], L_MAX, L_MAX, L_MAX)

m1_aligned, m2_aligned = [], []
multipliers, mu_aligned = [], []
for mu in range(0, 2 * L_MAX + 1):
    for el in indices[mu]:
        m1, m2, multiplier = el
        m1_aligned.append(m1)
        m2_aligned.append(m2)
        multipliers.append(multiplier)
        mu_aligned.append(mu)
m1_aligned = torch.LongTensor(m1_aligned)
m2_aligned = torch.LongTensor(m2_aligned)
mu_aligned = torch.LongTensor(mu_aligned)
multipliers = torch.FloatTensor(multipliers)
print("transformation rule is computed")

def get_input(BATCH_SIZE, N_FEATURES, active_dim, device):
    if active_dim == 0:
        X1 = torch.randn(2 * L_MAX + 1, BATCH_SIZE, N_FEATURES, device = device)
        X2 = torch.randn(2 * L_MAX + 1, BATCH_SIZE, N_FEATURES, device = device)
    
    if active_dim == 1:
        X1 = torch.randn(BATCH_SIZE, 2 * L_MAX + 1, N_FEATURES, device = device)
        X2 = torch.randn(BATCH_SIZE, 2 * L_MAX + 1, N_FEATURES, device = device)
        
    if active_dim == 2:
        X1 = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1, device = device)
        X2 = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1, device = device)   
   
        
    if (active_dim != 0) and (active_dim != 2) and (active_dim != 1):
        raise ValueError("active dim should be one of 0, 1, 2")
        
    return X1, X2

def benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, active_dim, function, n_trials):
    X1, X2 = get_input(BATCH_SIZE, N_FEATURES, active_dim, 'cpu')
    times = []
            
    for _ in range(n_trials):
        begin = time.time() 
        output = function(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
        times.append(time.time() - begin)
    return times


def benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, active_dim, function, n_trials):
    X1, X2 = get_input(BATCH_SIZE, N_FEATURES, active_dim, 'cuda')
    times = []
    torch.cuda.synchronize('cuda')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)
    
    for _ in range(n_trials):
        starter.record()
        output = function(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
        ender.record()
        torch.cuda.synchronize('cuda')
        delta_time = starter.elapsed_time(ender)
        times.append(delta_time / 1000.0)        
    return times


def benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, active_dim, function, n_trials):
    X1, X2 = get_input(BATCH_SIZE, N_FEATURES, active_dim, 'cpu')
        
    X1.requires_grad = True
    X2.requires_grad = True
    times = []
    for _ in range(n_trials):
        begin = time.time()
        output = function(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
        output.backward(gradient=torch.ones_like(output))
        times.append(time.time() - begin)
    return np.array(times)

def benchmark_backward_gpu(BATCH_SIZE, N_FEATURES, active_dim, function, n_trials):
    X1, X2 = get_input(BATCH_SIZE, N_FEATURES, active_dim, 'cuda')
        
    X1.requires_grad = True
    X2.requires_grad = True
    times = []
    
    torch.cuda.synchronize('cuda')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)    
    
    for _ in range(n_trials):
        starter.record()
        output = function(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
        output.backward(gradient=torch.ones_like(output))
        ender.record()
        torch.cuda.synchronize('cuda')
        delta_time = starter.elapsed_time(ender)
        times.append(delta_time / 1000.0)  
    return np.array(times)


def benchmark_backward_gpu_cuda(BATCH_SIZE, N_FEATURES, active_dim, function, n_trials):
    X1, X2 = get_input(BATCH_SIZE, N_FEATURES, active_dim, 'cuda')
        
    X1.requires_grad = True
    X2.requires_grad = True
    times = []
    
    torch.cuda.synchronize('cuda')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)    
    
    for _ in range(n_trials):
        starter.record()
        output = function(torch.ones((X1.size()[0], X1.size()[1], 2 * L_MAX + 1),dtype=X1.dtype,device='cuda'),X1, X2, mu_aligned,  m1_aligned, m2_aligned, multipliers)
        #output.backward(gradient=torch.ones_like(output))
        ender.record()
        torch.cuda.synchronize('cuda')
        delta_time = starter.elapsed_time(ender)
        times.append(delta_time / 1000.0)  
    return np.array(times)


# In[5]:


def get_func_fixed_dim(func, active_dim):
    def func_fixed_dim(*args):
        return func(*args, active_dim = active_dim)
    return func_fixed_dim


print("*************")
print("CUDA")
print("*************")

m1_aligned = m1_aligned.cuda() 
m2_aligned = m2_aligned.cuda()
mu_aligned = mu_aligned.cuda()
multipliers = multipliers.cuda()

print("***forward***")
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 0, 
                          get_func_fixed_dim(sparse_accumulation_loops, 0), 10)
print("python loops; active dim 0; forward; cuda: ", np.mean(times[1:]))
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 0, 
                          get_func_fixed_dim(sparse_accumulation_index_add, 0), 10)
print("torch index_add_; active dim 0; forward; cuda: ", np.mean(times[1:]))
'''times = benchmark_forward(BATCH_SIZE, N_FEATURES, 0,
                          sparse_accumulation_active_dim_first.SparseAccumulationActiveDimFirst.apply, 10)
print("cpp; active dim 0; forward: ", np.mean(times[1:]))'''

print()
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 1, 
                          get_func_fixed_dim(sparse_accumulation_loops, 1), 10)
print("python loops; active dim 1; forward; cuda: ", np.mean(times[1:]))
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 1, 
                          get_func_fixed_dim(sparse_accumulation_index_add, 1), 10)
print("torch index_add_; active dim 1; forward; cuda: ", np.mean(times[1:]))
'''times = benchmark_forward(BATCH_SIZE, N_FEATURES, 1,
                          sparse_accumulation_active_dim_middle.SparseAccumulationActiveDimMiddle.apply, 10)
print("cpp; active dim 1; forward: ", np.mean(times[1:]))'''


print()
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 2, 
                          get_func_fixed_dim(sparse_accumulation_loops, 2), 1)
print("python loops first; active dim 2; forward; cuda: ", np.mean(times))
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 2,
                          get_func_fixed_dim(sparse_accumulation_index_add, 2), 1)
print("torch index_add_ first; active dim 2; forward; cuda: ", np.mean(times))
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 2, 
                          torch.ops.sparse_accumulation_cuda.forward, 1)
print("CUDA kernel first; active dim 2; forward; cuda: ", np.mean(times))
'''times = benchmark_forward(BATCH_SIZE, N_FEATURES, 2, sparse_accumulation.SparseAccumulation.apply, 10)
print("cpp; active dim 2; forward: ", np.mean(times[1:]))'''

print()
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 2, 
                          get_func_fixed_dim(sparse_accumulation_loops, 2), 10)
print("python loops; active dim 2; forward; cuda: ", np.mean(times[1:]))
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 2,
                          get_func_fixed_dim(sparse_accumulation_index_add, 2), 10)
print("torch index_add_; active dim 2; forward; cuda: ", np.mean(times[1:]))
times = benchmark_forward_gpu(BATCH_SIZE, N_FEATURES, 2, 
                          torch.ops.sparse_accumulation_cuda.forward, 10)
print("CUDA kernel; active dim 2; forward; cuda: ", np.mean(times[1:]))
'''times = benchmark_forward(BATCH_SIZE, N_FEATURES, 2, sparse_accumulation.SparseAccumulation.apply, 10)
print("cpp; active dim 2; forward: ", np.mean(times[1:]))'''

# In[10]:

print("***backward***")
times = benchmark_backward_gpu(BATCH_SIZE, N_FEATURES, 0, 
                           get_func_fixed_dim(sparse_accumulation_loops, 0), 10)
print("python loops; active dim 0; backward; cuda: ", np.mean(times[1:]))
times = benchmark_backward_gpu(BATCH_SIZE, N_FEATURES, 0, 
                           get_func_fixed_dim(sparse_accumulation_index_add, 0), 10)
print("torch index_add_; active dim 0; backward; cuda: ", np.mean(times[1:]))
'''times = benchmark_backward(BATCH_SIZE, N_FEATURES, 0,
                           sparse_accumulation_active_dim_first.SparseAccumulationActiveDimFirst.apply, 10)
print("cpp; active dim 0; backward: ", np.mean(times[1:]))'''

print()

times = benchmark_backward_gpu(BATCH_SIZE, N_FEATURES, 1, 
                           get_func_fixed_dim(sparse_accumulation_loops, 1), 10)
print("python loops; active dim 1; backward; cuda: ", np.mean(times[1:]))
times = benchmark_backward_gpu(BATCH_SIZE, N_FEATURES, 1, 
                           get_func_fixed_dim(sparse_accumulation_index_add, 1), 10)
print("torch index_add_; active dim 1; backward; cuda: ", np.mean(times[1:]))
'''times = benchmark_backward(BATCH_SIZE, N_FEATURES, 1,
                           sparse_accumulation_active_dim_middle.SparseAccumulationActiveDimMiddle.apply, 10)
print("cpp; active dim 1; backward: ", np.mean(times[1:]))'''


print()
times = benchmark_backward_gpu(BATCH_SIZE, N_FEATURES, 2, 
                           get_func_fixed_dim(sparse_accumulation_index_add, 2), 10)
print("python loops; active dim 2; backward; cuda: ", np.mean(times[1:]))
times = benchmark_backward_gpu(BATCH_SIZE, N_FEATURES, 2, 
                           get_func_fixed_dim(sparse_accumulation_index_add, 2), 10)
print("torch index_add_; active dim 2; backward; cuda: ", np.mean(times[1:]))
times = benchmark_backward_gpu_cuda(BATCH_SIZE, N_FEATURES, 2, 
                          torch.ops.sparse_accumulation_cuda.backward, 10)
print("CUDA kernel; active dim 2; backward; cuda: ", np.mean(times[1:]))
'''times = benchmark_backward(BATCH_SIZE, N_FEATURES, 2, sparse_accumulation.SparseAccumulation.apply, 10)
print("cpp; active dim 2; backward: ", np.mean(times[1:]))'''


print("*************")
print("CPU")
print("*************")

m1_aligned = m1_aligned.cpu() 
m2_aligned = m2_aligned.cpu()
mu_aligned = mu_aligned.cpu()
multipliers = multipliers.cpu()
print("***forward***")
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 0, 
                          get_func_fixed_dim(sparse_accumulation_loops, 0), 10)
print("python loops; active dim 0; forward; cpu: ", np.mean(times[1:]))
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 0, 
                          get_func_fixed_dim(sparse_accumulation_index_add, 0), 10)
print("torch index_add_; active dim 0; forward; cpu: ", np.mean(times[1:]))
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 0,
                          sparse_accumulation_active_dim_first.SparseAccumulationActiveDimFirst.apply, 10)
print("cpp; active dim 0; forward; cpu: ", np.mean(times[1:]))

print()
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 1, 
                          get_func_fixed_dim(sparse_accumulation_loops, 1), 10)
print("python loops; active dim 1; forward; cpu: ", np.mean(times[1:]))
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 1, 
                          get_func_fixed_dim(sparse_accumulation_index_add, 1), 10)
print("torch index_add_; active dim 1; forward; cpu: ", np.mean(times[1:]))
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 1,
                          sparse_accumulation_active_dim_middle.SparseAccumulationActiveDimMiddle.apply, 10)
print("cpp; active dim 1; forward; cpu: ", np.mean(times[1:]))

print()
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 2, get_func_fixed_dim(sparse_accumulation_loops, 2), 10)
print("python loops; active dim 2; forward; cpu:", np.mean(times[1:]))
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 2, get_func_fixed_dim(sparse_accumulation_index_add, 2), 10)
print("torch index_add_; active dim 2; forward; cpu: ", np.mean(times[1:]))
times = benchmark_forward_cpu(BATCH_SIZE, N_FEATURES, 2, sparse_accumulation.SparseAccumulation.apply, 10)
print("cpp; active dim 2; forward; cpu ", np.mean(times[1:]))
print("***backward***")
times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 0, 
                           get_func_fixed_dim(sparse_accumulation_loops, 0), 10)
print("python loops; active dim 0; backward; cpu ", np.mean(times[1:]))
times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 0, 
                           get_func_fixed_dim(sparse_accumulation_index_add, 0), 10)
print("torch index_add_; active dim 0; backward; cpu ", np.mean(times[1:]))
times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 0,
                           sparse_accumulation_active_dim_first.SparseAccumulationActiveDimFirst.apply, 10)
print("cpp; active dim 0; backward; cpu ", np.mean(times[1:]))

print()

times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 1, 
                           get_func_fixed_dim(sparse_accumulation_loops, 1), 10)
print("python loops; active dim 1; backward; cpu ", np.mean(times[1:]))
times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 1, 
                           get_func_fixed_dim(sparse_accumulation_index_add, 1), 10)
print("torch index_add_; active dim 1; backward; cpu ", np.mean(times[1:]))
times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 1,
                           sparse_accumulation_active_dim_middle.SparseAccumulationActiveDimMiddle.apply, 10)
print("cpp; active dim 1; backward; cpu ", np.mean(times[1:]))


print()
times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 2, 
                           get_func_fixed_dim(sparse_accumulation_index_add, 2), 10)
print("python loops; active dim 2; backward; cpu ", np.mean(times[1:]))
times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 2, 
                           get_func_fixed_dim(sparse_accumulation_index_add, 2), 10)
print("torch index_add_; active dim 2; backward; cpu ", np.mean(times[1:]))
times = benchmark_backward_cpu(BATCH_SIZE, N_FEATURES, 2, sparse_accumulation.SparseAccumulation.apply, 10)
print("cpp; active dim 2; backward; cpu ", np.mean(times[1:]))