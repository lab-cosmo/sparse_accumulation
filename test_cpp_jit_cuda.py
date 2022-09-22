from email import generator
import torch
from clebsch_gordan import get_real_clebsch_gordan, ClebschGordan
from sparse_accumulation_plain_torch import sparse_accumulation_loops
from torch.utils import cpp_extension
#import sparse_accumulation #, sparse_accumulation_active_dim_first
print(' compilation')
cpp_extension.load(
    name="sparse_accumulation_cuda",
    sources=["cuda/sparse_accumulation_cuda_kernel2D.cu"],
    is_python_module=False,
    extra_cuda_cflags=None,
    verbose=True,
)
#torch.ops.sparse_accumulation_cuda_cpp .reduce_custom_autograd

print('finished compilation')
def get_rule(L_MAX):
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
    multipliers = torch.tensor(multipliers,dtype=torch.float32)
    
    return m1_aligned, m2_aligned, mu_aligned, multipliers


def get_rule_gpu(L_MAX):
    clebsch = ClebschGordan(L_MAX).precomputed_
    indices = get_real_clebsch_gordan(clebsch[L_MAX, L_MAX, L_MAX], L_MAX, L_MAX, L_MAX)
    
    m1_aligned, m2_aligned = [], []
    multipliers, mu_aligned = [], []
    for mu in range(0, 2 * L_MAX + 1):
        for el in indices[mu]:
            m1, m2, multiplier = el
            m1_aligned.append(m1)
            m2_aligned.append(m2)
            multipliers.append(multiplier*1.0)
            mu_aligned.append(mu)
    m1_aligned = torch.tensor(m1_aligned,dtype=torch.int64,device='cuda')
    m2_aligned = torch.tensor(m2_aligned,dtype=torch.int64,device='cuda')
    mu_aligned = torch.tensor(mu_aligned,dtype=torch.int64,device='cuda')
    #multipliers = torch.tensor(multipliers,dtype=torch.float64,device='cuda')
    multipliers = torch.tensor(multipliers,dtype=torch.float32,device='cuda')
    
    return m1_aligned, m2_aligned, mu_aligned, multipliers

def test_forward(epsilon = 1e-10):
    L_MAX = 2
    BATCH_SIZE = 2
    N_FEATURES = 2
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)
    #m1_aligned_d, m2_aligned_d, mu_aligned_d, multipliers_d = get_rule_gpu(L_MAX)
    m1_aligned_d = m1_aligned.clone().cuda()
    m2_aligned_d = m2_aligned.clone().cuda()
    mu_aligned_d = mu_aligned.clone().cuda()
    multipliers_d = multipliers.clone().cuda()
    generator = torch.Generator()
    generator.manual_seed(30)
    X1 = torch.randn((BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1),generator=generator)
    X2 = torch.randn((BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1),generator=generator)
    #X1_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    #X2_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    X1_d = X1.clone().cuda() # torch.randn(BATCH_SIZE, N_FEATURES,device="cuda")
    X2_d = X2.clone().cuda() # torch.randn(BATCH_SIZE,device="cuda")
    print(f"{X2_d.stride()=}")
    print(f"{X2_d.transpose(0,1).stride()=}")
    
    python_loops_output = sparse_accumulation_loops(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers,active_dim=2)
    cuda_output = torch.ops.sparse_accumulation_cuda.forward(X1_d,
                            X2_d,mu_aligned_d, 2 * L_MAX + 1, m1_aligned_d, m2_aligned_d,multipliers_d)
    #delta = cuda_output[0] - X1_d  
    #python_loops_output_gpu = torch.tensor_
    #delta = python_loops_output - cuda_output[0]
    
    #relative_error = torch.mean(torch.abs(delta))# / torch.mean(torch.abs(python_loops_output))
    #relative_error = torch.mean(torch.abs(delta)) / torch.mean(torch.abs(python_loops_output))
    print("now I print")
    print('X1 ', X1)
    print('X2 ', X2)
    print('X1_d ', X1_d)
    print('X2_d ', X2_d)
    print('cuda_output ',cuda_output)
    #print(f'{multipliers=} ')
    #print(f'{multipliers_d=} ')
    #print('X1_d ',X1_d)
    print(f'{python_loops_output=}')
    #print(f'{relative_error=}')


    #assert relative_error < epsilon


if __name__ =="__main__" : 
    test_forward()

