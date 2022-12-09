import os
import pytest
from functools import partial

import torch
from torch.utils import cpp_extension
import numpy as np
from sparse_accumulation.clebsch_gordan import ClebschGordan, get_real_clebsch_gordan
from sparse_accumulation.reference_implementations import sparse_accumulation_loops
from sparse_accumulation import accumulate, get_cg_transformation_rule

cpp_extension.load(
    name="sparse_accumulation_cuda",
    sources=["sparse_accumulation/cuda_extension/sparse_accumulation_cuda_kernel2D.cu"],
    is_python_module=False,
    extra_cuda_cflags=None,
    verbose=True,
)


def get_rule(l1, l2, l_output, dtype=torch.float64, device="cpu"):
    cachepath = f".cache/clebsch_gordan_l1_{l1}_l2_{l2}_l_output_{l_output}_dtype_{dtype}.pt"
    if os.path.isfile(cachepath):
        return torch.load(cachepath, map_location=device)

    
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_cg_transformation_rule(l1, l2, l_output, dtype = dtype, device = device)
    os.makedirs(os.path.dirname(cachepath), exist_ok=True)
    torch.save([m1_aligned, m2_aligned, mu_aligned, multipliers], cachepath)
    return m1_aligned, m2_aligned, mu_aligned, multipliers

def test_forward(L1, L2, L_OUTPUT, BATCH_SIZE, N_FEATURES, dtype):
    if (L_OUTPUT < abs(L1 - L2)) or (L_OUTPUT > L1 + L2):
        pytest.skip()
        
    atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (1e-7, 1e-8)
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L1, L2, L_OUTPUT, dtype)

    m1_aligned_d = m1_aligned.clone().cuda()
    m2_aligned_d = m2_aligned.clone().cuda()
    mu_aligned_d = mu_aligned.clone().cuda()
    multipliers_d = multipliers.clone().cuda()

    generator = torch.Generator()
    generator.manual_seed(30)
    X1 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L1 + 1),
        generator=generator,
        dtype=dtype,
    )
    X2 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L2 + 1),
        generator=generator,
        dtype=dtype,
    )
    
    # X1_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    # X2_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    X1_d = X1.clone().cuda()  # torch.randn(BATCH_SIZE, N_FEATURES,device="cuda")
    X2_d = X2.clone().cuda()  # torch.randn(BATCH_SIZE,device="cuda")

    python_loops_output = sparse_accumulation_loops(
        X1,
        X2,
        mu_aligned,
        2 * L_OUTPUT + 1,
        m1_aligned,
        m2_aligned,
        multipliers,
        active_dim=2,
    )

    cuda_output = accumulate(
        X1_d,
        X2_d,
        mu_aligned_d,
        2 * L_OUTPUT + 1,
        m1_aligned_d,
        m2_aligned_d,
        multipliers_d,
    )
    X1_d_2D = X1_d.reshape(-1,2 * L2 + 1)
    X2_d_2D = X2_d.reshape(-1,2 * L2 + 1)
    
    cuda_output2D = accumulate(
        X1_d_2D,
        X2_d_2D,
        mu_aligned_d,
        2 * L_OUTPUT + 1,
        m1_aligned_d,
        m2_aligned_d,
        multipliers_d,
    )

    cuda_output_cpu = cuda_output.cpu()
    cuda_output2D_cpu = cuda_output2D.cpu()
    
    cuda_output2D_cpu = cuda_output2D_cpu.reshape((BATCH_SIZE, N_FEATURES, 2 * L1 + 1))
    
    delta = python_loops_output - cuda_output_cpu
    delta2D = python_loops_output - cuda_output2D_cpu
    
    relative_error = torch.amax(torch.abs(delta / python_loops_output))
    relative_error2D = torch.amax(torch.abs(delta2D / python_loops_output))

    assert torch.allclose(
        python_loops_output, cuda_output_cpu, atol=atol, rtol=rtol
    ), f"assertion failed \n {cuda_output=} \n {python_loops_output=}"

    errmax = torch.amax(torch.abs(delta))
    print(f"{errmax=}")
    print(f"{torch.amin(torch.abs(cuda_output_cpu))=}")
    
    errmax = torch.amax(torch.abs(delta2D))
    print(f"{errmax=}")
    print(f"{torch.amin(torch.abs(cuda_output2D_cpu))=}")
    
    assert torch.allclose(python_loops_output, cuda_output_cpu, atol=atol, rtol=rtol)
    # print(f"{python_time=} s")
    print()

def test_backward(L1, L2, L_OUTPUT, BATCH_SIZE, N_FEATURES, seed, dtype):
    if (L_OUTPUT < abs(L1 - L2)) or (L_OUTPUT > L1 + L2):
        pytest.skip()
    atol, rtol = (1e-5, 1e-6) if dtype is torch.float32 else (1e-7, 1e-8)
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L1, L2, L_OUTPUT, dtype)    

    m1_aligned_d = m1_aligned.clone().cuda()
    m2_aligned_d = m2_aligned.clone().cuda()
    mu_aligned_d = mu_aligned.clone().cuda()
    multipliers_d = multipliers.clone().cuda()
    generator = torch.Generator()
    generator.manual_seed(seed)
    X1 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L1 + 1),
        generator=generator,
        dtype=dtype,
    )
    X2 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L2 + 1),
        generator=generator,
        dtype=dtype,
    )
    # X1_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    # X2_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    X1_d = X1.clone().cuda()  # torch.randn(BATCH_SIZE, N_FEATURES,device="cuda")
    X2_d = X2.clone().cuda()  # torch.randn(BATCH_SIZE,device="cuda")

    X1.requires_grad = True
    X2.requires_grad = True
   
    python_loops_output = sparse_accumulation_loops(
        X1,
        X2,
        mu_aligned,
        2 * L_OUTPUT + 1,
        m1_aligned,
        m2_aligned,
        multipliers,
        active_dim=2,
    )
    output_grad = torch.zeros(*python_loops_output.shape, dtype=dtype)
    python_loops_output.backward(gradient=output_grad)

    X1_grad_python_loops = torch.detach(torch.clone(X1.grad))
    X2_grad_python_loops = torch.detach(torch.clone(X2.grad))

   
    output_grad_d = output_grad.clone().cuda()

  
    X1_d.requires_grad = True
    X2_d.requires_grad = True
    cuda_output = accumulate(
        X1_d,
        X2_d,
        mu_aligned_d,
        2 * L_OUTPUT + 1,
        m1_aligned_d,
        m2_aligned_d,
        multipliers_d
    )
    
    cuda_output.backward(gradient=output_grad_d)
    X1_grad_cuda = torch.detach(torch.clone(X1_d.grad))
    X2_grad_cuda = torch.detach(torch.clone(X2_d.grad))
    
    
   
    X1_grad_cuda = X1_grad_cuda.cpu()
    X2_grad_cuda = X2_grad_cuda.cpu()

    errmax1 = torch.amax(torch.abs(X1_grad_python_loops - X1_grad_cuda))
    errmax2 = torch.amax(torch.abs(X2_grad_python_loops - X2_grad_cuda))
    print(f"{errmax1=}")
    print(f"{torch.amin(torch.abs(X1_grad_cuda))=}")
    print(f"{errmax2=}")
    print(f"{torch.amin(torch.abs(X2_grad_cuda))=}")
    
    # print(f"{X1_grad_cuda=}")
    # print(f"{X1_grad_python_loops=}")
    
    # print(f"{X2_grad_cuda=}")
    # print(f"{X2_grad_python_loops=}")
    # assert torch.allclose(python_loops_output , cuda_output_cpu,atol=atol)

    assert torch.allclose(
        X1_grad_python_loops, X1_grad_cuda, atol=atol, rtol=rtol
    )
    assert torch.allclose(
        X2_grad_python_loops, X2_grad_cuda, atol=atol, rtol=rtol
    )




# @pytest.mark.parametrize("function", [
#     partial(sparse_accumulation_loops, active_dim=2),
#     accumulate
# ])
# @pytest.mark.parametrize("L1", [3, 5, 7])
# @pytest.mark.parametrize("L2", [3, 5, 7])
# @pytest.mark.parametrize("L_OUTPUT", [3, 5, 7])
# @pytest.mark.parametrize("BATCH_SIZE", [1, 20, 200])
# @pytest.mark.parametrize("N_FEATURES", [1, 20, 105])
# @pytest.mark.parametrize("dtype", [torch.float64])
# @pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_backward_gradcheck(function, L1, L2, L_OUTPUT, BATCH_SIZE, N_FEATURES, dtype, device):
    if (L_OUTPUT < abs(L1 - L2)) or (L_OUTPUT > L1 + L2):
        pytest.skip()
    atol, rtol = (5e-2, 1e-3) if dtype == torch.float32 else (1e-7, 1e-8)
    if device == 'cpu' and function == accumulate:
        pytest.skip()

    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L1, L2, L_OUTPUT, dtype, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(0xDEADBEEF)
    X1 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L1 + 1),
        requires_grad=True,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    X2 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L2 + 1),
        requires_grad=True,
        generator=generator,
        dtype=dtype,
        device=device,
    )

    assert torch.autograd.gradcheck(
        function,
        (X1, X2, mu_aligned, 2 * L_OUTPUT + 1, m1_aligned, m2_aligned, multipliers),
        fast_mode=True, atol=atol, rtol=rtol,
    )

    print("torch.autograd.gradcheck passed\n")

if __name__ == '__main__':
    
    # @pytest.mark.parametrize("L1", [3, 5, 7])
    # @pytest.mark.parametrize("L2", [3, 5, 7])
    # @pytest.mark.parametrize("L_OUTPUT", [3, 5, 7])
    # @pytest.mark.parametrize("BATCH_SIZE", [1, 20, 200])
    # @pytest.mark.parametrize("N_FEATURES", [1, 20, 105])
    L1 = 3
    L2 = 3
    L_OUTPUT = 3
    BATCH_SIZE = 1
    N_FEATURES = 1
    dtype = torch.float64
    device = 'cuda'
    test_forward(L1, L2, L_OUTPUT, BATCH_SIZE, N_FEATURES, dtype)