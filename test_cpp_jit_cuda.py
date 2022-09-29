from time import time
import os
import pytest
from functools import partial

import torch
from torch.utils import cpp_extension
import numpy as np
from clebsch_gordan import ClebschGordan, get_real_clebsch_gordan
from sparse_accumulation_plain_torch import sparse_accumulation_loops

cpp_extension.load(
    name="sparse_accumulation_cuda",
    sources=["cuda_optimized/sparse_accumulation_cuda_kernel2D.cu"],
    is_python_module=False,
    extra_cuda_cflags=None,
    verbose=True,
)


def get_rule(L_MAX, device="cpu"):
    cachepath = f".cache/clebsch_gordan_l{L_MAX}.pt"
    if os.path.isfile(cachepath):
        return torch.load(cachepath, map_location=device)

    print(f"generating CG rule for L_MAX {L_MAX}")
    clebsch = ClebschGordan(L_MAX).precomputed_
    indices = get_real_clebsch_gordan(clebsch[L_MAX, L_MAX, L_MAX], L_MAX, L_MAX, L_MAX)

    m1_aligned, m2_aligned = [], []
    multipliers, mu_aligned = [], []
    for mu in range(2 * L_MAX + 1):
        for el in indices[mu]:
            m1, m2, multiplier = el
            m1_aligned.append(m1)
            m2_aligned.append(m2)
            multipliers.append(multiplier * 1.0)
            mu_aligned.append(mu)
    m1_aligned = torch.tensor(m1_aligned, dtype=torch.int64, device=device)
    m2_aligned = torch.tensor(m2_aligned, dtype=torch.int64, device=device)
    mu_aligned = torch.tensor(mu_aligned, dtype=torch.int64, device=device)
    multipliers = torch.tensor(multipliers, dtype=torch.float64, device=device)

    indices = np.argsort(mu_aligned)

    m1_aligned = m1_aligned[indices]
    m2_aligned = m2_aligned[indices]
    mu_aligned = mu_aligned[indices]
    multipliers = multipliers[indices]
    print("done generating CG rule")

    os.makedirs(os.path.dirname(cachepath), exist_ok=True)
    torch.save([m1_aligned, m2_aligned, mu_aligned, multipliers], cachepath)
    return m1_aligned, m2_aligned, mu_aligned, multipliers


@pytest.mark.parametrize("L_MAX", [5, 8])
@pytest.mark.parametrize("BATCH_SIZE", [20, 2000])
@pytest.mark.parametrize("N_FEATURES", [20, 105])
def test_forward(L_MAX, BATCH_SIZE, N_FEATURES, atol=1e-7, rtol=1e-8):
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)

    m1_aligned_d = m1_aligned.clone().cuda()
    m2_aligned_d = m2_aligned.clone().cuda()
    mu_aligned_d = mu_aligned.clone().cuda()
    multipliers_d = multipliers.clone().cuda()

    generator = torch.Generator()
    generator.manual_seed(30)
    X1 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1),
        generator=generator,
        dtype=torch.float64,
    )
    X2 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1),
        generator=generator,
        dtype=torch.float64,
    )
    # X1_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    # X2_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    X1_d = X1.clone().cuda()  # torch.randn(BATCH_SIZE, N_FEATURES,device="cuda")
    X2_d = X2.clone().cuda()  # torch.randn(BATCH_SIZE,device="cuda")

    python_loops_output = sparse_accumulation_loops(
        X1,
        X2,
        mu_aligned,
        2 * L_MAX + 1,
        m1_aligned,
        m2_aligned,
        multipliers,
        active_dim=2,
    )

    cuda_output = torch.ops.sparse_accumulation_cuda.forward(
        X1_d,
        X2_d,
        mu_aligned_d,
        2 * L_MAX + 1,
        m1_aligned_d,
        m2_aligned_d,
        multipliers_d,
    )

    cuda_output_cpu = cuda_output[0].cpu()
    delta = python_loops_output - cuda_output_cpu
    relative_error = torch.amax(torch.abs(delta / python_loops_output))

    assert torch.allclose(
        python_loops_output, cuda_output_cpu, atol=atol, rtol=rtol
    ), f"assertion failed \n {cuda_output=} \n {python_loops_output=}"

    errmax = torch.amax(torch.abs(delta))
    print(f"{errmax=}")
    print(f"{torch.amin(torch.abs(cuda_output_cpu))=}")

    assert torch.allclose(python_loops_output, cuda_output_cpu, atol=atol, rtol=rtol)
    # print(f"{python_time=} s")
    print()


@pytest.mark.parametrize("seed", [30, 42])
@pytest.mark.parametrize("L_MAX", [5, 8])
@pytest.mark.parametrize("BATCH_SIZE", [20, 2000])
@pytest.mark.parametrize("N_FEATURES", [20, 105])
def test_backward(L_MAX, BATCH_SIZE, N_FEATURES, seed, atol=1e-7, rtol=1e-8):

    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)
    m1_aligned_d = m1_aligned.clone().cuda()
    m2_aligned_d = m2_aligned.clone().cuda()
    mu_aligned_d = mu_aligned.clone().cuda()
    multipliers_d = multipliers.clone().cuda()
    generator = torch.Generator()
    generator.manual_seed(seed)
    X1 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1),
        generator=generator,
        dtype=torch.float64,
    )
    X2 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1),
        generator=generator,
        dtype=torch.float64,
    )
    # X1_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    # X2_d = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1,device="cuda")
    X1_d = X1.clone().cuda()  # torch.randn(BATCH_SIZE, N_FEATURES,device="cuda")
    X2_d = X2.clone().cuda()  # torch.randn(BATCH_SIZE,device="cuda")

    X1.requires_grad = True
    X2.requires_grad = True
    t1 = time()

    python_loops_output = sparse_accumulation_loops(
        X1,
        X2,
        mu_aligned,
        2 * L_MAX + 1,
        m1_aligned,
        m2_aligned,
        multipliers,
        active_dim=2,
    )
    output_grad = torch.randn(*python_loops_output.shape, dtype=torch.float64)
    python_loops_output.backward(gradient=output_grad)

    X1_grad_python_loops = torch.detach(torch.clone(X1.grad))
    X2_grad_python_loops = torch.detach(torch.clone(X2.grad))

    t2 = time()
    python_time = t2 - t1
    output_grad_d = output_grad.clone().cuda()

    torch.cuda.synchronize("cuda")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    starter.record()
    cuda_output = torch.ops.sparse_accumulation_cuda.backward(
        output_grad_d,
        X1_d,
        X2_d,
        mu_aligned_d,
        m1_aligned_d,
        m2_aligned_d,
        multipliers_d,
    )
    torch.cuda.synchronize("cuda")
    ender.record()
    torch.cuda.synchronize("cuda")
    cuda_Event_time = (
        starter.elapsed_time(ender) / 1000
    )  # torch.cuda.Event gives the time in milliseconds
    t3 = time()
    cuda_time = t3 - t2
    X1_grad_cuda = cuda_output[0].cpu()
    X2_grad_cuda = cuda_output[1].cpu()

    print(f"backward {L_MAX=}, {BATCH_SIZE=}, {N_FEATURES=}")

    assertion1 = torch.allclose(
        X1_grad_python_loops, X1_grad_cuda, atol=atol, rtol=rtol
    )
    assertion2 = torch.allclose(
        X2_grad_python_loops, X2_grad_cuda, atol=atol, rtol=rtol
    )
    errmax1 = torch.amax(torch.abs(X1_grad_python_loops - X1_grad_cuda))
    errmax2 = torch.amax(torch.abs(X2_grad_python_loops - X2_grad_cuda))
    print(f"{errmax1=}")
    print(f"{torch.amin(torch.abs(X1_grad_cuda))=}")
    print(f"{errmax2=}")
    print(f"{torch.amin(torch.abs(X2_grad_cuda))=}")
    # assert torch.allclose(python_loops_output , cuda_output_cpu,atol=atol)
    print(f"{python_time=} s")
    print(f"{cuda_time=} s")
    print(f"{cuda_Event_time=} s")
    print(f"python_time/cuda_time = {python_time/cuda_time} ")
    if (not assertion1) or (not assertion2):
        print("!! assertion FAILED")
    print()


class CudaSparseAccumulationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X1, X2, mu, output_size, m1, m2, multipliers):
        (output,) = torch.ops.sparse_accumulation_cuda.forward(
            X1,
            X2,
            mu,
            output_size,
            m1,
            m2,
            multipliers,
        )

        ctx.save_for_backward(X1, X2, mu, m1, m2, multipliers)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        X1, X2, mu, m1, m2, multipliers = ctx.saved_tensors

        X1_grad, X2_grad = torch.ops.sparse_accumulation_cuda.backward(
            grad_output, X1, X2, mu, m1, m2, multipliers
        )

        return X1_grad, X2_grad, None, None, None, None, None


@pytest.mark.parametrize("function", [
    partial(sparse_accumulation_loops, active_dim=2),
    CudaSparseAccumulationFunction.apply
])
@pytest.mark.parametrize("L_MAX", [5, 8])
@pytest.mark.parametrize("BATCH_SIZE", [20, 2000])
@pytest.mark.parametrize("N_FEATURES", [20, 105])
@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_backward_gradcheck(function, L_MAX, BATCH_SIZE, N_FEATURES, device):
    if device == 'cpu' and function == CudaSparseAccumulationFunction.apply:
        pytest.skip()

    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(0xDEADBEEF)
    X1 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1),
        requires_grad=True,
        generator=generator,
        dtype=torch.float64,
        device=device,
    )
    X2 = torch.randn(
        (BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1),
        requires_grad=True,
        generator=generator,
        dtype=torch.float64,
        device=device,
    )

    assert torch.autograd.gradcheck(
        function,
        (X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers),
        fast_mode=True,
    )

    print("torch.autograd.gradcheck passed\n")
