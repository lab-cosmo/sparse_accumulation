import torch
from sparse_accumulation.clebsch_gordan import get_real_clebsch_gordan, ClebschGordan
from sparse_accumulation_plain_torch import sparse_accumulation_loops
from sparse_accumulation.cpu_extension import sparse_accumulation_active_dim_first, sparse_accumulation_active_dim_middle
from sparse_accumulation import accumulate

import numpy as np

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
    multipliers = torch.FloatTensor(multipliers)
    
    indices = np.argsort(mu_aligned)

    m1_aligned = m1_aligned[indices]
    m2_aligned = m2_aligned[indices]
    mu_aligned = mu_aligned[indices]
    multipliers = multipliers[indices]

    return m1_aligned, m2_aligned, mu_aligned, multipliers

def test_forward(epsilon = 1e-7):
    print("Testing forward pass with the active dimension being the last one")
    L_MAX = 5
    BATCH_SIZE = 1000
    N_FEATURES = 100
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)
    
    
    X1 = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1)
    X2 = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1)

    
    python_loops_output = sparse_accumulation_loops(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers,
                                                    active_dim = 2)
    cpp_output = accumulate(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
    delta = python_loops_output - cpp_output
    
    relative_error = torch.mean(torch.abs(delta)) / torch.mean(torch.abs(python_loops_output))
    assert  relative_error < epsilon
    
    
def test_forward_active_dim_first(epsilon = 1e-7):
    print("Testing forward pass with the active dimension being the first one")
    L_MAX = 5
    BATCH_SIZE = 1000
    N_FEATURES = 100
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)
    
    
    X1 = torch.randn(2 * L_MAX + 1, BATCH_SIZE, N_FEATURES)
    X2 = torch.randn(2 * L_MAX + 1, BATCH_SIZE, N_FEATURES)

    
    python_loops_output = sparse_accumulation_loops(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers,
                                                    active_dim = 0)
    cpp_output = sparse_accumulation_active_dim_first.SparseAccumulationActiveDimFirst.apply(X1, X2, mu_aligned,
                                                          2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
    delta = python_loops_output - cpp_output
    
    relative_error = torch.mean(torch.abs(delta)) / torch.mean(torch.abs(python_loops_output))
    assert  relative_error < epsilon
    
def test_forward_active_dim_middle(epsilon = 1e-7):
    print("Testing forward pass with the active dimension being the middle one")
    L_MAX = 5
    BATCH_SIZE = 1000
    N_FEATURES = 100
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)
    
    
    X1 = torch.randn(BATCH_SIZE, 2 * L_MAX + 1, N_FEATURES)
    X2 = torch.randn(BATCH_SIZE, 2 * L_MAX + 1, N_FEATURES)

    
    python_loops_output = sparse_accumulation_loops(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers,
                                                    active_dim = 1)
    cpp_output = sparse_accumulation_active_dim_middle.SparseAccumulationActiveDimMiddle.apply(X1, X2, mu_aligned,
                                                          2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
    delta = python_loops_output - cpp_output
    
    relative_error = torch.mean(torch.abs(delta)) / torch.mean(torch.abs(python_loops_output))
    assert  relative_error < epsilon
    
    
    
def get_relative_error(first, second):
    delta = first - second
    return torch.sum(torch.abs(delta)) / torch.sum(torch.abs(first))


def test_backward(epsilon = 1e-7):
    print("Testing backward pass with the active dimension being the last one")
    L_MAX = 5
    BATCH_SIZE = 1000
    N_FEATURES = 100
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)
  
    X1 = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1)
    X2 = torch.randn(BATCH_SIZE, N_FEATURES, 2 * L_MAX + 1)

    X1.requires_grad = True
    X2.requires_grad = True
    python_loops_output = sparse_accumulation_loops(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers, 
                                                   active_dim = 2)
    output_grad = torch.randn(*python_loops_output.shape)
    python_loops_output.backward(gradient = output_grad)
    
    X1_grad_python_loops = torch.detach(torch.clone(X1.grad))
    X2_grad_python_loops = torch.detach(torch.clone(X2.grad))

    X1.grad.zero_()
    X2.grad.zero_()

    cpp_output = accumulate(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
    cpp_output.backward(gradient = output_grad)

    X1_grad_cpp = torch.detach(torch.clone(X1.grad))
    X2_grad_cpp = torch.detach(torch.clone(X2.grad))
    
    assert get_relative_error(X1_grad_python_loops, X1_grad_cpp) < epsilon
    assert get_relative_error(X2_grad_python_loops, X2_grad_cpp) < epsilon
    

def test_backward_active_dim_middle(epsilon = 1e-7):
    print("Testing backward pass with the active dimension being the middle one")
    L_MAX = 5
    BATCH_SIZE = 1000
    N_FEATURES = 100
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)
  
    X1 = torch.randn(BATCH_SIZE, 2 * L_MAX + 1, N_FEATURES)
    X2 = torch.randn(BATCH_SIZE, 2 * L_MAX + 1, N_FEATURES)

    X1.requires_grad = True
    X2.requires_grad = True
    python_loops_output = sparse_accumulation_loops(X1, X2, mu_aligned, 2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers,
                                                    active_dim = 1)
   
    output_grad = torch.randn(*python_loops_output.shape)
    python_loops_output.backward(gradient = output_grad)
    
    X1_grad_python_loops = torch.detach(torch.clone(X1.grad))
    X2_grad_python_loops = torch.detach(torch.clone(X2.grad))

    X1.grad.zero_()
    X2.grad.zero_()

    cpp_output = sparse_accumulation_active_dim_middle.SparseAccumulationActiveDimMiddle.apply(X1, X2, mu_aligned,
                                                          2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
    
    cpp_output.backward(gradient = output_grad)

    X1_grad_cpp = torch.detach(torch.clone(X1.grad))
    X2_grad_cpp = torch.detach(torch.clone(X2.grad))
    
    assert get_relative_error(X1_grad_python_loops, X1_grad_cpp) < epsilon
    assert get_relative_error(X2_grad_python_loops, X2_grad_cpp) < epsilon
    
    
def test_backward_active_dim_first(epsilon = 1e-7):
    print("Testing backward pass with the active dimension being the first one")
    L_MAX = 5
    BATCH_SIZE = 1000
    N_FEATURES = 100
    m1_aligned, m2_aligned, mu_aligned, multipliers = get_rule(L_MAX)
  
    X1 = torch.randn(2 * L_MAX + 1, BATCH_SIZE, N_FEATURES)
    X2 = torch.randn(2 * L_MAX + 1, BATCH_SIZE, N_FEATURES)

    X1.requires_grad = True
    X2.requires_grad = True
    python_loops_output = sparse_accumulation_loops(X1, X2, mu_aligned, 2 * L_MAX + 1,
                                                    m1_aligned, m2_aligned, multipliers,
                                                    active_dim = 0)
    output_grad = torch.randn(*python_loops_output.shape)
    python_loops_output.backward(gradient = output_grad)
    
    X1_grad_python_loops = torch.detach(torch.clone(X1.grad))
    X2_grad_python_loops = torch.detach(torch.clone(X2.grad))

    X1.grad.zero_()
    X2.grad.zero_()

    cpp_output = sparse_accumulation_active_dim_first.SparseAccumulationActiveDimFirst.apply(X1, X2, mu_aligned,
                                                          2 * L_MAX + 1, m1_aligned, m2_aligned, multipliers)
    cpp_output.backward(gradient = output_grad)

    X1_grad_cpp = torch.detach(torch.clone(X1.grad))
    X2_grad_cpp = torch.detach(torch.clone(X2.grad))
    
    assert get_relative_error(X1_grad_python_loops, X1_grad_cpp) < epsilon
    assert get_relative_error(X2_grad_python_loops, X2_grad_cpp) < epsilon
    
    