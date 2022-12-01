from sympy import S
from sympy.physics.wigner import clebsch_gordan

try:
    import wigners
except ImportError:
    wigners = None

import numpy as np
import torch
from .unified_operation import accumulate

def _compute_cg(l1, l2, l, m1, m2):
    if wigners is None:
        # use sympy
        return float(clebsch_gordan(S(l1), S(l2), S(l), S(m1), S(m2), S(m1 + m2)))
    else:
        if abs(m1) > l1 or abs(m2) > l2 or abs(m1 + m2) > l:
            return 0.0
        return wigners.clebsch_gordan(l1, m1, l2, m2, l, m1 + m2)


class ClebschGordan:
    def __init__(self, l_max):
        self.l_max_ = l_max
        self.precomputed_ = np.zeros(
            [l_max + 1, l_max + 1, l_max + 1, 2 * l_max + 1, 2 * l_max + 1]
        )

        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                for l in range(l_max + 1):
                    for m1 in range(-l_max, l_max + 1):
                        for m2 in range(-l_max, l_max + 1):
                            now = _compute_cg(l1, l2, l, m1, m2)
                            self.precomputed_[l1, l2, l, m1 + l1, m2 + l2] = now

class PartialClebschGordan:
    def __init__(self, l1, l2, l_output):
        self.l1 = l1
        self.l2 = l2
        self.l_output = l_output
        
        self.values = np.zeros([2 * l1 + 1, 2 * l2 + 1])
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                self.values[m1 + l1, m2 + l2] = _compute_cg(l1, l2, l_output, m1, m2)
        
def _multiply(first, second, multiplier):
    return [first[0], second[0], first[1] * second[1] * multiplier]


def _multiply_sequence(sequence, multiplier):
    result = []

    for el in sequence:
        # print(el)
        # print(len(el))
        result.append([el[0], el[1], el[2] * multiplier])
    return result


def _get_conversion(l, m):
    if m < 0:
        X_re = [abs(m) + l, 1.0 / np.sqrt(2)]
        X_im = [m + l, -1.0 / np.sqrt(2)]
    if m == 0:
        X_re = [l, 1.0]
        X_im = [l, 0.0]
    if m > 0:
        if m % 2 == 0:
            X_re = [m + l, 1.0 / np.sqrt(2)]
            X_im = [-m + l, 1.0 / np.sqrt(2)]
        else:
            X_re = [m + l, -1.0 / np.sqrt(2)]
            X_im = [-m + l, -1.0 / np.sqrt(2)]
    return X_re, X_im


def _compress(sequence, epsilon=1e-15):
    result = []
    for i in range(len(sequence)):
        m1, m2, multiplier = sequence[i][0], sequence[i][1], sequence[i][2]
        already = False
        for j in range(len(result)):
            if (m1 == result[j][0]) and (m2 == result[j][1]):
                already = True
                break

        if not already:
            multiplier = 0.0
            for j in range(i, len(sequence)):
                if (m1 == sequence[j][0]) and (m2 == sequence[j][1]):
                    multiplier += sequence[j][2]
            if np.abs(multiplier) > epsilon:
                result.append([m1, m2, multiplier])
    # print(len(sequence), '->', len(result))
    return result


def get_real_clebsch_gordan(clebsch, l1, l2, lambd):
    result = [[] for _ in range(2 * lambd + 1)]
    for mu in range(0, lambd + 1):
        real_now = []
        imag_now = []
        for m2 in range(max(-l2, mu - l1), min(l2, mu + l1) + 1):
            m1 = mu - m2
            X1_re, X1_im = _get_conversion(l1, m1)
            X2_re, X2_im = _get_conversion(l2, m2)

            real_now.append(_multiply(X1_re, X2_re, clebsch[m1 + l1, m2 + l2]))
            real_now.append(_multiply(X1_im, X2_im, -clebsch[m1 + l1, m2 + l2]))

            imag_now.append(_multiply(X1_re, X2_im, clebsch[m1 + l1, m2 + l2]))
            imag_now.append(_multiply(X1_im, X2_re, clebsch[m1 + l1, m2 + l2]))
        # print(real_now)
        if (l1 + l2 - lambd) % 2 == 1:
            imag_now, real_now = real_now, _multiply_sequence(imag_now, -1)
        if mu > 0:
            if mu % 2 == 0:
                result[mu + lambd] = _multiply_sequence(real_now, np.sqrt(2))
                result[-mu + lambd] = _multiply_sequence(imag_now, np.sqrt(2))
            else:
                result[mu + lambd] = _multiply_sequence(real_now, -np.sqrt(2))
                result[-mu + lambd] = _multiply_sequence(imag_now, -np.sqrt(2))
        else:
            result[lambd] = real_now

    for i in range(len(result)):
        result[i] = _compress(result[i])
    return result

def check_l_consistency(l1, l2, l_output):
    if (l_output < abs(l1 - l2)) or (l_output > (l1 + l2)):
        raise ValueError("l_output must be in between |l1 - l2| and (l1 + l2)")
        
def get_cg_transformation_rule(l1, l2, l_output, dtype = torch.float32, device = "cpu"):
    check_l_consistency(l1, l2, l_output)
            
    clebsch = PartialClebschGordan(l1, l2, l_output).values
    indices = get_real_clebsch_gordan(clebsch, l1, l2, l_output)
    
    m1_aligned, m2_aligned = [], []
    multipliers, mu_aligned = [], []
    for mu in range(2 * l_output + 1):
        for el in indices[mu]:
            m1, m2, multiplier = el
            m1_aligned.append(m1)
            m2_aligned.append(m2)
            multipliers.append(multiplier * 1.0)
            mu_aligned.append(mu)
    m1_aligned = torch.tensor(m1_aligned, dtype=torch.int64, device=device)
    m2_aligned = torch.tensor(m2_aligned, dtype=torch.int64, device=device)
    mu_aligned = torch.tensor(mu_aligned, dtype=torch.int64, device=device)
    multipliers = torch.tensor(multipliers, dtype=dtype, device=device)

    indices = np.argsort(mu_aligned)

    m1_aligned = m1_aligned[indices]
    m2_aligned = m2_aligned[indices]
    mu_aligned = mu_aligned[indices]
    multipliers = multipliers[indices]
  
    return m1_aligned, m2_aligned, mu_aligned, multipliers

class CGCalculatorSingle(torch.nn.Module):
    def __init__(self, l1, l2, l_output, dtype = torch.float32):
        super(CGCalculatorSingle, self).__init__()
        check_l_consistency(l1, l2, l_output)
        self.l1 = l1
        self.l2 = l2
        self.l_output = l_output
        m1, m2, mu, C = get_cg_transformation_rule(l1, l2, l_output, dtype = dtype)
        self.register_buffer('m1', m1)
        self.register_buffer('m2', m2)
        self.register_buffer('mu', mu)
        self.register_buffer('C', C)
        
    def forward(self, X1, X2):
        return accumulate(X1, X2, self.mu, 2 * self.l_output + 1, self.m1, self.m2, self.C)