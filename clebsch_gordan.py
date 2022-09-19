from sympy.physics.wigner import clebsch_gordan
from sympy import S
import numpy as np


def _sympy_compute(l1, l2, l, m1, m2):
    return float(clebsch_gordan(S(l1), S(l2), S(l), S(m1), S(m2), S(m1 + m2)))


class ClebschGordan:
    def __init__(self, l_max):
        self.l_max_ = l_max
        self.precomputed_ = np.zeros(
            [l_max + 1, l_max + 1, l_max + 1, 2 * l_max + 1, 2 * l_max + 1])

        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                for l in range(l_max + 1):
                    for m1 in range(-l_max, l_max + 1):
                        for m2 in range(-l_max, l_max + 1):
                            now = _sympy_compute(l1, l2, l, m1, m2)
                            self.precomputed_[l1, l2, l, m1 + l1,
                                              m2 + l2] = now


def _multiply(first, second, multiplier):
    return [first[0], second[0], first[1] * second[1] * multiplier]


def _multiply_sequence(sequence, multiplier):
    result = []

    for el in sequence:
        #print(el)
        #print(len(el))
        result.append([el[0], el[1], el[2] * multiplier])
    return result


def _get_conversion(l, m):
    if (m < 0):
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
            if (np.abs(multiplier) > epsilon):
                result.append([m1, m2, multiplier])
    #print(len(sequence), '->', len(result))
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
            real_now.append(_multiply(X1_im, X2_im,
                                      -clebsch[m1 + l1, m2 + l2]))

            imag_now.append(_multiply(X1_re, X2_im, clebsch[m1 + l1, m2 + l2]))
            imag_now.append(_multiply(X1_im, X2_re, clebsch[m1 + l1, m2 + l2]))
        #print(real_now)
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