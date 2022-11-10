import time

import ase.io
import rascaline
import torch
import torch.utils.cpp_extension
from equistore import TensorBlock, TensorMap

from clebsch_gordan import ClebschGordan, get_real_clebsch_gordan
from sparse_accumulation_plain_torch import sparse_accumulation_index_add

torch.utils.cpp_extension.load(
    name="sparse_accumulation_cuda",
    sources=["cuda_optimized/sparse_accumulation_cuda_kernel2D.cu"],
    is_python_module=False,
    extra_cuda_cflags=None,
    verbose=True,
)

L_MAX = 8
HYPERS = {
    "cutoff": 3.5,
    "max_radial": 10,
    "max_angular": L_MAX - 1,
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "center_atom_weight": 1.0,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
}


def descriptor_to_torch(descriptor, device, dtype):
    blocks = []
    for _, block in descriptor:
        new_block = TensorBlock(
            values=torch.tensor(block.values, device=device, dtype=dtype),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            new_block.add_gradient(
                parameter,
                data=torch.tensor(block.values, device=device, dtype=dtype),
                samples=gradient.samples,
                components=gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(descriptor.keys, blocks)


def descriptor_to_torch_list(descriptor, device, dtype):
    data = []
    for (l,), block in descriptor:
        data.append(((l,), torch.tensor(block.values, device=device, dtype=dtype)))
    return data


def descriptor_to_torch_list_active_dim_last(descriptor, device, dtype):
    data = []
    for (l,), block in descriptor:
        values = torch.tensor(block.values, device=device, dtype=dtype)
        data.append(((l,), values.swapaxes(1, 2).contiguous()))
    return data


def create_real_data(path, selection):
    frames = ase.io.read(path, selection)

    calculator = rascaline.SphericalExpansion(**HYPERS)
    descriptor = calculator.compute(frames)
    descriptor.keys_to_samples("species_center")
    descriptor.keys_to_properties("species_neighbor")
    return descriptor


def generate_cg_multipliers(device, dtype):
    precomputed = {}
    clebsch = ClebschGordan(L_MAX).precomputed_

    for l1 in range(L_MAX):
        for l2 in range(L_MAX):
            for l in range(L_MAX):
                indices = get_real_clebsch_gordan(clebsch[l1, l2, l], l1, l2, l)
                m1_aligned = []
                m2_aligned = []
                mu_aligned = []
                multipliers = []

                for mu in range(0, 2 * l + 1):
                    for el in indices[mu]:
                        m1, m2, multiplier = el
                        m1_aligned.append(m1)
                        m2_aligned.append(m2)
                        multipliers.append(multiplier)
                        mu_aligned.append(mu)

                m1 = torch.tensor(m1_aligned, device=device, dtype=torch.int64)
                m2 = torch.tensor(m2_aligned, device=device, dtype=torch.int64)
                mu = torch.tensor(mu_aligned, device=device, dtype=torch.int64)

                multipliers = torch.tensor(multipliers, device=device, dtype=dtype)

                precomputed[(l1, l2, l)] = (m1, m2, mu, multipliers)

    return precomputed


def run_cg_combine(function, x1, x2, precomputed_cg):
    for (l1,), spx1 in x1:
        for (l2,), spx2 in x2:
            for l in range(L_MAX):
                m1, m2, mu, multipliers = precomputed_cg[(l1, l2, l)]
                output = function(spx1, spx2, mu, 2 * l + 1, m1, m2, multipliers)


def bench_cg_combine(function, x1, x2, precomputed_cg, n_iters=10):
    run_cg_combine(function, x1, x2, precomputed_cg)

    start = time.time()
    for _ in range(n_iters):
        run_cg_combine(function, x1, x2, precomputed_cg)

    elapsed = time.time() - start
    return elapsed / n_iters


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float64

    print(f"\n\nrunning on {device=}, {dtype=}")

    # descriptor = create_real_data("molecular_crystals.xyz", ":30")
    descriptor = create_real_data("random-methane-10k.extxyz", ":300")

    precomputed_cg = generate_cg_multipliers(device, dtype)

    print(f"done loading data\n")

    def index_add_impl(x1, x2, mu, size, m1, m2, multipliers):
        return sparse_accumulation_index_add(
            x1, x2, mu, size, m1, m2, multipliers, active_dim=1
        )

    x = descriptor_to_torch_list(descriptor, device, dtype)
    timing = bench_cg_combine(index_add_impl, x, x, precomputed_cg)
    print(f"index_add: {1e3 * timing:.5} ms")

    x = descriptor_to_torch_list_active_dim_last(descriptor, device, dtype)
    timing = bench_cg_combine(
        torch.ops.sparse_accumulation_cuda.forward, x, x, precomputed_cg
    )
    print(f"Custom CUDA: {1e3 * timing:.5} ms")
