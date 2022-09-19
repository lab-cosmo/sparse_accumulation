import torch
import sparse_accumulation_cpp

class SparseAccumulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X1, X2, idx_output, output_size, idx_1, idx_2):
        output = sparse_accumulation_cpp.forward(X1, X2, idx_output, output_size, idx_1, idx_2)
        ctx.save_for_backward(*[X1, X2, idx_output, idx_1, idx_2])
        return output
        

    @staticmethod
    def backward(ctx, grad_output):
        X1, X2, idx_output, idx_1, idx_2 = ctx.saved_tensors
        d_X1, d_X2 = sparse_accumulation_cpp.backward(grad_output, X1, X2, idx_output, idx_1, idx_2)
        return d_X1, d_X2, None, None, None, None