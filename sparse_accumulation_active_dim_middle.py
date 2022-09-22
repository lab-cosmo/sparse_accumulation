import torch
import sparse_accumulation_active_dim_middle_cpp

class SparseAccumulationActiveDimMiddle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X1, X2, idx_output, output_size, idx_1, idx_2, multipliers):
        all_contiguous = X1.is_contiguous() and X2.is_contiguous() and idx_output.is_contiguous() and idx_1.is_contiguous() and idx_2.is_contiguous() and multipliers.is_contiguous()
        if all_contiguous:
            output = sparse_accumulation_active_dim_middle_cpp.forward_contiguous(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers)
        else:
            output = sparse_accumulation_active_dim_middle_cpp.forward(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers)
        ctx.save_for_backward(*[X1, X2, idx_output, idx_1, idx_2, multipliers])
        return output
        

    @staticmethod
    def backward(ctx, grad_output):        
        X1, X2, idx_output, idx_1, idx_2, multipliers = ctx.saved_tensors
        all_contiguous = X1.is_contiguous() and X2.is_contiguous() and idx_output.is_contiguous() and idx_1.is_contiguous() and idx_2.is_contiguous() and multipliers.is_contiguous()
        if all_contiguous:
            d_X1, d_X2 = sparse_accumulation_active_dim_middle_cpp.backward_contiguous(grad_output, X1, X2, idx_output, idx_1, idx_2, multipliers)
        else:
            d_X1, d_X2 = sparse_accumulation_active_dim_middle_cpp.backward(grad_output, X1, X2, idx_output, idx_1, idx_2, multipliers)
        return d_X1, d_X2, None, None, None, None, None