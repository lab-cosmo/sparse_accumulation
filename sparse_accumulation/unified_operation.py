import torch
import sparse_accumulation_active_dim_last_cpp
import sparse_accumulation_cuda

def check_all_contiguous(tensors):   
    for tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError("all the tensors must be contiguous")
            

def check_all_on_cpu(tensors):
    for tensor in tensors:
        if str(tensor.device) != 'cpu':
            raise ValueError("all the tensors must be on cpu")
          

def check_all_on_cuda(tensors):
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError("all the tensors must be on cuda gpu")
            
            
def check_all_on_same_device(tensors):
    if len(tensors) == 0:
        return
    device = tensors[0].get_device()
    for tensor in tensors:
        if tensor.get_device() != device:
            raise ValueError("all the tensors must be on the same device")

def accumulate(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers):
    tensors = [X1, X2, idx_output, idx_1, idx_2, multipliers]
    check_all_on_same_device(tensors)
    
    if X1.is_cuda:
        return SparseAccumulationCUDA.apply(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers)
    else:
        return SparseAccumulationCPU.apply(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers) 
    

class SparseAccumulationCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X1, X2, idx_output, output_size, idx_1, idx_2, multipliers):
        tensors = [X1, X2, idx_output, idx_1, idx_2, multipliers]
        check_all_on_cuda(tensors)
        check_all_on_same_device(tensors)
        check_all_contiguous(tensors)  
        
        output = sparse_accumulation_cuda.forward(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers)[0]
        ctx.save_for_backward(*[X1, X2, idx_output, idx_1, idx_2, multipliers])
        return output
        

    @staticmethod
    def backward(ctx, grad_output):
        X1, X2, idx_output, idx_1, idx_2, multipliers = ctx.saved_tensors
        if idx_output.requires_grad or idx_1.requires_grad or idx_2.requires_grad:
            raise ValueError("can not compute gradients with respect to tensors with integers")
            
        if multipliers.requires_grad:
            raise ValueError("gradients with respect to multipliers (tensor named C in the documentastion) are not supported")
            
        check_all_on_cuda(ctx.saved_tensors)
        check_all_contiguous(ctx.saved_tensors)                                                                              
        check_all_on_same_device(ctx.saved_tensors)
        
        
        d_X1, d_X2 = sparse_accumulation_cuda.backward(grad_output, X1, X2, idx_output, idx_1, idx_2, multipliers)
        
        if not X1.requires_grad:
            d_X1 = None
        if not X2.requires_grad:
            d_X2 = None
        
        return d_X1, d_X2, None, None, None, None, None
    
    
class SparseAccumulationCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X1, X2, idx_output, output_size, idx_1, idx_2, multipliers):
        tensors = [X1, X2, idx_output, idx_1, idx_2, multipliers]
        check_all_on_cpu(tensors)
        
        # we have implementation for non-contiguous arrays, but it is so slow that I (SP) think that 
        # it is better to force the user to make the tensors contiguous
        check_all_contiguous(tensors)  
        
        output = sparse_accumulation_active_dim_last_cpp.forward_contiguous(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers)
        ctx.save_for_backward(*[X1, X2, idx_output, idx_1, idx_2, multipliers])
        return output
        

    @staticmethod
    def backward(ctx, grad_output):
        X1, X2, idx_output, idx_1, idx_2, multipliers = ctx.saved_tensors
        if idx_output.requires_grad or idx_1.requires_grad or idx_2.requires_grad:
            raise ValueError("can not compute gradients with respect to tensors with integers")
            
        if multipliers.requires_grad:
            raise ValueError("gradients with respect to multipliers (tensor named C in the documentastion) are not supported")
            
        check_all_on_cpu(ctx.saved_tensors)
        
        # we have implementation for non-contiguous arrays, but it is so slow that I (SP) think that 
        # it is better to force the user to make the tensors contiguous
        check_all_contiguous(ctx.saved_tensors)                                                                              
      
        d_X1, d_X2 = sparse_accumulation_active_dim_last_cpp.backward_contiguous(grad_output, X1, X2, idx_output, idx_1, idx_2, multipliers)
        
        if not X1.requires_grad:
            d_X1 = None
        if not X2.requires_grad:
            d_X2 = None
        
        return d_X1, d_X2, None, None, None, None, None