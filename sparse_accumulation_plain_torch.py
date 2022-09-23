import torch

def sparse_accumulation_loops(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers, active_dim):
    device = X1.device #all tensors must be on the same device and blah, blah, blah    
    dtype = X1.dtype
    
    if active_dim == 0:
        output = torch.zeros([output_size, X1.shape[1], X2.shape[2]], device = device,dtype=dtype)
        for index in range(idx_output.shape[0]):       
            output[idx_output[index], :, :] += X1[idx_1[index], :, :] * X2[idx_2[index], :, :] * multipliers[index]
        return output
    
    if active_dim == 1:
        output = torch.zeros([X1.shape[0], output_size, X2.shape[2]], device = device,dtype=dtype)
        for index in range(idx_output.shape[0]):
            output[:, idx_output[index], :] += X1[:, idx_1[index], :] * X2[:, idx_2[index], :] * multipliers[index] 
        return output
        
    if active_dim == 2:
        output = torch.zeros([X1.shape[0], X2.shape[1], output_size], device = device,dtype=dtype)
        for index in range(idx_output.shape[0]):       
            output[:, :, idx_output[index]] += X1[:, :, idx_1[index]] * X2[:, :, idx_2[index]] * multipliers[index]
        return output
    
    raise ValueError("active dim should be one of 0, 1, 2")
                                                                       
                                                                       
def sparse_accumulation_index_add(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers, active_dim):
    device = X1.device #all tensors must be on the same device and blah, blah, blah   
    dtype = X1.dtype
    
    if active_dim == 0:
        contributions = X1[idx_1, :, :] * X2[idx_2, :, :] * multipliers[:, None, None]
        output = torch.zeros([output_size, X1.shape[1], X2.shape[2]], device = device,dtype=dtype)
        output.index_add_(0, idx_output, contributions)
        return output
    
    if active_dim == 1:
        contributions = X1[:, idx_1, :] * X2[:, idx_2, :] * multipliers[None, :, None]
        output = torch.zeros([X1.shape[0], output_size, X2.shape[2]], device = device,dtype=dtype)
        output.index_add_(1, idx_output, contributions)
        return output
    
    if active_dim == 2:
        contributions = X1[:, :, idx_1] * X2[:, :, idx_2] * multipliers[None, None, :]     
        output = torch.zeros([X1.shape[0], X2.shape[1], output_size], device = device,dtype=dtype)
        output.index_add_(2, idx_output, contributions)
        return output
    
    raise ValueError("active dim should be one of 0, 1, 2")
       
    

                                                                       
                                                                       