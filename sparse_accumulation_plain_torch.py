import torch

def sparse_accumulation_loops(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers, active_dimension_first = False):
    device = X1.device #all tensors must be on the same device and blah, blah, blah    
    
    if not active_dimension_first:
        output = torch.zeros([X1.shape[0], X2.shape[1], output_size], device = device)
        for index in range(idx_output.shape[0]):       
            output[:, :, idx_output[index]] += X1[:, :, idx_1[index]] * X2[:, :, idx_2[index]] * multipliers[index]
        return output
    else:
        output = torch.zeros([output_size, X1.shape[1], X2.shape[2]], device = device)
        for index in range(idx_output.shape[0]):       
            output[idx_output[index], :, :] += X1[idx_1[index], :, :] * X2[idx_2[index], :, :] * multipliers[index]
        return output
                                                                       
                                                                       
def sparse_accumulation_index_add(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers, active_dimension_first = False):
    device = X1.device #all tensors must be on the same device and blah, blah, blah   
    
    if not active_dimension_first:
        contributions = X1[:, :, idx_1] * X2[:, :, idx_2] * multipliers[None, None, :]     
        output = torch.zeros([X1.shape[0], X2.shape[1], output_size], device = device)
        output.index_add_(2, idx_output, contributions)
        return output
    else:
        contributions = X1[idx_1, :, :] * X2[idx_2, :, :] * multipliers[:, None, None]
        output = torch.zeros([output_size, X1.shape[1], X2.shape[2]], device = device)
        output.index_add_(0, idx_output, contributions)
        return output
       
    

                                                                       
                                                                       