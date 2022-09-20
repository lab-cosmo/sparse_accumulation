import torch

def sparse_accumulation_loops(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers):
    device = X1.device #all tensors must be on the same device and blah, blah, blah    
    output = torch.zeros([X1.shape[0], X2.shape[1], output_size], device = device)
    for index in range(idx_output.shape[0]):
        #output[:, :, idx_output[index]] += torch.mul(X1[:, :, idx_1[index]], X2[:, :, idx_2[index]]) * multipliers[index]
        output[:, :, idx_output[index]] += X1[:, :, idx_1[index]] * X2[:, :, idx_2[index]] * multipliers[index]
    return output
                                                                       
                                                                       
def sparse_accumulation_index_add(X1, X2, idx_output, output_size, idx_1, idx_2, multipliers):
    contributions = X1[:, :, idx_1] * X2[:, :, idx_2] * multipliers[None, None, :]
    device = X1.device #all tensors must be on the same device and blah, blah, blah    
    output = torch.zeros([X1.shape[0], X2.shape[1], output_size], device = device)
    output.index_add_(2, idx_output, contributions)
    return output
       
    

                                                                       
                                                                       