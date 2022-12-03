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

def get_transformation(idx_output, output_size, X1_size, X2_size, idx_1, idx_2, multipliers):
    transformation = torch.zeros([X1_size, X2_size, output_size])
    for index in range(idx_output.shape[0]):
        transformation[idx_1[index], idx_2[index], idx_output[index]] = multipliers[index]
    transformation = transformation.reshape([-1, transformation.shape[2]])
    return transformation
    

def get_transformation_sparse(idx_output, output_size, X1_size, X2_size, idx_1, idx_2, multipliers):
    i_idx, j_idx, values = [], [], []
    for index in range(idx_output.shape[0]):
        i_idx.append(idx_1[index] * X2_size + idx_2[index])
        j_idx.append(idx_output[index])
        values.append(multipliers[index])
    transformation_sparse = torch.sparse_coo_tensor([j_idx, i_idx], values, (output_size, X1_size * X2_size))    
    return transformation_sparse


def sparse_accumulation_matrix_multiply(X1, X2, transformation):
   
    initial_0 = X1.shape[0]
    initial_1 = X1.shape[1]
    X = X1[:, :, :, None] * X2[:, :, None, :] #..., m1, m2
    first_dim = X.shape[0] * X.shape[1]
    second_dim = X.shape[2] * X.shape[3]
    X = X.reshape([first_dim, second_dim])
    output = torch.matmul(X, transformation)
    output = output.reshape([initial_0, initial_1, -1])
    return output


def sparse_accumulation_sparse_matrix_multiply(X1, X2, transformation_sparse):
   
    initial_0 = X1.shape[0]
    initial_1 = X1.shape[1]
    X = X1[:, :, :, None] * X2[:, :, None, :] #..., m1, m2
    first_dim = X.shape[0] * X.shape[1]
    second_dim = X.shape[2] * X.shape[3]
    X = X.reshape([first_dim, second_dim])
    second = X.T
    output = torch.matmul(transformation_sparse, second)
    output = output.T
    output = output.reshape([initial_0, initial_1, -1])
    return output
                                                                       
def sparse_accumulation_sparse_matrix_multiply_optimized(X1, X2, transformation_sparse):
    #initial_1 = X1.shape[1]
    #initial_2 = X1.shape[2]

    X = X1[:, None, :, :] * X2[None, :, :, :]
    first_dim = X.shape[0] * X.shape[1]
    second_dim = X.shape[2] * X.shape[3]   
    X = X.reshape([first_dim, second_dim])
    output = torch.matmul(transformation_sparse, X)
    return output                                                       