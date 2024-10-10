import torch


def _matrix_reduction_step(matrix: torch.tensor):
    b, h, w = matrix.shape
    assert h > 1 and w > 2
    matrix = matrix.detach().clone()
    # ratio.shape = b,h-1
    ratio = matrix[:,:-1,0] / matrix[:,-1,0].unsqueeze(-1)
    matrix[:,:-1,:] = matrix[:,:-1,:] / ratio.unsqueeze(-1)
    matrix = matrix[:,:-1,:] - matrix[:,-1,:].unsqueeze(-2)
    return matrix[:,:,1:]


def matrix_reduction(matrix: torch.tensor):
    a = matrix
    while(a.shape[1] > 1 and a.shape[2] > 2):
        a = _matrix_reduction_step(a)
    return a


def reduce_and_get_ratio(matrix):
    if matrix.shape[-1] == 1:
        return matrix, None
    reduced_mat = matrix_reduction(matrix)
    # reduced_mat.shape = b, h, w
    if reduced_mat.shape[2] > 2:
        return reduced_mat, None
    mat = matrix.detach().clone()
    ratio = -reduced_mat[:,0,1] / reduced_mat[:,0,0]
    mat[:,:,-2] = mat[:,:,-2] * ratio.unsqueeze(-1)
    mat[:,:,-2] = mat[:,:,-2] + mat[:,:,-1]
    mat = mat[:,:,:-1]
    return mat, ratio


def multi_matrix_reslover(matrix: torch.tensor):
    matrix = matrix.detach().clone()
    b, c, h, w = matrix.shape  
    matrix = matrix.reshape(b*c, h, w)
    mat, ratio = reduce_and_get_ratio(matrix)
    if ratio is None:
        # mini_reslove.shape = b, c, mini_w
        mini_reslove = multi_matrix_reslover(mat.reshape(b, 1, c, -1))
        matrix = matrix.reshape(b, c, h, w)
        matrix[:,:,:,-mini_reslove.shape[-1]-1:-1] *= mini_reslove.unsqueeze(-2)
        matrix[:,:,:,-mini_reslove.shape[-1]-1] = matrix[:,:,:,-mini_reslove.shape[-1]-1:].sum(-1)
        matrix = matrix[:,:,:,:-mini_reslove.shape[-1]]
        results = torch.cat([mini_reslove] * c, dim=1).reshape(b*c,-1)
        mat = matrix.reshape(b*c, h, -1)
    else:
        results = ratio.unsqueeze(-1)
    while(True):
        mat, ratio = reduce_and_get_ratio(mat)
        if ratio is None:
            break
        results = torch.cat((ratio.unsqueeze(-1), results), dim=-1)
    return results.reshape(b, c, -1)


def get_homography(x, y):
    """
    get homography from sets of 4 key-points.
    """
    # x.shape = b, 4, 2
    # y.shape = b, 4, 2
    device = x.device
    b, _, _ = x.shape
    # sel-dim=epochs,4,3
    x = torch.cat((x, torch.ones(b, 4, 1, device=device)), dim=-1)
    y = torch.cat((y, torch.ones(b, 4, 1, device=device)), dim=-1)
    # matrix.shape = b, 2, h, w
    matrix = torch.zeros(b, 2, 4, 6, device=device)
    # 6/w, b, 4/h -> b, h, w -> b, 2, h, w
    matrix[:,0] = torch.stack([x[:,:,0],x[:,:,1],x[:,:,2],
                               -y[:,:,0]*x[:,:,0],-y[:,:,0]*x[:,:,1],-y[:,:,0]*x[:,:,2]])\
                             .permute(1, 2, 0)
    matrix[:,1] = torch.stack([x[:,:,0],x[:,:,1],x[:,:,2],
                               -y[:,:,1]*x[:,:,0],-y[:,:,1]*x[:,:,1],-y[:,:,1]*x[:,:,2]])\
                             .permute(1, 2, 0)
    # mat_result.shape = b, 2, w
    mat_result = multi_matrix_reslover(matrix)
    # 3/first, b, 3/second -> b, 3/first, 3/second
    h = torch.stack([mat_result[:,0,:3],
                     mat_result[:,1,:3],
                     torch.cat((mat_result[:,0,3:], torch.ones(b,1, device=device)), dim=-1)]).permute(1, 0, 2)
    return h


def distance_vectors(homographies, x, y):
    """
    return the distance between estimates of y by x and homographies to real y
    """
    device = x.device
    # x,y -> n,2 -> n,3 -> 3,n
    x = torch.cat((x, torch.ones(*x.shape[:-1], 1, device=device)), dim=-1).to(device).transpose(1,0)
    y = torch.cat((y, torch.ones(*y.shape[:-1], 1, device=device)), dim=-1).to(device).transpose(1,0)
    # shape=epochs,3,n
    estimates = homographies @ x
    # 3,epochs,n
    estimates = estimates.permute(1,0,2)
    estimates /= estimates[2,:,:]
    estimates[estimates!=estimates] = 0.
    # epochs,n,3
    estimates = estimates.permute(1,2,0)
    axis_distance = estimates - y.transpose(1,0)
    return torch.norm(axis_distance, dim=-1)
