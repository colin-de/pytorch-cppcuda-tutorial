import torch
import cppcuda_tutorial
import time

def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats: (N, 8, F)
        points: (N, 3) local coordinates in [-1, 1]
    
    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp


class Trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points):
        feat_interp = cppcuda_tutorial.trilinear_interpolation_fw(feats, points)

        ctx.save_for_backward(feats, points)

        return feat_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp):
        feats, points = ctx.saved_tensors

        dL_dfeats = cppcuda_tutorial.trilinear_interpolation_bw(dL_dfeat_interp.contiguous(), feats, points)

        return dL_dfeats, None
    

if __name__ == '__main__':
    print('cuda is available: ', torch.cuda.is_available())
    print('cuda device count: ', torch.cuda.device_count())
    print('cuda device name: ', torch.cuda.get_device_name(0))
    print('cuda device capability: ', torch.cuda.get_device_capability(0))
    print('cuda device properties: ', torch.cuda.get_device_properties(0))
    print('cuda current device: ', torch.cuda.current_device())

    N = 65535; F = 256
    rand = torch.rand(N, 8, F, device='cuda')
    feats = rand.clone().requires_grad_()
    feats2 = rand.clone().requires_grad_()

    # feats = torch.rand(N, 8, F, device='cuda').requires_grad_()
    # feats2 = torch.rand(N, 8, F, device='cuda').requires_grad_()
    points = torch.rand(N, 3, device='cuda') * 2 - 1

    t = time.time()
    # out_cuda = cppcuda_tutorial.trilinear_interpolation_fw(feats, points)
    out_cuda = Trilinear_interpolation_cuda.apply(feats, points)

    torch.cuda.synchronize()
    print('cuda fw time', time.time()-t, 's')

    t = time.time()
    out_py = trilinear_interpolation_py(feats, points)
    torch.cuda.synchronize()
    print('pytorch fw time', time.time()-t, 's')

    print(out_cuda.shape)
    print(out_py.shape)

    print('fw all close', torch.allclose(out_py, out_cuda))

    t = time.time()
    loss2 = out_cuda.sum()
    loss2.backward()
    torch.cuda.synchronize()
    print('cuda bw time', time.time()-t, 's')

    t = time.time()
    loss = out_py.sum()
    loss.backward()
    torch.cuda.synchronize()
    print('pytorch bw time', time.time()-t, 's')
