from logging import log
from turtle import pd
import torch
import wandb
import numpy as np
import torch.autograd as autograd
import torch.utils.checkpoint as checkpoint
import pdb

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

# This method assumes that a single image is being inverted!
def compute_multi_step(xt, model, all_xT, et_coeff, et_prevsum_coeff, T, t, xT, image_dim, **kwargs):
    xt_in = xt[kwargs['next_idx']]
    # pdb.set_trace()
    # et = model(xt_in, t)
    et = checkpoint.checkpoint(model, xt_in, t)
    if et.shape[1] == 6:    
        et = et[:, :3, :, :]
    
    et_updated = et_coeff * et
    et_cumsum_all = et_updated.cumsum(dim=0)
    
    et_prevsum = et_cumsum_all

    xt_next = all_xT + et_prevsum_coeff * et_prevsum
    xt_all = torch.zeros_like(xt)
    xt_all[kwargs['xT_idx']] = xT
    xt_all[kwargs['prev_idx']] = xt_next
    
    xt_all = xt_all.cuda()
    # xt_all = xt_all.cpu()
    return xt_all

def compute_multi_step_sr(xt, model, all_xT, et_coeff1, et_coeff2, et_prevsum_coeff, T, t, xT, all_y, A, Ap, noise_coeff, all_noise, use_svd, **kwargs):
    xt_in = xt[kwargs['next_idx']]

    # et = model(xt_in, t)
    et = checkpoint.checkpoint(model, xt_in, t)
    if et.shape[1] == 6:    
        et = et[:, :3, :, :]
    
    if use_svd:
        ApA_et = Ap(A(et.reshape(et.size(0), -1))).reshape(*et.size())
    else:
        ApA_et = Ap(A(et))
    et_updated = (et_coeff2 - et_coeff1) * (et - (ApA_et))  
    et_cumsum_all = et_updated.cumsum(dim=0)
    et_prevsum = et_cumsum_all

    ApA_et_updated = et_coeff2 * ApA_et

    if use_svd:
        ApA_noise = Ap(A(all_noise.reshape(all_noise.size(0), -1))).reshape(*all_noise.size())
    else:
        ApA_noise = Ap(A(all_noise))

    ApA_noise_updated = noise_coeff * ApA_noise
    noise_updated = noise_coeff * all_noise
    noise_updated = noise_updated - ApA_noise_updated
    noise_cumsum_all = noise_updated.cumsum(dim=0)
    noise_prevsum = noise_cumsum_all

    xt_next = all_xT + et_prevsum_coeff * (et_prevsum + ApA_et_updated + ApA_noise_updated + noise_prevsum) + all_y 
    
    xt_all = torch.zeros_like(xt)

    xt_all[kwargs['xT_idx']] = xT
    xt_all[kwargs['prev_idx']] = xt_next
    
    xt_all = xt_all.cuda()

    return xt_all

def simple_anderson(f, x0, m=3, lam=1e-3, threshold=30, eps=1e-3, stop_mode='rel', beta=1.0, **kwargs):
    """ Anderson acceleration for fixed point iteration. """
    bsz, ch, h0, w0 = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)

    X[:,0], F[:,0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].reshape_as(x0)).reshape(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]

        ## This matrix should be invertible
        A = H[:,:n+1,:n+1]
        # D = torch.diag(A)
        # A = A + D
        alpha = torch.solve(y[:,:n+1], A)[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:,k%m] - X[:,k%m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:,k%m].norm().item())
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode: 
                    lowest_xest, lowest_gx =  X[:,k%m].view_as(x0).clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold-1-k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    X = F = None
    return lowest_xest

def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)

def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite

def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite

def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)   # (N, threshold)
    return -x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)    # (N, 2d, L'), but should really be (N, 1, (2d*L'))

def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     # (N, 2d, L'), but should really be (N, (2d*L'), 1)

def broyden(f, x0, threshold, eps=1e-12, stop_mode="abs", ls=False, layer_loss=True, layer_idx=[10], name="unknown"):
    dev = x0.device
    init_shape = x0.shape
    if len(x0.shape) == 4:
        init_shape = x0.shape
        bsz, C, H, W = x0.shape
        x0 = x0.view(bsz, -1, 1)
        total_hsize = C*H*W
        seq_len = 1
        Us = torch.zeros(bsz, total_hsize, seq_len, threshold).to(dev)     # One can also use an L-BFGS scheme to further reduce memory
        VTs = torch.zeros(bsz, threshold, total_hsize, seq_len).to(dev)
    elif len(x0.shape) == 3:
        init_shape = x0.shape
        bsz, total_hsize, seq_len = x0.size()

        # For fast calculation of inv_jacobian (approximately)
        Us = torch.zeros(bsz, total_hsize, seq_len, threshold).to(dev)     # One can also use an L-BFGS scheme to further reduce memory
        VTs = torch.zeros(bsz, threshold, total_hsize, seq_len).to(dev)
    else:
        raise ValueError(f"{x0.shape} not currently supported in Broyden iterations")

    if len(init_shape) == 4:
        g = lambda y: f(y.view(init_shape)).view(bsz, -1, 1) - y.view(bsz, -1, 1)
    else:
        g = lambda y: f(y.view(init_shape)) - y
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    
    x_est = x0           # (bsz, 2d, L')
    gx = g(x_est)        # (bsz, 2d, L')
    nstep = 0
    tnstep = 0
    

    update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx)      # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False
    
    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * seq_len
    new_objective = 1e8

    trace_dict = {'abs': [],
                'rel': []}
    lowest_dict = {'abs': 1e8,
                'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    nstep, lowest_xest, lowest_gx = 0, x_est, gx

    zm = []
    while nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)
        abs_diff = torch.norm(gx).item()
        rel_diff = abs_diff / (torch.norm(gx + x_est).item() + 1e-9)
        diff_dict = {'abs': abs_diff,
                    'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode: 
                    lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep
        if nstep in layer_idx:
            zm.append(lowest_xest.view(init_shape))

        new_objective = diff_dict[stop_mode]
        if new_objective < eps: break
        if new_objective < 3*eps and nstep > 30 and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:,:,:,:nstep-1], VTs[:,:nstep-1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:,None,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,nstep-1] = vT
        Us[:,:,:,nstep-1] = u
        update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx)

    if len(layer_idx) > 0 and nstep < layer_idx[-1]:
        while len(zm) < len(layer_idx):
            zm.append(lowest_xest.view(init_shape))

    # Fill everything up to the threshold length
    for _ in range(threshold+1-len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    return lowest_xest.view(init_shape)

# @torch.no_grad()
def anderson(f, x0, args, m=3, lam=1e-3, max_iter=50, tol=1e-3, beta = 1.0, logger=None):
    """ Anderson acceleration for fixed point iteration. """
    with torch.no_grad():
        bsz, ch, h0, w0 = x0.shape
        
        X = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, ch * h0 * w0, dtype=x0.dtype, device=x0.device)
        
        X[:,0] = x0.view(bsz, -1)
        F[:,0] = f(xt=x0.view(x0.shape), **args).view(bsz, -1)

        X[:,1] = F[:,0].view(bsz, -1)
        F[:,1] = f(xt=F[:,0].view(x0.shape), **args).view(bsz, -1)

        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1

        iter_count = 0
        log_metrics = {}
        res = []
        norm_res = []
        for k in range(2, max_iter):
            n_ = min(k, m)
            G = F[:,:n_]-X[:,:n_]
            
            H[:,1:n_+1,1:n_+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n_, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.solve(y[:,:n_+1], H[:,:n_+1,:n_+1])[0][:, 1:n_+1, 0]   # (bsz x n)
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n_])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n_])[:,0]
            
            F[:,k%m] = f(xt=X[:,k%m].view(x0.shape), **args).view(bsz, -1)

            residual = (F[:,k%m] - X[:,k%m]).norm().item()
            normalized_residual = (F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm()).item()

            res.append(residual)
            norm_res.append(normalized_residual)
            iter_count += 1

            if (norm_res[-1] < tol):
                print("Breaking out early at {}".format(k))
                break

            if logger is not None:
                log_metrics["residual"] = residual
                log_metrics["normalized_residual"] = normalized_residual

                log_metrics["alpha"] = torch.norm(alpha, dim=-1).mean()
                log_metrics["samples"] = [wandb.Image(X[:, k%m].view_as(x0).to('cpu')[ts]) for ts in args['plot_timesteps']]
                logger(log_metrics)
    x_eq = X[:,k%m].view_as(x0)#[args['gather_idx']].to('cpu')
    X = F = None
    # print("Abs residual ", min(res), " Rel residual ", min(norm_res))
    return x_eq

# x: source image
# model: diffusion model 
# args: required args
def find_source_noise(x, model, args, logger=None):
    T = args['T']
    at = args['at']
    at_next = args['at_next']
    alpha_ratio = args['alpha_ratio']
    _,C,H,W = x.shape

    xT = x[0].view(1, C, H, W)
    all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(x.device)
    
    et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

    et_coeff = (1 / at_next.sqrt()) * et_coeff2

    et_prevsum_coeff = at_next.sqrt()
    with torch.no_grad():
        for _ in range(T, -1, -1):
            all_xt = compute_multi_step(xt=x, model=model, et_coeff=et_coeff,
                            et_prevsum_coeff=et_prevsum_coeff, image_dim=x.shape, all_xT=all_xT, xT=xT, **args)
            if logger is not None:
                logger({"generated images": [wandb.Image(all_xt[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
            x = all_xt
    return all_xt

class DEQLatentSpaceOpt(object):
    def __init__(self):
        self.hook =  None

    def find_source_noise_deq_sr(self, x, y, model, args, anderson_params=None, tau=0.5, pg_steps=5, logger=None):
        T = args['T']
        at = args['at']
        at_next = args['at_next']
        alpha_ratio = args['alpha_ratio']
        Ap = args['Ap']
        A = args['A']
        use_svd = args['use_svd']
        nw = args['nw']
        etw = args['etw']
        _,C,H,W = x.shape

        if anderson_params is None:
            anderson_params = {
                "m": 3,
                "lambda": 1e-3,
                "max_anderson_iters": 15,
                "tol": 0.01,
                "beta": 1
            }
        
        xT = x[0].view(1, C, H, W)
        
        all_xT = torch.repeat_interleave(xT, T, dim=0).to(x.device)

        all_y = torch.repeat_interleave(y, T, dim=0).to(x.device)
        y_coeff = at_next.sqrt() 
        if use_svd:
            all_xT = alpha_ratio * (all_xT - Ap(A(all_xT.reshape(all_xT.size(0), -1))).reshape(*all_xT.size()))
            all_y = y_coeff * Ap(all_y.reshape(all_y.size(0), -1)).reshape(*all_xT.size()) 
        else:
            all_xT = alpha_ratio * (all_xT - Ap(A(all_xT))) #lambda_t* TODO
            all_y = y_coeff * Ap(all_y)      

        sigma_t = (1 - at_next**2).sqrt().to(all_xT.device)

        # nw = torch.tensor(nw).expand(T, 1, 1, 1).to(all_xT.device)  #TODO
        # etw = torch.tensor(etw).expand(T, 1, 1, 1).to(all_xT.device)  #TODO
        c1 = (1 - at_next).sqrt() * nw
        c2 = (1 - at_next).sqrt() * (1 - etw**2) ** 0.5 
        c3 = (((1 - at)*at_next)/at).sqrt()

        et_coeff1 = (1 / at_next.sqrt()) * c3
        if use_svd:
            et_coeff2 = (1 / at_next.sqrt()) * c2 #TODO
            noise_coeff = (1 / at_next.sqrt()) * c1 * sigma_t
        else:
            et_coeff2 = (1 / at_next.sqrt()) * c2 * sigma_t  #TODO
            noise_coeff = (1 / at_next.sqrt()) * c1 * sigma_t

        et_prevsum_coeff = at_next.sqrt()

        # all_noise = torch.randn_like(all_xT)
        all_noise = torch.repeat_interleave(torch.randn_like(xT), T, dim=0).to(x.device)
        
        args['model'] = model
        args['xT'] = xT
        args['all_xT'] = all_xT
        args['all_y'] = all_y

        args['et_coeff1'] = et_coeff1
        args['et_coeff2'] = et_coeff2
        args['et_prevsum_coeff'] = et_prevsum_coeff
        args['noise_coeff'] = noise_coeff
        args['all_noise'] = all_noise
        args['image_dim'] = x.shape
        
        with torch.no_grad():
            x_eq = anderson(compute_multi_step_sr, x, args, m=3, lam=1e-3, max_iter=15, tol=1e-2, beta=0.9, logger=None)
            
            if logger is not None:
                logger({"generated images": [wandb.Image(x_eq[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
        
        torch.cuda.empty_cache()

        x_eq.requires_grad_()
        
        for _ in range(pg_steps):
            update_x_eq = compute_multi_step_sr(xt=x_eq, **args)
            x_eq = (1 - tau) * x_eq + tau * update_x_eq #compute_multi_step(xt=x_eq, **args)
        return x_eq

    def find_source_noise_deq(self, x, model, args, anderson_params=None, tau=0.5, pg_steps=5, logger=None):
        T = args['T']
        at = args['at']
        at_next = args['at_next']
        alpha_ratio = args['alpha_ratio']
        _,C,H,W = x.shape

        if anderson_params is None:
            anderson_params = {
                "m": 3,
                "lambda": 1e-3,
                "max_anderson_iters": 15,
                "tol": 0.01,
                "beta": 1
            }
        xT = x[0].view(1, C, H, W)
        all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(x.device)

        et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

        et_coeff = (1 / at_next.sqrt()) * et_coeff2

        et_prevsum_coeff = at_next.sqrt()
        
        args['model'] = model
        args['xT'] = xT
        args['all_xT'] = all_xT
        args['et_coeff'] = et_coeff
        args['et_prevsum_coeff'] = et_prevsum_coeff
        args['image_dim'] = x.shape

        with torch.no_grad():
            x_eq = anderson(compute_multi_step, x, args, m=3, lam=1e-3, max_iter=15, tol=1e-2, beta = 0.9, logger=None)
            if logger is not None:
                logger({"generated images": [wandb.Image(x_eq[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
        
        torch.cuda.empty_cache()

        x_eq.requires_grad_()
        
        for _ in range(pg_steps):
            update_x_eq = compute_multi_step(xt=x_eq, **args)
            x_eq = (1 - tau) * x_eq + tau * update_x_eq #compute_multi_step(xt=x_eq, **args)
        return x_eq

    def find_source_noise_deq_ift(self, x, model, args, anderson_params=None, logger=None):
        T = args['T']
        at = args['at']
        at_next = args['at_next']
        alpha_ratio = args['alpha_ratio']
        _,C,H,W = x.shape

        if anderson_params is None:
            anderson_params = {
                "m": 3,
                "lambda": 1e-3,
                "max_anderson_iters": 15,
                "tol": 0.01,
                "beta": 1
            }
        xT = x[0].view(1, C, H, W)
        all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(x.device)

        et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()

        et_coeff = (1 / at_next.sqrt()) * et_coeff2

        et_prevsum_coeff = at_next.sqrt()
        
        args['model'] = model
        args['xT'] = xT
        args['all_xT'] = all_xT
        args['et_coeff'] = et_coeff
        args['et_prevsum_coeff'] = et_prevsum_coeff
        args['image_dim'] = x.shape

        with torch.no_grad():
            x_eq = anderson(compute_multi_step, x, args, m=3, lam=1e-3, max_iter=15, tol=1e-2, beta = 0.9, logger=None)
            if logger is not None:
                logger({"generated images": [wandb.Image(x_eq[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
        
        # torch.cuda.empty_cache()
        new_z1 = compute_multi_step(xt=x_eq.requires_grad_(), **args)
        def backward_hook(grad):
            if self.hook is not None:
                self.hook.remove()
                torch.cuda.synchronize()
            # result = simple_anderson(lambda y: autograd.grad(new_z1, x_eq, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), 
            #                         m=3, lam=1e-4, threshold=20, eps=1e-3, beta = 0.9, logger=None)
            result = broyden(lambda y: autograd.grad(new_z1, x_eq, y, retain_graph=True)[0] + grad, torch.zeros_like(grad), 
                        threshold=20, eps=1e-3)
                                    #lam=self.anderson_lam, m=self.anderson_m, threshold=self.b_thres, stop_mode=self.stop_mode, name="backward")
            return result

        self.hook = new_z1.register_hook(backward_hook)
        return new_z1

def get_additional_lt_opt_args(all_xt, seq, betas, batch_size):
    from functions.ddim_anderson import compute_alpha

    cur_seq = list(seq)
    seq_next = [-1] + list(seq[:-1])

    gather_idx = [idx for idx in range(len(cur_seq), len(all_xt), len(cur_seq)+2)]
    xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq)+1)]
    next_idx = [idx for idx in range(len(all_xt)) if idx not in gather_idx]
    prev_idx = [idx + 1 for idx in next_idx]

    T = len(cur_seq)
    t = torch.tensor(cur_seq[::-1]).repeat(batch_size).to(all_xt.device)
    next_t = torch.tensor(seq_next[::-1]).repeat(batch_size).to(all_xt.device)

    at = compute_alpha(betas, t.long())
    at_next = compute_alpha(betas, next_t.long())

    alpha_ratio = (at_next/at[0]).sqrt() 
    
    additional_args = {
        "T" : T, 
        "t" : t,
        "bz": batch_size,
        "gather_idx": gather_idx,
        "xT_idx": xT_idx,
        "prev_idx": prev_idx,
        "next_idx": next_idx,
        "alpha_ratio": alpha_ratio,
        'at_next': at_next,
        'at': at
    }
    return additional_args

def get_additional_lt_opt_args_sr(all_xt, seq, betas, batch_size, scale, lq, A, Ap, args):
    
    from functions.ddim_anderson import compute_alpha

    # seq = self.get_timestep_sequence_sr(start_timesteps)    #range(0, 500, 20)
    cur_seq = list(seq)                 #[0, 40, ..., 960]
    seq_next = [-1] + list(seq[:-1])    #[-1, 0, ..., 920]

    gather_idx = [idx for idx in range(len(cur_seq), len(all_xt), len(cur_seq)+2)]    #[24]
    xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq)+1)]   #[0]
    next_idx = [idx for idx in range(len(all_xt)) if idx not in gather_idx]
    prev_idx = [idx + 1 for idx in next_idx]    #[1, 2, ..., 24]

    T = len(cur_seq)    #25
    t = torch.tensor(cur_seq[::-1]).repeat(batch_size).to(all_xt.device)        #[960, 920, ...,  40,   0]
    next_t = torch.tensor(seq_next[::-1]).repeat(batch_size).to(all_xt.device)  #[920, 880, ...,   0,  -1]
    
    at = compute_alpha(betas, t.long())             #[0.0001, ..., 0.9999]
    at_next = compute_alpha(betas, next_t.long())   #[0.0002, ..., 1.0000]

    alpha_ratio = (at_next/at[0]).sqrt()            #[1.4627, ..., 106.9576]
    # all_xT = torch.repeat_interleave(xT, T, dim=0).to(all_xt.device)  #[25, 3, 256, 256]
    # if args.use_svd:
    #     all_xT = alpha_ratio * (all_xT - Ap(A(all_xT.reshape(all_xT.size(0), -1))).reshape(*all_xT.size()))
    # else:
    #     all_xT = alpha_ratio * (all_xT - Ap(A(all_xT))) #lambda_t* TODO

    et_prevsum_coeff = at_next.sqrt()   #[0.0137, ..., 1.0000]

    sigma_t = (1 - at_next**2).sqrt().to(all_xt.device) #sigma_t good for psnr
    sigma_t1 = (1 - at**2).sqrt().to(all_xt.device)
    sigma_t1[-1] = 0    #sigma_t1 good for lpips
    sigma_t2 = (1 - at**2).sqrt().to(all_xt.device)
    sigma_t3 = (1 - at**3).sqrt().sqrt().to(all_xt.device)
    sigma_t4 = (1 - at**3).sqrt().to(all_xt.device)
    sigma_t5 = torch.ones(sigma_t4.shape).to(all_xt.device)
    sigma_t5[-1] = 0
    sigma_t6 = (1 - at**4).sqrt().sqrt().to(all_xt.device)
    sigma_t7 = (1 - at**6).to(all_xt.device)
    sigma_t8 = (1 - at_next**2 - (at_next*0.01)**2).sqrt().to(all_xt.device)
    sigma_t8[-1] = 0

    # nw = torch.tensor(args.nw).expand(T, 1, 1, 1).to(all_xt.device)  #TODO
    # etw = torch.tensor(args.etw).expand(T, 1, 1, 1).to(all_xt.device)  #TODO

    c1 = (1 - at_next).sqrt() * args.nw
    c2 = (1 - at_next).sqrt() * (1 - args.etw**2)**0.5
    c3 = (((1 - at)*at_next)/at).sqrt()

    # et_coeff1 = (1 / at_next.sqrt()) * c3
    # if args.use_svd:
    #     et_coeff2 = (1 / at_next.sqrt()) * c2 #TODO
    # else:
    #     et_coeff2 = (1 / at_next.sqrt()) * c2 * sigma_t7  #TODO
    
    all_y = torch.repeat_interleave(lq, T, dim=0).to(all_xt.device)
    y_coeff = at_next.sqrt() 
    if args.use_svd:
        all_y = y_coeff * Ap(all_y.reshape(all_y.size(0), -1)).reshape(*all_xt[:all_y.size(0), :].size())  #all_xT 
    else:
        all_y = y_coeff * Ap(all_y) 

    # if args.use_svd:
    #     noise_coeff = (1 / at_next.sqrt()) * c1 * sigma_t
    # else:
    #     noise_coeff = (1 / at_next.sqrt()) * c1 * sigma_t7
    # all_noise = torch.randn_like(all_xt)    #all_xT 
    
    additional_args = {
        # "all_xT": all_xT, 
        "at": at,
        "at_next": at_next,
        "alpha_ratio": alpha_ratio,
        # "et_coeff1": et_coeff1,
        # "et_coeff2": et_coeff2,
        # "et_prevsum_coeff": et_prevsum_coeff, 
        "T" : T, 
        "t" : t,
        # "bz": batch_size,
        "gather_idx": gather_idx,
        "xT_idx": xT_idx,
        "prev_idx": prev_idx,
        "next_idx": next_idx,
        # "xT": xT,
        "nw": args.nw,
        "etw": args.etw,
        "A": A,
        "Ap": Ap,
        'sf': scale,
        # 'y_coeff': y_coeff,
        'all_y': all_y,
        # 'noise_coeff': noise_coeff,
        # 'all_noise': all_noise,
        'use_svd': args.use_svd
    }
    return additional_args

### Rest of the code is just for the sake of validation
def fp_validity_check(all_xt, model, additional_args, max_steps=100, logger=None):
    with torch.no_grad():
        all_xt = find_source_noise(all_xt, model, additional_args, logger=logger)
    return all_xt

def anderson_validity_check(all_xt, model, additional_args, max_steps=100, logger=None):
    deq_ls_opt = DEQLatentSpaceOpt()
    with torch.no_grad():
        all_xt = deq_ls_opt.find_source_noise_deq(all_xt, model, additional_args, logger=logger)
    return all_xt 