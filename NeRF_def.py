import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import special

# render method: C(o, d)=integrate(exp(-integrate(dense(o+s*d)*ds, ts, t)) * dense(o+t*d) * color(o+t*d, d) * dt, ts, te)
# apply hierarchical sampling (origin) or adaptive quadrature (scipy)

# Misc Function
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log10(x)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
max_order = 20
int_legendre = [torch.Tensor(np.array(special.roots_legendre(i))) for i in range(1, max_order + 1)]
int_laguerre = [torch.Tensor(np.array(special.roots_laguerre(i))) for i in range(1, max_order + 1)]


# Positional Embedding
# x -> (..., sin(2^i*x*pi), cos(2^i*x*pi), ...) to enhance high frequency part of the function
# add different embedding to different freq part
class Embedder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()  # embedding function

    def create_embedding_fn(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.kwargs.get('device'):
            self.device = self.kwargs.get('device')
        embed_fns = []
        d = self.kwargs.get('input_dims')
        out_dim = 0
        # embed as (x, embed(x))
        if self.kwargs.get('include_input'):
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs.get('max_freq_log2')
        N_freqs = self.kwargs.get('num_freqs')

        # log sampling or evenly sampling
        if self.kwargs.get('log_sampling'):
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs.get('periodic_fns'):
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        if self.kwargs.get('include_embedding'):
            self.learnt_embed = nn.Embedding(len(embed_fns), d)
            for i in range(len(embed_fns)):
                embed_fns[i] = lambda x, tmpf=embed_fns[i], embed=self.learnt_embed, i=i: tmpf(x) + embed(
                    torch.tensor(i, dtype=torch.long).to(self.device))

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(multires, device, i=0):
    # what is this???
    # i represent input dimensions
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'include_embedding': True,
        'device': device
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.forward(x)
    return embed, embedder_obj.out_dim, embedder_obj


# volume Representation Model
class NeRF(torch.nn.Module):
    # D layer, W neuro per layer, input 3 ordinates xyz, input view vector d, output rgb and absorption rate alpha
    # if use viewing direction, add d to input to predict the color(xyz, d)
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,
                 views_linear_depth=1):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        # position embedding network
        # skips contains layers that need residuals ??? what is skips meaning and effect
        self.pts_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch, W)] +
                                               [torch.nn.Linear(W, W) if i not in skips else torch.nn.Linear(
                                                   W + input_ch, W) for i in range(D - 1)])
        # viewing embedding network, its scale shrinks because the rank of input_views is 2 < 3 ???
        # can be substituted with a shallow 1-layer linear net
        self.views_linears_depth = views_linear_depth
        self.views_linears = torch.nn.ModuleList([torch.nn.Linear(input_ch_views + W, W // 2)] +
                                                 [torch.nn.Linear(W // 2, W // 2) for i in
                                                  range(self.views_linears_depth - 1)])
        if use_viewdirs:
            self.feature_linear = torch.nn.Linear(W, W)
            self.alpha_linear = torch.nn.Linear(W, 1)
            self.rgb_linear = torch.nn.Linear(W // 2, 3)
        else:
            self.output_linear = torch.nn.Linear(W, output_ch)

    def forward(self, x):
        # divide the last dimension to pts & views
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            # reconcatenate xyz with feature h for the sake of residual
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        # alpha is only affected by pts, cat pts's feature with views encoding for colors
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    # load NeRF weight from .npz, with only shallow one-layer view linear layer
    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# H represents range of y, W represents range of x, K represents resolution of pic, c2w represents orthograhic transform for camera to world
# the function returns the corresponding ray_o, ray_d of all pixels, returns world coordinates
def get_rays(H, W, K, c2w):
    # first calc ordinates of each meshgrid (x, y ,z)
    # the indexing sequence of pytorch meshgrid is ij, while the numpy meshgrid is xy
    grid_i, grid_j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    grid_i = grid_i.t()  # the same as grid_i.transpose(0, 1)
    grid_j = grid_j.t()
    dirs = torch.stack([(grid_i - K[0][2]) / K[0][0], -(grid_j - K[1][2]) / K[1][1], -torch.ones_like(grid_i)], dim=-1)
    # Rotate ray directions
    # rays_d = [c2w.dot(dir) for dir in dirs]
    # dot (i, j, -1) with (rx,ry,rz) to get the new cord of rays' directions
    # also can be written as torch.einsum('ijk,lk->ijl', dirs, c2w[:3,:3])
    rays_d = torch.einsum('ijk,lk->ijl', dirs, c2w[:3, :3])
    # rays_d = torch.sum(dirs[..., np.newaxis, :]*c2w[:3,:3],dim=-1)
    # translate operation
    # also can be written as torch.einsum('i, jk->jki', c2w[:3,-1], torch.ones_like(rays_d))
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    # substitue torch with np , the same as get_rays
    # first calc ordinates of each meshgrid (x, y ,z)
    grid_i, grid_j = np.meshgrid(np.linspace(0, W - 1, W, dtype=np.float32), np.linspace(0, H - 1, H, dtype=np.float32),
                                 indexing='xy')
    dirs = np.stack([(grid_i - K[0][2]) / K[0][0], -(grid_j - K[1][2]) / K[1][1], -np.ones_like(grid_i)], axis=-1)
    # Rotate ray directions
    # rays_d = [c2w.dot(dir) for dir in dirs]
    # dot (i, j, -1) with (rx,ry,rz) to get the new cord of rays' directions
    # also can be written as np.einsum('ijk,lk->ijl', dirs, c2w[:3,:3])
    # rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    rays_d = np.einsum('ijk,lk->ijl', dirs, c2w[:3, :3])
    # translate operation
    # also can be written as np.einsum('i, jk->jki', c2w[:3,-1], np.ones_like(rays_d))
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


'''
c2w=[ 
R   t
  
0   1 
] = [
I   t
0   1
] * [
R   0
0   1
]

Note that it maybe inappropriate to perform ndc transformation:
We assume that norm2(dx, dy, dz) = 1, then s represents the Euclid distance
t' = 1-oz/(oz+s*dz)   # s represents Euclid ratio, saying r = o + s*d -> o' + t'*d'
d'x = -n/r * (dx/dz-ox/oz)
d'y = -n/t * (dy/dz-oy/oz)
d'z = 2nf/(n-f) / oz 
set f -> inf we got
d'z = -2n / oz

In normal case, we get integral in ndc space, for s' from 1-oz/(oz+near*dz) to 1-oz/(oz+far*dz) , other cases may be dismissed
if oz = -0 ->  d'z = inf, d'x = n/r*ox, d'y = n/t*oy, integral calculated directly for s in [near, far] 
if dz = 0 -> d'z = 0, d'x = -n/r*dx, d'y = -n/t*dy, integral calculated directly for s in [near, far]
 
1. the corresponding rendering integral is numerical equal to integrate(c'(o'+s'd', d'/||d'||) * d(1 - exp(-integrate(a(o'+t'd')/dz/(1-t')^2 * dt'))))
2. shift ray origins to the near plane -near will results in different upper & lower of integral, need to adjust the corresponding bounds
3. focal is equal to -near, which is redundant. What if focal != -near ???
4. what if rays_o[...,2], the same as oz = -near is equal to zero ???
5. what if rays_d[...,2], the same as dz is equal to zero ??? in this case rays are in xy plane
 '''


def ndc_rays(H, W, focal, near, rays_o, rays_d, isShift=False):
    if isShift:
        # shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]  # rays_d[...,2] = dz maybe 0 ???
        rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def fixed_integrate_tensor(network_fn, network_query_fn, rays_o, rays_d, viewdirs, lower, upper,
                           N_samples=200,
                           white_bkgd=False, raw_noise_std=0,
                           weight='1'):
    '''
    Calculate integral c*a*exp(-integrate(a*dt))*ds from lower to upper
        or c*a*exp(-integrate(a*dt/(1-t)**2/d_z))*ds/(1-s)**2/d_z in ndc space from near to far
    :param network_query_fn: network_query_fn(pts, views, net) -->  raw:[num_rays, num_samples along ray, rgba] prediction from NeRF
    :param lower:[num_rays, 1] lower times distance bound of integral
    :param upper:[num_rays, 1] upper times distance bound of integral
    :param N_samples: number of sampled points. The same as interpolation points. Must be divisible by max_order!!!
    :param weight: Weight of integral. w(x) = 1
    :return:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    '''
    '''
# use composite rule and Guassian rule to find interpolation points and weights in tensor
# return integrals and errs in tensor
# for example:ff  = lambda s, o = o, d = d: F.silu(torch.einsum('i, ...k->...ik', s, d) + o[...,None,:].expand(tuple(np.concatenate([np.array(o.shape[:-1]), [s.shape[0], o.shape[-1]]]))))
# fa = lambda s, o=o, d=d: rgba=NeRF(torch.einsum('...ij,...ik->...ijk',s,d) + torch.einsum('...ij,...ik->...ijk',torch.ones_like(s),o), torch.einsum('...ij,...ik->...ijk',torch.ones_like(s),d))
                           c = rgba[...,:3], a = rgba[...,3:]
                           T = torch.cumsum(a, dim=-2)

# the overall assumption:
# s & w: [...,i,j], o & d: [...,i,k], meaning ray_i, intp_j, rgba_k
# s * d = einsum('...ij,...ik->...ijk', s,d);  w * f(s) = einsum('...ij, ...ijk->...ik', w, f(s))
# s & w need to expand to list(d.shape[:-1]) + [max_interp_num]
# the key is to calculate the interpolation points s of all different rays
# the intp. depends on the integral method, the method depends on the lower and upper and error of integral, surplus intp. is set to 1e-17 and w is set to 0.
# Concretely, unbouned -> bounded + laguerre
              bounded  -> legendre + composite
    '''
    # no ndc:
    # bounded case: integrate(f, a, b) = (b-a)/2*integrate(f((b-a)/2*x+(a+b)/2), -1,1) -> (b-a)/2*f(legendre_roots*(b-a)/2+(a+b)/2)
    # composite num: N_samples//max_order -> N_samples, order of roots: max_order -> 1
    # First calc all the subregions
    comp_num = N_samples // max_order
    order = max_order
    if comp_num == 0:
        comp_num = 1
        order = N_samples
    mids = torch.linspace(0, 1, steps=comp_num + 1)
    mids = mids * (upper - lower) + lower  # [num_rays, comp_num+1]
    rays_norm = rays_d.norm(dim=-1)  # [num_rays]
    mids = mids * rays_norm[..., None]
    # Second calc the intp pts, weights and intervals
    dists = mids[..., 1:] - mids[..., :-1]  # [num_rays, comp_num]
    centres = (mids[..., 1:] + mids[..., :-1]) / 2
    # s_1, w_1 = int_legendre[order] # in different device err !!!
    s_1, w_1 = special.roots_legendre(order)
    s_1, w_1 = torch.Tensor(s_1), torch.Tensor(w_1)
    s = s_1 * dists[..., None] / 2 + centres[..., None]  # [num_rays, comp_num, intp_num]
    w_dist = torch.cat([s, mids[..., 1:, None]], dim=-1) - torch.cat([mids[..., :-1, None], s], dim=-1)
    w_dist = w_dist[..., :-1]  # [num_rays, comp_num, intp_num]
    s = torch.reshape(s, list(s.shape[:-2]) + [-1])  # [num_rays, N_samples]
    w = w_1 * dists[..., None] / 2
    w = torch.reshape(w, list(w.shape[:-2]) + [-1])  # [num_rays, N_samples]
    pts = rays_o[..., None, :] + viewdirs[..., None, :] * s[..., :, None]  # [num_rays, N_samples, xyz]
    # Third calc the f vals: c*a*exp(-integrate(a*dt))
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, rgb]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
    alpha = F.relu(raw[..., 3]) + noise # [N_rays, N_samples]
    # estimate integrate(a*dt) by legendre and dist rule then cumsum it together
    # cumsum by dist rule in every region
    alpha_inner_int = torch.cumsum(w_dist * torch.reshape(alpha, w_dist.shape), dim=-1)  # [N_rays, comp_num, intp_num]
    # cumsum by legendre across regions
    alpha_outter_int = torch.cumsum(
        torch.sum(torch.reshape(w, w_dist.shape) * torch.reshape(alpha, w_dist.shape), dim=-1),
        dim=-1)  # [N_rays, comp_num]
    alpha_int = alpha_inner_int
    alpha_int[..., 1:, :] += alpha_outter_int[..., :-1, None]
    T = torch.reshape(torch.exp(-alpha_int), alpha.shape)
    weights = T*alpha*w # [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, rgb]

    depth_map = torch.sum(weights * s, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)+1e-10))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map, raw

    # unbounded case: integrate(f*exp(-x), a, inf) = exp(-a)*integrate(f(x+a)*exp(-x), 0,inf) -> exp(-a)*f(laguerre_roots+a)
    # integrals = torch.sum(torch.where(torch.eq(upper, torch.inf), f(roots_laguerre.r)*roots_laguerre.w, f(roots_legendre)*roots_legendre), -1)
