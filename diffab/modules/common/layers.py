import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def clampped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes) # (N, L)
    x = x.clamp(min=0, max=num_classes-1)
    y = F.one_hot(x, num_classes) * mask[...,None]  # (N, L, C)
    return y


class DistanceToBins(nn.Module):

    def __init__(self, dist_min=0.0, dist_max=20.0, num_bins=64, use_onehot=False):
        super().__init__()
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.num_bins = num_bins
        self.use_onehot = use_onehot

        if use_onehot:
            offset = torch.linspace(dist_min, dist_max, self.num_bins)
        else:
            offset = torch.linspace(dist_min, dist_max, self.num_bins-1)    # 1 overflow flag
            self.coeff = -0.5 / ((offset[1] - offset[0]) * 0.2).item() ** 2  # `*0.2`: makes it not too blurred
        self.register_buffer('offset', offset)

    @property
    def out_channels(self):
        return self.num_bins 

    def forward(self, dist, dim, normalize=True):
        """
        Args:
            dist:   (N, *, 1, *)
        Returns:
            (N, *, num_bins, *)
        """
        assert dist.size()[dim] == 1
        offset_shape = [1] * len(dist.size())
        offset_shape[dim] = -1

        if self.use_onehot:
            diff = torch.abs(dist - self.offset.view(*offset_shape))  # (N, *, num_bins, *)
            bin_idx = torch.argmin(diff, dim=dim, keepdim=True)  # (N, *, 1, *)
            y = torch.zeros_like(diff).scatter_(dim=dim, index=bin_idx, value=1.0)
        else:
            overflow_symb = (dist >= self.dist_max).float()  # (N, *, 1, *)
            y = dist - self.offset.view(*offset_shape)  # (N, *, num_bins-1, *)
            y = torch.exp(self.coeff * torch.pow(y, 2))  # (N, *, num_bins-1, *)
            y = torch.cat([y, overflow_symb], dim=dim)  # (N, *, num_bins, *)
            if normalize:
                y = y / y.sum(dim=dim, keepdim=True)

        return y


class PositionalEncoding(nn.Module):
    
    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_funcs-1, num_funcs))
    
    def get_out_dim(self, in_dim):
        return in_dim * (2 * self.num_funcs + 1)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


class AngularEncoding(nn.Module):

    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', torch.FloatTensor(
            [i+1 for i in range(num_funcs)] + [1./(i+1) for i in range(num_funcs)]
        ))

    def get_out_dim(self, in_dim):
        return in_dim * (1 + 2*2*self.num_funcs)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


class LayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-10):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super().__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )
