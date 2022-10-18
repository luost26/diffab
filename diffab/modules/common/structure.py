import torch
from torch.nn import Module, Linear, LayerNorm, Sequential, ReLU

from ..common.geometry import compose_rotation_and_translation, quaternion_to_rotation_matrix, repr_6d_to_rotation_matrix


class FrameRotationTranslationPrediction(Module):

    def __init__(self, feat_dim, rot_repr, nn_type='mlp'):
        super().__init__()
        assert rot_repr in ('quaternion', '6d')
        self.rot_repr = rot_repr
        if rot_repr == 'quaternion':
            out_dim = 3 + 3
        elif rot_repr == '6d':
            out_dim = 6 + 3
        
        if nn_type == 'linear':
            self.nn = Linear(feat_dim, out_dim)
        elif nn_type == 'mlp':
            self.nn = Sequential(
                Linear(feat_dim, feat_dim), ReLU(),
                Linear(feat_dim, feat_dim), ReLU(),
                Linear(feat_dim, out_dim)
            )
        else:
            raise ValueError('Unknown nn_type: %s' % nn_type)

    def forward(self, x):
        y = self.nn(x)  # (..., d+3)
        if self.rot_repr == 'quaternion':
            quaternion = torch.cat([torch.ones_like(y[..., :1]), y[..., 0:3]], dim=-1)
            R_delta = quaternion_to_rotation_matrix(quaternion)
            t_delta = y[..., 3:6]
            return R_delta, t_delta
        elif self.rot_repr == '6d':
            R_delta = repr_6d_to_rotation_matrix(y[..., 0:6])
            t_delta = y[..., 6:9]
            return R_delta, t_delta


class FrameUpdate(Module):

    def __init__(self, node_feat_dim, rot_repr='quaternion', rot_tran_nn_type='mlp'):
        super().__init__()
        self.transition_mlp = Sequential(
            Linear(node_feat_dim, node_feat_dim), ReLU(),
            Linear(node_feat_dim, node_feat_dim), ReLU(),
            Linear(node_feat_dim, node_feat_dim),
        )
        self.transition_layer_norm = LayerNorm(node_feat_dim)

        self.rot_tran = FrameRotationTranslationPrediction(node_feat_dim, rot_repr, nn_type=rot_tran_nn_type)
    
    def forward(self, R, t, x, mask_generate):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3). Unit: Angstrom.
            x:  Node-wise features, (N, L, F).
            mask_generate:   Masks, (N, L).
        Returns:
            R': Updated basis matrices, (N, L, 3, 3_index).
            t': Updated coordinates, (N, L, 3).
        """
        x = self.transition_layer_norm(x + self.transition_mlp(x))

        R_delta, t_delta = self.rot_tran(x) # (N, L, 3, 3), (N, L, 3)
        R_new, t_new = compose_rotation_and_translation(R, t, R_delta, t_delta)

        mask_R = mask_generate[:, :, None, None].expand_as(R)
        mask_t = mask_generate[:, :, None].expand_as(t)

        R_new = torch.where(mask_R, R_new, R)
        t_new = torch.where(mask_t, t_new, t)

        return R_new, t_new
