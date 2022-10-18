import torch

from ._base import _mask_select_data, register_transform
from ..protein import constants


@register_transform('patch_around_anchor')
class PatchAroundAnchor(object):

    def __init__(self, initial_patch_size=128, antigen_size=128):
        super().__init__()
        self.initial_patch_size = initial_patch_size
        self.antigen_size = antigen_size

    def _center(self, data, origin):
        origin = origin.reshape(1, 1, 3)
        data['pos_heavyatom'] -= origin # (L, A, 3)
        data['pos_heavyatom'] = data['pos_heavyatom'] * data['mask_heavyatom'][:, :, None]
        data['origin'] = origin.reshape(3)
        return data

    def __call__(self, data):        
        anchor_flag = data['anchor_flag']   # (L,)
        anchor_points = data['pos_heavyatom'][anchor_flag, constants.BBHeavyAtom.CA]    # (n_anchors, 3)
        antigen_mask = (data['fragment_type'] == constants.Fragment.Antigen)
        antibody_mask = torch.logical_not(antigen_mask)

        if anchor_flag.sum().item() == 0:
            # Generating full antibody-Fv, no antigen given
            data_patch = _mask_select_data(
                data = data,
                mask = antibody_mask,
            )
            data_patch = self._center(
                data_patch,
                origin = data_patch['pos_heavyatom'][:, constants.BBHeavyAtom.CA].mean(dim=0)
            )
            return data_patch

        pos_alpha = data['pos_heavyatom'][:, constants.BBHeavyAtom.CA]  # (L, 3)
        dist_anchor = torch.cdist(pos_alpha, anchor_points).min(dim=1)[0]    # (L, )
        initial_patch_idx = torch.topk(
            dist_anchor,
            k = min(self.initial_patch_size, dist_anchor.size(0)),
            largest=False,
        )[1]   # (initial_patch_size, )

        dist_anchor_antigen = dist_anchor.masked_fill(
            mask = antibody_mask, # Fill antibody with +inf
            value = float('+inf')
        )   # (L, )
        antigen_patch_idx = torch.topk(
            dist_anchor_antigen, 
            k = min(self.antigen_size, antigen_mask.sum().item()), 
            largest=False, sorted=True
        )[1]    # (ag_size, )
        
        patch_mask = torch.logical_or(
            data['generate_flag'],
            data['anchor_flag'],
        )
        patch_mask[initial_patch_idx] = True
        patch_mask[antigen_patch_idx] = True

        patch_idx = torch.arange(0, patch_mask.shape[0])[patch_mask]

        data_patch = _mask_select_data(data, patch_mask)
        data_patch = self._center(
            data_patch,
            origin = anchor_points.mean(dim=0)
        )
        data_patch['patch_idx'] = patch_idx
        return data_patch
