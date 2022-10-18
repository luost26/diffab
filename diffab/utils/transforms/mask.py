import torch
import random
from typing import List, Optional

from ..protein import constants
from ._base import register_transform


def random_shrink_extend(flag, min_length=5, shrink_limit=1, extend_limit=2):
    first, last = continuous_flag_to_range(flag)
    length = flag.sum().item()
    if (length - 2*shrink_limit) < min_length:
        shrink_limit = 0
    first_ext = max(0, first-random.randint(-shrink_limit, extend_limit))
    last_ext = min(last+random.randint(-shrink_limit, extend_limit), flag.size(0)-1)
    flag_ext = flag.clone()
    flag_ext[first_ext : last_ext+1] = True
    return flag_ext


def continuous_flag_to_range(flag):
    first = (torch.arange(0, flag.size(0))[flag]).min().item()
    last = (torch.arange(0, flag.size(0))[flag]).max().item()
    return first, last


@register_transform('mask_single_cdr')
class MaskSingleCDR(object):

    def __init__(self, selection=None, augmentation=True):
        super().__init__()
        cdr_str_to_enum = {
            'H1': constants.CDR.H1,
            'H2': constants.CDR.H2,
            'H3': constants.CDR.H3,
            'L1': constants.CDR.L1,
            'L2': constants.CDR.L2,
            'L3': constants.CDR.L3,
            'H_CDR1': constants.CDR.H1,
            'H_CDR2': constants.CDR.H2,
            'H_CDR3': constants.CDR.H3,
            'L_CDR1': constants.CDR.L1,
            'L_CDR2': constants.CDR.L2,
            'L_CDR3': constants.CDR.L3,
            'CDR3': 'CDR3',     # H3 first, then fallback to L3
        }
        assert selection is None or selection in cdr_str_to_enum
        self.selection = cdr_str_to_enum.get(selection, None)
        self.augmentation = augmentation

    def perform_masking_(self, data, selection=None):
        cdr_flag = data['cdr_flag']

        if selection is None:
            cdr_all = cdr_flag[cdr_flag > 0].unique().tolist()
            cdr_to_mask = random.choice(cdr_all)
        else:
            cdr_to_mask = selection

        cdr_to_mask_flag = (cdr_flag == cdr_to_mask)
        if self.augmentation:
            cdr_to_mask_flag = random_shrink_extend(cdr_to_mask_flag)

        cdr_first, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
        left_idx = max(0, cdr_first-1)
        right_idx = min(data['aa'].size(0)-1, cdr_last+1)
        anchor_flag = torch.zeros(data['aa'].shape, dtype=torch.bool)
        anchor_flag[left_idx] = True
        anchor_flag[right_idx] = True

        data['generate_flag'] = cdr_to_mask_flag
        data['anchor_flag'] = anchor_flag

    def __call__(self, structure):
        if self.selection is None:
            ab_data = []
            if structure['heavy'] is not None:
                ab_data.append(structure['heavy'])
            if structure['light'] is not None:
                ab_data.append(structure['light'])
            data_to_mask = random.choice(ab_data)
            sel = None
        elif self.selection in (constants.CDR.H1, constants.CDR.H2, constants.CDR.H3, ):
            data_to_mask = structure['heavy']
            sel = int(self.selection)
        elif self.selection in (constants.CDR.L1, constants.CDR.L2, constants.CDR.L3, ):
            data_to_mask = structure['light']
            sel = int(self.selection)
        elif self.selection == 'CDR3':
            if structure['heavy'] is not None:
                data_to_mask = structure['heavy']
                sel = constants.CDR.H3
            else:
                data_to_mask = structure['light']
                sel = constants.CDR.L3

        self.perform_masking_(data_to_mask, selection=sel)
        return structure


@register_transform('mask_multiple_cdrs')
class MaskMultipleCDRs(object):
    
    def __init__(self, selection: Optional[List[str]]=None, augmentation=True):
        super().__init__()
        cdr_str_to_enum = {
            'H1': constants.CDR.H1,
            'H2': constants.CDR.H2,
            'H3': constants.CDR.H3,
            'L1': constants.CDR.L1,
            'L2': constants.CDR.L2,
            'L3': constants.CDR.L3,
            'H_CDR1': constants.CDR.H1,
            'H_CDR2': constants.CDR.H2,
            'H_CDR3': constants.CDR.H3,
            'L_CDR1': constants.CDR.L1,
            'L_CDR2': constants.CDR.L2,
            'L_CDR3': constants.CDR.L3,
        }
        if selection is not None:
            self.selection = [cdr_str_to_enum[s] for s in selection]
        else:
            self.selection = None
        self.augmentation = augmentation

    def mask_one_cdr_(self, data, cdr_to_mask):
        cdr_flag = data['cdr_flag']

        cdr_to_mask_flag = (cdr_flag == cdr_to_mask)
        if self.augmentation:
            cdr_to_mask_flag = random_shrink_extend(cdr_to_mask_flag)

        cdr_first, cdr_last = continuous_flag_to_range(cdr_to_mask_flag)
        left_idx = max(0, cdr_first-1)
        right_idx = min(data['aa'].size(0)-1, cdr_last+1)
        anchor_flag = torch.zeros(data['aa'].shape, dtype=torch.bool)
        anchor_flag[left_idx] = True
        anchor_flag[right_idx] = True

        if 'generate_flag' not in data:
            data['generate_flag'] = cdr_to_mask_flag
            data['anchor_flag'] = anchor_flag
        else:
            data['generate_flag'] |= cdr_to_mask_flag
            data['anchor_flag'] |= anchor_flag

    def mask_for_one_chain_(self, data):
        cdr_flag = data['cdr_flag']
        cdr_all = cdr_flag[cdr_flag > 0].unique().tolist()
    
        num_cdrs_to_mask = random.randint(1, len(cdr_all))

        if self.selection is not None:
            cdrs_to_mask = list(set(cdr_all).intersection(self.selection))
        else:
            random.shuffle(cdr_all)
            cdrs_to_mask = cdr_all[:num_cdrs_to_mask]

        for cdr_to_mask in cdrs_to_mask:
            self.mask_one_cdr_(data, cdr_to_mask)

    def __call__(self, structure):
        if structure['heavy'] is not None:
            self.mask_for_one_chain_(structure['heavy'])
        if structure['light'] is not None:
            self.mask_for_one_chain_(structure['light'])
        return structure

        
@register_transform('mask_antibody')
class MaskAntibody(object):

    def mask_ab_chain_(self, data):
        data['generate_flag'] = torch.ones(data['aa'].shape, dtype=torch.bool)

    def __call__(self, structure):
        pos_ab_alpha = []
        if structure['heavy'] is not None:
            self.mask_ab_chain_(structure['heavy'])
            pos_ab_alpha.append(
                structure['heavy']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            )
        if structure['light'] is not None:
            self.mask_ab_chain_(structure['light'])
            pos_ab_alpha.append(
                structure['light']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            )
        pos_ab_alpha = torch.cat(pos_ab_alpha, dim=0)   # (L_Ab, 3)

        if structure['antigen'] is not None:
            pos_ag_alpha = structure['antigen']['pos_heavyatom'][:, constants.BBHeavyAtom.CA]
            ag_ab_dist = torch.cdist(pos_ag_alpha, pos_ab_alpha)    # (L_Ag, L_Ab)
            nn_ab_dist = ag_ab_dist.min(dim=1)[0]   # (L_Ag)
            contact_flag = (nn_ab_dist <= 6.0)      # (L_Ag)
            if contact_flag.sum().item() == 0:
                contact_flag[nn_ab_dist.argmin()] = True

            anchor_idx = torch.multinomial(contact_flag.float(), num_samples=1).item()
            anchor_flag = torch.zeros(structure['antigen']['aa'].shape, dtype=torch.bool)
            anchor_flag[anchor_idx] = True
            structure['antigen']['anchor_flag'] = anchor_flag
            structure['antigen']['contact_flag'] = contact_flag
        
        return structure


@register_transform('remove_antigen')
class RemoveAntigen:

    def __call__(self, structure):
        structure['antigen'] = None
        structure['antigen_seqmap'] = None
        return structure
