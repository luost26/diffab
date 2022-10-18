import torch
from .protein import constants


def find_cdrs(structure):
    cdrs = []
    if structure['heavy'] is not None:
        flag = structure['heavy']['cdr_flag']
        if int(constants.CDR.H1) in flag:
            cdrs.append('H_CDR1')
        if int(constants.CDR.H2) in flag:
            cdrs.append('H_CDR2')
        if int(constants.CDR.H3) in flag:
            cdrs.append('H_CDR3')

    if structure['light'] is not None:
        flag = structure['light']['cdr_flag']
        if int(constants.CDR.L1) in flag:
            cdrs.append('L_CDR1')
        if int(constants.CDR.L2) in flag:
            cdrs.append('L_CDR2')
        if int(constants.CDR.L3) in flag:
            cdrs.append('L_CDR3')
    
    return cdrs


def get_residue_first_last(data):
    loop_flag = data['generate_flag']
    loop_idx = torch.arange(loop_flag.size(0))[loop_flag]
    idx_first, idx_last = loop_idx.min().item(), loop_idx.max().item()
    residue_first = (data['chain_id'][idx_first], data['resseq'][idx_first].item(), data['icode'][idx_first])
    residue_last = (data['chain_id'][idx_last], data['resseq'][idx_last].item(), data['icode'][idx_last])
    return residue_first, residue_last


class RemoveNative(object):

    def __init__(self, remove_structure, remove_sequence):
        super().__init__()
        self.remove_structure = remove_structure
        self.remove_sequence = remove_sequence

    def __call__(self, data):
        generate_flag = data['generate_flag'].clone()
        if self.remove_sequence:
            data['aa'] = torch.where(
                generate_flag, 
                torch.full_like(data['aa'], fill_value=int(constants.AA.UNK)),    # Is loop
                data['aa']
            )

        if self.remove_structure:
            data['pos_heavyatom'] = torch.where(
                generate_flag[:, None, None].expand(data['pos_heavyatom'].shape),
                torch.randn_like(data['pos_heavyatom']) * 10,
                data['pos_heavyatom']
            )

        return data