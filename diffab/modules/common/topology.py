import torch
import torch.nn.functional as F


def get_consecutive_flag(chain_nb, res_nb, mask):
    """
    Args:
        chain_nb, res_nb
    Returns:
        consec: A flag tensor indicating whether residue-i is connected to residue-(i+1), 
                BoolTensor, (B, L-1)[b, i].
    """
    d_res_nb = (res_nb[:, 1:] - res_nb[:, :-1]).abs()   # (B, L-1)
    same_chain = (chain_nb[:, 1:] == chain_nb[:, :-1])
    consec = torch.logical_and(d_res_nb == 1, same_chain)
    consec = torch.logical_and(consec, mask[:, :-1])
    return consec


def get_terminus_flag(chain_nb, res_nb, mask):
    consec = get_consecutive_flag(chain_nb, res_nb, mask)
    N_term_flag = F.pad(torch.logical_not(consec), pad=(1, 0), value=1)
    C_term_flag = F.pad(torch.logical_not(consec), pad=(0, 1), value=1)
    return N_term_flag, C_term_flag
