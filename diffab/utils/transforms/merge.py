import torch

from ..protein import constants
from ._base import register_transform


@register_transform('merge_chains')
class MergeChains(object):

    def __init__(self):
        super().__init__()

    def assign_chain_number_(self, data_list):
        chains = set()
        for data in data_list:
            chains.update(data['chain_id'])
        chains = {c: i for i, c in enumerate(chains)}

        for data in data_list:
            data['chain_nb'] = torch.LongTensor([
                chains[c] for c in data['chain_id']
            ])

    def _data_attr(self, data, name):
        if name in ('generate_flag', 'anchor_flag') and name not in data:
            return torch.zeros(data['aa'].shape, dtype=torch.bool)
        else:
            return data[name]

    def __call__(self, structure):
        data_list = []
        if structure['heavy'] is not None:
            structure['heavy']['fragment_type'] = torch.full_like(
                structure['heavy']['aa'],
                fill_value = constants.Fragment.Heavy,
            )
            data_list.append(structure['heavy'])

        if structure['light'] is not None:
            structure['light']['fragment_type'] = torch.full_like(
                structure['light']['aa'],
                fill_value = constants.Fragment.Light,
            )
            data_list.append(structure['light'])

        if structure['antigen'] is not None:
            structure['antigen']['fragment_type'] = torch.full_like(
                structure['antigen']['aa'],
                fill_value = constants.Fragment.Antigen,
            )
            structure['antigen']['cdr_flag'] = torch.zeros_like(
                structure['antigen']['aa'],
            )
            data_list.append(structure['antigen'])

        self.assign_chain_number_(data_list)

        list_props = {
            'chain_id': [],
            'icode': [],
        }
        tensor_props = {
            'chain_nb': [],
            'resseq': [],
            'res_nb': [],
            'aa': [],
            'pos_heavyatom': [],
            'mask_heavyatom': [],
            'generate_flag': [],
            'cdr_flag': [],
            'anchor_flag': [],
            'fragment_type': [],
        }

        for data in data_list:
            for k in list_props.keys():
                list_props[k].append(self._data_attr(data, k))
            for k in tensor_props.keys():
                tensor_props[k].append(self._data_attr(data, k))

        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        data_out = {
            **list_props,
            **tensor_props,
        }
        return data_out

