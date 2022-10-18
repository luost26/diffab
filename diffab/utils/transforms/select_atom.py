
from ._base import register_transform


@register_transform('select_atom')
class SelectAtom(object):

    def __init__(self, resolution):
        super().__init__()
        assert resolution in ('full', 'backbone')
        self.resolution = resolution

    def __call__(self, data):
        if self.resolution == 'full':
            data['pos_atoms'] = data['pos_heavyatom'][:, :]
            data['mask_atoms'] = data['mask_heavyatom'][:, :]
        elif self.resolution == 'backbone':
            data['pos_atoms'] = data['pos_heavyatom'][:, :5]
            data['mask_atoms'] = data['mask_heavyatom'][:, :5]
        return data
