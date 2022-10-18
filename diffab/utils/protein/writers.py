import torch
import warnings
from Bio import BiopythonWarning
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder

from .constants import AA, restype_to_heavyatom_names


def save_pdb(data, path=None):
    """
    Args:
        data:   A dict that contains: `chain_nb`, `chain_id`, `aa`, `resseq`, `icode`,
                `pos_heavyatom`, `mask_heavyatom`.
    """

    def _mask_select(v, mask):
        if isinstance(v, str):
            return ''.join([s for i, s in enumerate(v) if mask[i]])
        elif isinstance(v, list):
            return [s for i, s in enumerate(v) if mask[i]]
        elif isinstance(v, torch.Tensor):
            return v[mask]
        else:
            return v

    def _build_chain(builder, aa_ch, pos_heavyatom_ch, mask_heavyatom_ch, chain_id_ch, resseq_ch, icode_ch):
        builder.init_chain(chain_id_ch[0])
        builder.init_seg('    ')

        for aa_res, pos_allatom_res, mask_allatom_res, resseq_res, icode_res in \
            zip(aa_ch, pos_heavyatom_ch, mask_heavyatom_ch, resseq_ch, icode_ch):
            if not AA.is_aa(aa_res.item()): 
                print('[Warning] Unknown amino acid type at %d%s: %r' % (resseq_res.item(), icode_res, aa_res.item()))
                continue
            restype = AA(aa_res.item())
            builder.init_residue(
                resname = str(restype),
                field = ' ',
                resseq = resseq_res.item(),
                icode = icode_res,
            )

            for i, atom_name in enumerate(restype_to_heavyatom_names[restype]):
                if atom_name == '': continue    # No expected atom
                if (~mask_allatom_res[i]).any(): continue     # Atom is missing
                if len(atom_name) == 1: fullname = ' %s  ' % atom_name
                elif len(atom_name) == 2: fullname = ' %s ' % atom_name
                elif len(atom_name) == 3: fullname = ' %s' % atom_name
                else: fullname = atom_name # len == 4
                builder.init_atom(atom_name, pos_allatom_res[i].tolist(), 0.0, 1.0, ' ', fullname,)

    warnings.simplefilter('ignore', BiopythonWarning)
    builder = StructureBuilder()
    builder.init_structure(0)
    builder.init_model(0)

    unique_chain_nb = data['chain_nb'].unique().tolist()
    for ch_nb in unique_chain_nb:
        mask = (data['chain_nb'] == ch_nb)
        aa = _mask_select(data['aa'], mask)
        pos_heavyatom = _mask_select(data['pos_heavyatom'], mask)
        mask_heavyatom = _mask_select(data['mask_heavyatom'], mask)
        chain_id = _mask_select(data['chain_id'], mask)
        resseq = _mask_select(data['resseq'], mask)
        icode = _mask_select(data['icode'], mask)

        _build_chain(builder, aa, pos_heavyatom, mask_heavyatom, chain_id, resseq, icode)
    
    structure = builder.get_structure()
    if path is not None:
        io = PDBIO()
        io.set_structure(structure)
        io.save(path)
    return structure
