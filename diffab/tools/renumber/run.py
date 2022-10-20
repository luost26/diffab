import argparse
import abnumber
from Bio import PDB
from Bio.PDB import Model, Chain, Residue, Selection
from Bio.Data import SCOPData
from typing import List, Tuple


def biopython_chain_to_sequence(chain: Chain.Chain):
    residue_list = Selection.unfold_entities(chain, 'R')
    seq = ''.join([SCOPData.protein_letters_3to1.get(r.resname, 'X') for r in residue_list])
    return seq, residue_list


def assign_number_to_sequence(seq):
    abchain = abnumber.Chain(seq, scheme='chothia')
    offset = seq.index(abchain.seq)
    if not (offset >= 0):
        raise ValueError(
            'The identified Fv sequence is not a subsequence of the original sequence.'
        )

    numbers = [None for _ in range(len(seq))]
    for i, (pos, aa) in enumerate(abchain):
        resseq = pos.number
        icode = pos.letter if pos.letter else ' '
        numbers[i+offset] = (resseq, icode)
    return numbers, abchain


def renumber_biopython_chain(chain_id, residue_list: List[Residue.Residue], numbers: List[Tuple[int, str]]):
    chain = Chain.Chain(chain_id)
    for residue, number in zip(residue_list, numbers):
        if number is None:
            continue
        residue = residue.copy()
        new_id = (residue.id[0], number[0], number[1])
        residue.id = new_id
        chain.add(residue)
    return chain


def renumber(in_pdb, out_pdb, return_other_chains=False):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(None, in_pdb)
    model = structure[0]
    model_new = Model.Model(0)

    heavy_chains, light_chains, other_chains = [], [], []

    for chain in model:
        try:
            seq, reslist = biopython_chain_to_sequence(chain)
            numbers, abchain = assign_number_to_sequence(seq)
            chain_new = renumber_biopython_chain(chain.id, reslist, numbers)
            print(f'[INFO] Renumbered chain {chain_new.id} ({abchain.chain_type})')
            if abchain.chain_type == 'H':
                heavy_chains.append(chain_new.id)
            elif abchain.chain_type in ('K', 'L'):
                light_chains.append(chain_new.id)
        except abnumber.ChainParseError as e:
            print(f'[INFO] Chain {chain.id} does not contain valid Fv: {str(e)}')
            chain_new = chain.copy()
            other_chains.append(chain_new.id)
        model_new.add(chain_new)

    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(model_new)
    pdb_io.save(out_pdb)
    if return_other_chains:
        return heavy_chains, light_chains, other_chains
    else:
        return heavy_chains, light_chains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_pdb', type=str)
    parser.add_argument('out_pdb', type=str)
    args = parser.parse_args()

    renumber(args.in_pdb, args.out_pdb)

if __name__ == '__main__':
    main()
