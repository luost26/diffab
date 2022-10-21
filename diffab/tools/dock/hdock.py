import os
import shutil
import tempfile
import subprocess
import dataclasses as dc
from typing import List, Optional
from Bio import PDB
from Bio.PDB import Model as PDBModel

from diffab.tools.renumber import renumber as renumber_chothia
from .base import DockingEngine


def fix_docked_pdb(pdb_path):
    fixed = []
    with open(pdb_path, 'r') as f:
        for ln in f.readlines():
            if (ln.startswith('ATOM') or ln.startswith('HETATM')) and len(ln) == 56:
                fixed.append( ln[:-1] + ' 1.00  0.00              \n' )
            else:
                fixed.append(ln)
    with open(pdb_path, 'w') as f:
        f.write(''.join(fixed))


class HDock(DockingEngine):

    def __init__(
        self, 
        hdock_bin='./bin/hdock',
        createpl_bin='./bin/createpl',
    ):
        super().__init__()
        self.hdock_bin = os.path.realpath(hdock_bin)
        self.createpl_bin = os.path.realpath(createpl_bin)
        self.tmpdir = tempfile.TemporaryDirectory()

        self._has_receptor = False
        self._has_ligand = False

        self._receptor_chains = []
        self._ligand_chains = []

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.tmpdir.cleanup()

    def set_receptor(self, pdb_path):
        shutil.copyfile(pdb_path, os.path.join(self.tmpdir.name, 'receptor.pdb'))
        self._has_receptor = True

    def set_ligand(self, pdb_path):
        shutil.copyfile(pdb_path, os.path.join(self.tmpdir.name, 'ligand.pdb'))
        self._has_ligand = True

    def _dump_complex_pdb(self):
        parser = PDB.PDBParser(QUIET=True)
        model_receptor = parser.get_structure(None, os.path.join(self.tmpdir.name, 'receptor.pdb'))[0]
        docked_pdb_path = os.path.join(self.tmpdir.name, 'ligand_docked.pdb')
        fix_docked_pdb(docked_pdb_path)
        structure_ligdocked = parser.get_structure(None, docked_pdb_path)

        pdb_io = PDB.PDBIO()
        paths = []
        for i, model_ligdocked in enumerate(structure_ligdocked):
            model_complex = PDBModel.Model(0)
            for chain in model_receptor:
                model_complex.add(chain.copy())
            for chain in model_ligdocked:
                model_complex.add(chain.copy())
            pdb_io.set_structure(model_complex)
            save_path = os.path.join(self.tmpdir.name, f"complex_{i}.pdb")
            pdb_io.save(save_path)
            paths.append(save_path)
        return paths

    def dock(self):
        if not (self._has_receptor and self._has_ligand):
            raise ValueError('Missing receptor or ligand.')
        subprocess.run(
            [self.hdock_bin, "receptor.pdb", "ligand.pdb"],
            cwd=self.tmpdir.name, check=True
        )
        subprocess.run(
            [self.createpl_bin, "Hdock.out", "ligand_docked.pdb"], 
            cwd=self.tmpdir.name, check=True
        )
        return self._dump_complex_pdb()


@dc.dataclass
class DockSite:
    chain: str
    resseq: int


class HDockAntibody(HDock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heavy_chain_id = None
        self._epitope_sites: Optional[List[DockSite]] = None

    def set_ligand(self, pdb_path):
        raise NotImplementedError('Please use set_antibody')
    
    def set_receptor(self, pdb_path):
        raise NotImplementedError('Please use set_antigen')

    def set_antigen(self, pdb_path, epitope_sites: Optional[List[DockSite]]=None):
        super().set_receptor(pdb_path)
        self._epitope_sites = epitope_sites

    def set_antibody(self, pdb_path):
        heavy_chains, _ = renumber_chothia(pdb_path, os.path.join(self.tmpdir.name, 'ligand.pdb'))
        self._has_ligand = True
        self._heavy_chain_id = heavy_chains[0]

    def _prepare_lsite(self):
        lsite_content = f"95-102:{self._heavy_chain_id}\n"  # Chothia CDR H3
        with open(os.path.join(self.tmpdir.name, 'lsite.txt'), 'w') as f:
            f.write(lsite_content)
        print(f"[INFO] lsite content: {lsite_content}")

    def _prepare_rsite(self):
        rsite_content = ""
        for site in self._epitope_sites:
            rsite_content += f"{site.resseq}:{site.chain}\n"
        with open(os.path.join(self.tmpdir.name, 'rsite.txt'), 'w') as f:
            f.write(rsite_content)
        print(f"[INFO] rsite content: {rsite_content}")

    def dock(self):
        if not (self._has_receptor and self._has_ligand):
            raise ValueError('Missing receptor or ligand.')
        self._prepare_lsite()

        cmd_hdock = [self.hdock_bin, "receptor.pdb", "ligand.pdb", "-lsite", "lsite.txt"]
        if self._epitope_sites is not None:
            self._prepare_rsite()
            cmd_hdock += ["-rsite", "rsite.txt"]
        subprocess.run(
            cmd_hdock,
            cwd=self.tmpdir.name, check=True
        )

        cmd_pl = [self.createpl_bin, "Hdock.out", "ligand_docked.pdb", "-lsite", "lsite.txt"]
        if self._epitope_sites is not None:
            self._prepare_rsite()
            cmd_pl += ["-rsite", "rsite.txt"]
        subprocess.run(
            cmd_pl, 
            cwd=self.tmpdir.name, check=True
        )
        return self._dump_complex_pdb()


if __name__ == '__main__':
    with HDockAntibody('hdock', 'createpl') as dock:
        dock.set_antigen('./data/dock/receptor.pdb', [DockSite('A', 991)])
        dock.set_antibody('./data/example_dock/3qhf_fv.pdb')
        print(dock.dock())
