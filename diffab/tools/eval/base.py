import os
import re
import json
import shelve
from Bio import PDB
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


@dataclass
class EvalTask:
    in_path: str
    ref_path: str
    info: dict
    structure: str
    name: str
    method: str
    cdr: str
    ab_chains: List

    residue_first: Optional[Tuple] = None
    residue_last: Optional[Tuple] = None
    
    scores: dict = field(default_factory=dict)

    def get_gen_biopython_model(self):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(self.in_path, self.in_path)[0]

    def get_ref_biopython_model(self):
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(self.ref_path, self.ref_path)[0]

    def save_to_db(self, db: shelve.Shelf):
        db[self.in_path] = self

    def to_report_dict(self):
        return {
            'method': self.method,
            'structure': self.structure,
            'cdr': self.cdr,
            'filename': os.path.basename(self.in_path),
            **self.scores
        }


class TaskScanner:

    def __init__(self, root, postfix=None, db: Optional[shelve.Shelf]=None):
        super().__init__()
        self.root = root
        self.postfix = postfix
        self.visited = set()
        self.db = db
        if db is not None:
            for k in db.keys():
                self.visited.add(k)

    def _get_metadata(self, fpath):
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(fpath)), 
            'metadata.json'
        )
        tag_name = os.path.basename(os.path.dirname(fpath))
        method_name = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(fpath)))
        )
        try:
            antibody_chains = set()
            info = None
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            for item in metadata['items']:
                if item['tag'] == tag_name:
                    info = item
                antibody_chains.add(item['residue_first'][0])
            if info is not None:
                info['antibody_chains'] = list(antibody_chains)
                info['structure'] = metadata['identifier']
                info['method'] = method_name
            return info
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return None

    def scan(self) -> List[EvalTask]: 
        tasks = []
        if self.postfix is None or not self.postfix:
            input_fname_pattern = '^\d+\.pdb$'
            ref_fname = 'REF1.pdb'
        else:
            input_fname_pattern = f'^\d+\_{self.postfix}\.pdb$'
            ref_fname = f'REF1_{self.postfix}.pdb'
        for parent, _, files in os.walk(self.root):
            for fname in files:
                fpath = os.path.join(parent, fname)
                if not re.match(input_fname_pattern, fname):
                    continue
                if os.path.getsize(fpath) == 0:
                    continue
                if fpath in self.visited:
                    continue

                # Path to the reference structure
                ref_path = os.path.join(parent, ref_fname)
                if not os.path.exists(ref_path):
                    continue

                # CDR information
                info = self._get_metadata(fpath)
                if info is None:
                    continue
                tasks.append(EvalTask(
                    in_path = fpath,
                    ref_path = ref_path,
                    info = info,
                    structure = info['structure'],
                    name = info['name'],
                    method = info['method'],
                    cdr = info['tag'],
                    ab_chains = info['antibody_chains'],
                    residue_first = info.get('residue_first', None),
                    residue_last  = info.get('residue_last', None),
                ))
                self.visited.add(fpath)
        return tasks
