import os
import logging
import joblib
import pickle
import lmdb
from Bio import PDB
from Bio.PDB import PDBExceptions
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..utils.protein import parsers
from .sabdab import _label_heavy_chain_cdr, _label_light_chain_cdr
from ._base import register_dataset


def preprocess_antibody_structure(task):
    pdb_path = task['pdb_path']
    H_id = task.get('heavy_id', 'H')
    L_id = task.get('light_id', 'L')

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdb_path)[0]

    all_chain_ids = [c.id for c in model]

    parsed = {
        'id': task['id'],
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }
    try:
        if H_id in all_chain_ids:
            (
                parsed['heavy'], 
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                model[H_id],
                max_resseq = 113    # Chothia, end of Heavy chain Fv
            ))
        
        if L_id in all_chain_ids:
            (
                parsed['light'], 
                parsed['light_seqmap']
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(
                model[L_id],
                max_resseq = 106    # Chothia, end of Light chain Fv
            ))

        if parsed['heavy'] is None and parsed['light'] is None:
            raise ValueError(
                f'Neither valid antibody H-chain or L-chain is found. '
                f'Please ensure that the chain id of heavy chain is "{H_id}" '
                f'and the id of the light chain is "{L_id}".'
            )

        
        ag_chain_ids = [cid for cid in all_chain_ids if cid not in (H_id, L_id)]
        if len(ag_chain_ids) > 0:
            chains = [model[c] for c in ag_chain_ids]
            (
                parsed['antigen'], 
                parsed['antigen_seqmap']
            ) = parsers.parse_biopython_structure(chains)

    except (
        PDBExceptions.PDBConstructionException, 
        parsers.ParsingException, 
        KeyError,
        ValueError,
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'], 
            e.__class__.__name__, 
            str(e)
        ))
        return None

    return parsed


@register_dataset('custom')
class CustomDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(self, structure_dir, transform=None, reset=False):
        super().__init__()
        self.structure_dir = structure_dir
        self.transform = transform

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

    @property
    def _cache_db_path(self):
        return os.path.join(self.structure_dir, 'structure_cache.lmdb')

    def _connect_db(self):
        self._close_db()
        self.db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db_conn.begin() as txn:
            keys = [k.decode() for k in txn.cursor().iternext(values=False)]
            self.db_ids = keys

    def _close_db(self):
        if self.db_conn is not None:
            self.db_conn.close()
        self.db_conn = None
        self.db_ids = None
        
    def _load_structures(self, reset):
        all_pdbs = []
        for fname in os.listdir(self.structure_dir):
            if not fname.endswith('.pdb'): continue
            all_pdbs.append(fname)

        if reset or not os.path.exists(self._cache_db_path):
            todo_pdbs = all_pdbs
        else:
            self._connect_db()
            processed_pdbs = self.db_ids
            self._close_db()
            todo_pdbs = list(set(all_pdbs) - set(processed_pdbs))

        if len(todo_pdbs) > 0:
            self._preprocess_structures(todo_pdbs)
    
    def _preprocess_structures(self, pdb_list):
        tasks = []
        for pdb_fname in pdb_list:
            pdb_path = os.path.join(self.structure_dir, pdb_fname)
            tasks.append({
                'id': pdb_fname,
                'pdb_path': pdb_path,
            })

        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_antibody_structure)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )

        db_conn = lmdb.open(
            self._cache_db_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

    def __len__(self):
        return len(self.db_ids)

    def __getitem__(self, index):
        self._connect_db()
        id = self.db_ids[index]
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./data/custom')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = CustomDataset(
        structure_dir = args.dir,
        reset = args.reset,
    )
    print(dataset[0])
    print(len(dataset))
    