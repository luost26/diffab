import os
import argparse
import ray
import shelve
import time
import pandas as pd
from typing import Mapping

from tools.eval.base import EvalTask, TaskScanner
from tools.eval.similarity import eval_similarity
from tools.eval.energy import eval_interface_energy


@ray.remote(num_cpus=1)
def evaluate(task, args):
    funcs = []
    funcs.append(eval_similarity)
    if not args.no_energy:
        funcs.append(eval_interface_energy)
    for f in funcs:
        task = f(task)
    return task


def dump_db(db: Mapping[str, EvalTask], path):
    table = []
    for task in db.values():
        if 'abopt' in path and task.scores['seqid'] >= 100.0:
            # In abopt (Antibody Optimization) mode, ignore sequences identical to the wild-type
            continue
        table.append(task.to_report_dict())
    table = pd.DataFrame(table)
    table.to_csv(path, index=False, float_format='%.6f')
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--no_energy', action='store_true', default=False)
    args = parser.parse_args()
    ray.init()
    
    db_path = os.path.join(args.root, 'evaluation_db')
    with shelve.open(db_path) as db:
        scanner = TaskScanner(root=args.root, postfix=args.pfx, db=db)

        while True:        
            tasks = scanner.scan()
            futures = [evaluate.remote(t, args) for t in tasks]
            if len(futures) > 0:
                print(f'Submitted {len(futures)} tasks.')
            while len(futures) > 0:
                done_ids, futures = ray.wait(futures, num_returns=1)
                for done_id in done_ids:
                    done_task = ray.get(done_id)
                    done_task.save_to_db(db)
                    print(f'Remaining {len(futures)}. Finished {done_task.in_path}')
                db.sync()
            
            dump_db(db, os.path.join(args.root, 'summary.csv'))
            time.sleep(1.0)

if __name__ == '__main__':
    main()
