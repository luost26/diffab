import os
import shutil
import argparse
from diffab.tools.dock.hdock import HDockAntibody
from diffab.tools.runner.design_for_pdb import args_factory, design_for_pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--antigen', type=str, required=True)
    parser.add_argument('--antibody', type=str, default='./data/examples/3QHF_Fv.pdb')
    parser.add_argument('--heavy', type=str, default='H', help='Chain id of the heavy chain.')
    parser.add_argument('--light', type=str, default='L', help='Chain id of the light chain.')
    parser.add_argument('--hdock_bin', type=str, default='./bin/hdock')
    parser.add_argument('--createpl_bin', type=str, default='./bin/createpl')
    parser.add_argument('-n', '--num_docks', type=int, default=10)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    args = parser.parse_args()

    hdock_missing = []
    if not os.path.exists(args.hdock_bin):
        hdock_missing.append(args.hdock_bin)
    if not os.path.exists(args.createpl_bin):
        hdock_missing.append(args.createpl_bin)
    if len(hdock_missing) > 0:
        print("[WARNING] The following HDOCK applications are missing:")
        for f in hdock_missing:
            print(f" > {f}")
        print("Please download HDOCK from http://huanglab.phys.hust.edu.cn/software/hdocklite/ "
                "and put `hdock` and `createpl` to the above path.")
        exit()

    antigen_name = os.path.basename(os.path.splitext(args.antigen)[0])
    docked_pdb_dir = os.path.join(os.path.splitext(args.antigen)[0] + '_dock')
    os.makedirs(docked_pdb_dir, exist_ok=True)
    docked_pdb_paths = []
    for fname in os.listdir(docked_pdb_dir):
        if fname.endswith('.pdb'):
            docked_pdb_paths.append(os.path.join(docked_pdb_dir, fname))
    if len(docked_pdb_paths) < args.num_docks:
        with HDockAntibody() as dock_session:
            dock_session.set_antigen(args.antigen)
            dock_session.set_antibody(args.antibody)
            docked_tmp_paths = dock_session.dock()
            for i, tmp_path in enumerate(docked_tmp_paths[:args.num_docks]):
                dest_path = os.path.join(docked_pdb_dir, f"{antigen_name}_Ab_{i:04d}.pdb")
                shutil.copyfile(tmp_path, dest_path)
                print(f'[INFO] Copy {tmp_path} -> {dest_path}')
                docked_pdb_paths.append(dest_path)

    for pdb_path in docked_pdb_paths:
        current_args = vars(args)
        current_args['tag'] += antigen_name
        design_args = args_factory(
            pdb_path = pdb_path,
            **current_args,
        )
        design_for_pdb(design_args)


if __name__ == '__main__':
    main()
