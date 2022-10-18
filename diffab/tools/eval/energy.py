# pyright: reportMissingImports=false
import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
pyrosetta.init(' '.join([
    '-mute', 'all',
    '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig',
]))

from tools.eval.base import EvalTask


def pyrosetta_interface_energy(pdb_path, interface):
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    return pose.scores['dG_separated']


def eval_interface_energy(task: EvalTask):
    model_gen = task.get_gen_biopython_model()
    antigen_chains = set()
    for chain in model_gen:
        if chain.id not in task.ab_chains:
            antigen_chains.add(chain.id)
    antigen_chains = ''.join(list(antigen_chains))
    antibody_chains = ''.join(task.ab_chains)
    interface = f"{antibody_chains}_{antigen_chains}"

    dG_gen = pyrosetta_interface_energy(task.in_path, interface)
    dG_ref = pyrosetta_interface_energy(task.ref_path, interface)

    task.scores.update({
        'dG_gen': dG_gen,
        'dG_ref': dG_ref,
        'ddG': dG_gen - dG_ref
    })
    return task
