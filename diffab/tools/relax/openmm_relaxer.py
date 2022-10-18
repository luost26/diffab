import os
import time
import io
import logging
import pdbfixer
import openmm
from openmm import app as openmm_app
from openmm import unit
ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

from diffab.tools.relax.base import RelaxTask


def current_milli_time():
    return round(time.time() * 1000)


def _is_in_the_range(ch_rs_ic, flexible_residue_first, flexible_residue_last):
    if ch_rs_ic[0] != flexible_residue_first[0]: return False
    r_first, r_last = tuple(flexible_residue_first[1:]), tuple(flexible_residue_last[1:])
    rs_ic = ch_rs_ic[1:]
    return r_first <= rs_ic <= r_last


class ForceFieldMinimizer(object):

    def __init__(self, stiffness=10.0, max_iterations=0, tolerance=2.39*unit.kilocalories_per_mole, platform='CUDA'):
        super().__init__()
        self.stiffness = stiffness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        assert platform in ('CUDA', 'CPU')
        self.platform = platform

    def _fix(self, pdb_str):
        fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_str))
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        fixer.addMissingHydrogens()

        out_handle = io.StringIO()
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
        return out_handle.getvalue()

    def _get_pdb_string(self, topology, positions):
        with io.StringIO() as f:
            openmm_app.PDBFile.writeFile(topology, positions, f, keepIds=True)
            return f.getvalue()

    def _minimize(self, pdb_str, flexible_residue_first=None, flexible_residue_last=None):
        pdb = openmm_app.PDBFile(io.StringIO(pdb_str))

        force_field = openmm_app.ForceField("amber99sb.xml")
        constraints = openmm_app.HBonds
        system = force_field.createSystem(pdb.topology, constraints=constraints)

        # Add constraints to non-generated regions
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", self.stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        
        if flexible_residue_first is not None and flexible_residue_last is not None:
            for i, a in enumerate(pdb.topology.atoms()):
                ch_rs_ic = (a.residue.chain.id, int(a.residue.id), a.residue.insertionCode)
                if not _is_in_the_range(ch_rs_ic, flexible_residue_first, flexible_residue_last) and a.element.name != "hydrogen":
                    force.addParticle(i, pdb.positions[i])
                
        system.addForce(force)

        # Set up the integrator and simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName("CUDA")
        simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)

        # Perform minimization
        ret = {}
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        simulation.minimizeEnergy(maxIterations=self.max_iterations, tolerance=self.tolerance)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
        ret["min_pdb"] = self._get_pdb_string(simulation.topology, state.getPositions())

        return ret['min_pdb'], ret

    def _add_energy_remarks(self, pdb_str, ret):
        pdb_lines = pdb_str.splitlines()
        pdb_lines.insert(1, "REMARK   1  FINAL ENERGY:   {:.3f} KCAL/MOL".format(ret['efinal']))
        pdb_lines.insert(1, "REMARK   1  INITIAL ENERGY: {:.3f} KCAL/MOL".format(ret['einit']))
        return "\n".join(pdb_lines)

    def __call__(self, pdb_str, flexible_residue_first=None, flexible_residue_last=None, return_info=True):
        if '\n' not in pdb_str and pdb_str.lower().endswith(".pdb"):
            with open(pdb_str) as f:
                pdb_str = f.read()

        pdb_fixed = self._fix(pdb_str)
        pdb_min, ret = self._minimize(pdb_fixed, flexible_residue_first, flexible_residue_last)
        pdb_min = self._add_energy_remarks(pdb_min, ret)
        if return_info:
            return pdb_min, ret
        else:
            return pdb_min


def run_openmm(task: RelaxTask):
    if not task.can_proceed() :
        return task
    if task.update_if_finished('openmm'):
        return task

    try:
        minimizer = ForceFieldMinimizer()
        with open(task.current_path, 'r') as f:
            pdb_str = f.read()

        pdb_min = minimizer(
            pdb_str = pdb_str,
            flexible_residue_first = task.flexible_residue_first,
            flexible_residue_last = task.flexible_residue_last,
            return_info = False,
        )
        out_path = task.set_current_path_tag('openmm')
        with open(out_path, 'w') as f:
            f.write(pdb_min)
        task.mark_success()
    except ValueError as e:
        logging.warning(
            f'{e.__class__.__name__}: {str(e)} ({task.current_path})'
        )        
        task.mark_failure()
    return task

