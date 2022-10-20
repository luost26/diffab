import os
import shutil
import pandas as pd
import yaml
import subprocess
import streamlit as st
import stmol
import py3Dmol
import tempfile
import re
import abnumber
from tqdm.auto import tqdm
from Bio import PDB
from collections import OrderedDict

from diffab.tools.renumber import renumber as renumber_antibody
from diffab.tools.renumber.run import (
    biopython_chain_to_sequence, 
    assign_number_to_sequence,
)

CDR_OPTIONS = OrderedDict()
CDR_OPTIONS['H_CDR1'] = 'H1'
CDR_OPTIONS['H_CDR2'] = 'H2'
CDR_OPTIONS['H_CDR3'] = 'H3'
CDR_OPTIONS['L_CDR1'] = 'L1'
CDR_OPTIONS['L_CDR2'] = 'L2'
CDR_OPTIONS['L_CDR3'] = 'L3'

DESIGN_MODES = OrderedDict()
DESIGN_MODES['denovo'] = 'De novo design'
DESIGN_MODES['denovo_dock'] = 'De novo design (with docking)'
DESIGN_MODES['opt'] = 'Optimization'
DESIGN_MODES['fixbb'] = 'Fix-backbone'

MODE_CONFIG = {
    'denovo': './configs/test/codesign_multicdrs.yml',
    'denovo_dock': './configs/test/codesign_multicdrs.yml',
    'opt': './configs/test/abopt_singlecdr.yml',
    'fixbb': './configs/test/fixbb.yml',
}


def dict_to_func(d):
    def f(x):
        return d[x]
    return f


def get_config(save_dir, mode, cdrs, num_samples=5, optimization_step=4):
    tmpl_path = MODE_CONFIG[mode]
    with open(tmpl_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['sampling']['cdrs'] = cdrs
    cfg['sampling']['num_samples'] = num_samples
    cfg['sampling']['optimize_steps'] = [optimization_step, ]

    save_path = os.path.join(save_dir, 'design.yml')
    with open(save_path, 'w') as f:
        yaml.dump(cfg, f)
    return cfg, save_path


def run_design(pdb_path, config_path, output_dir, docking, display_widget):
    if docking:
        cmd = f"python design_dock.py --antigen {pdb_path} --config {config_path} "
    else:
        cmd = f"python design_pdb.py {pdb_path} --config {config_path} "
    cmd += f"--batch_size 1 --out_root {output_dir}"

    result_dir = os.path.join(output_dir, 'design')
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    
    output_buffer = ''
    proc = subprocess.Popen(
        cmd,
        shell=True,
        env=os.environ.copy(),
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
    )
    for line in iter(proc.stdout.readline, b''):
        output_buffer += line.decode()

        display_widget.code( 
            '\n'.join(output_buffer.splitlines()[-10:]),
        )
    proc.stdout.close()
    proc.wait()



def gather_results(result_dir):
    outputs = []
    for root, dirs, files in os.walk(result_dir):
        for fname in files:
            if not re.match('^\d\d\d\d\.pdb$', fname):
                continue
            fpath = os.path.join(root, fname)
            gname = os.path.basename(root)

            outputs.append((gname, fname, fpath))
    
    parser = PDB.PDBParser(QUIET=True)
    records = []
    fpath_to_name = {}
    for gname, fname, fpath in tqdm(outputs):
        name = f"{gname}_{fname}"
        structure = parser.get_structure(name, fpath)
        model = structure[0]
        record = {
            'name': name,
            'H1': None, 'H2': None, 'H3': None,
            'L1': None, 'L2': None, 'L3': None,
            'gname': gname, 'fname': fname, 'fpath': fpath,
        }
        for chain in model:
            try:
                seq, reslist = biopython_chain_to_sequence(chain)
                numbers, abchain = assign_number_to_sequence(seq)
                if abchain.chain_type == 'H':
                    record['H1'] = abchain.cdr1_seq
                    record['H2'] = abchain.cdr2_seq
                    record['H3'] = abchain.cdr3_seq
                elif abchain.chain_type in ('L', 'K'):
                    record['L1'] = abchain.cdr1_seq
                    record['L2'] = abchain.cdr2_seq
                    record['L3'] = abchain.cdr3_seq
            except abnumber.ChainParseError as e:
                pass
        records.append(record)
        fpath_to_name[fpath] = name
    records = pd.DataFrame(records)

    return records, dict_to_func(fpath_to_name)


def main():
    # Temporary workspace directory
    if 'tempdir_path' not in st.session_state:
        tempdir_path = tempfile.mkdtemp(prefix='streamlit')
        st.session_state.tempdir_path = tempdir_path
    else:
        tempdir_path = st.session_state.tempdir_path
    # Page layout
    st.set_page_config(layout="wide")
    left_col, right_col = st.columns(2)

    # Step 1: Upload PDB or choose an example
    uploaded_file = None
    with left_col:
        tab_upload, tab_example = st.tabs(['Upload PDB', 'Examples'])
        with tab_upload:
            uploaded_file = st.file_uploader(
                'Antigen structure or antibody-antigen complex',
                # disabled=True
            )
        with tab_example:
            pass

    # Step 1.2: Retrieve uploaded PDB
    if uploaded_file is not None:
        pdb_path = os.path.join(tempdir_path, 'structure.pdb')
        renum_path = os.path.join(tempdir_path, 'structure_renumber.pdb')
        with open(pdb_path, 'w') as f:
            f.write(uploaded_file.getvalue().decode())
        H_chains, L_chains, Ag_chains = renumber_antibody(
            in_pdb = pdb_path,
            out_pdb = renum_path,
            return_other_chains = True
        )
        H_chain = H_chains[0] if H_chains else None
        L_chain = L_chains[0] if L_chains else None
        docking = H_chain is None and L_chain is None

    # Step 2: Design options
    if 'submit' not in st.session_state:
        st.session_state.submit = False
    if 'done' not in st.session_state:
        st.session_state.done = False

    if uploaded_file is not None:
        with left_col:
            st.dataframe(pd.DataFrame({
                'Heavy': {'Chain': H_chain},
                'Light': {'Chain': L_chain},
                'Antigen': {'Chain': ','.join(Ag_chains)},
            }), use_container_width=True)

            form = st.form('design_form')
            with form:
                if H_chain is None and L_chain is None:
                    # Antigen only
                    cdr_options = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']
                    cdr_default = ['H_CDR1', 'H_CDR2', 'H_CDR3']
                    mode_options = ['denovo_dock']
                elif H_chain is not None and L_chain is None:
                    # Heavy chain + Antigen
                    cdr_options = ['H_CDR1', 'H_CDR2', 'H_CDR3']
                    cdr_default = ['H_CDR1', 'H_CDR2', 'H_CDR3']
                    mode_options = ['denovo', 'opt', 'fixbb']
                elif H_chain is None and L_chain is not None:
                    # Light chain + Antigen
                    cdr_options = ['L_CDR1', 'L_CDR2', 'L_CDR3']
                    cdr_default = ['L_CDR1', 'L_CDR2', 'L_CDR3']
                    mode_options = ['denovo', 'opt', 'fixbb']
                else:
                    # H + L + Ag
                    cdr_options = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']
                    cdr_default = ['H_CDR1', 'H_CDR2', 'H_CDR3']
                    mode_options = ['denovo', 'opt', 'fixbb']
                
                design_mode = st.radio(
                    'Mode',
                    mode_options,
                    format_func=dict_to_func(DESIGN_MODES),
                    # disabled=True,
                )
                cdr_choices = st.multiselect(
                    'CDRs',
                    cdr_options,
                    default = cdr_default,
                    format_func=dict_to_func(CDR_OPTIONS),
                    # disabled=True,
                )
                submit = st.form_submit_button('Run')
                st.session_state.submit = st.session_state.submit or submit
                if submit:
                    st.session_state.done = False

    # Step 3: Prepare configuration
    if uploaded_file is not None and st.session_state.submit:
        config, config_path = get_config(
            save_dir = tempdir_path,
            mode = design_mode,
            cdrs = cdr_choices,
        )

    # Step 4: Run design
    if st.session_state.submit:
        with right_col:
            result_molecule_display = st.empty()
            result_select_widget = st.empty()
            result_table_display = st.empty()
            output_display = st.empty()
        if not st.session_state.done:
            run_design(
                pdb_path = renum_path,
                config_path = config_path,
                output_dir = tempdir_path,
                docking = docking,
                display_widget = output_display
            )
            st.session_state.done = True
    
    # Step 5: Show results:
    if st.session_state.submit and st.session_state.done:
        result_dir = os.path.join(tempdir_path, 'design')
        df_cols = ['name'] + list(CDR_OPTIONS.values())
        df_results, fpath_to_name = gather_results(result_dir)
        result_table_display.dataframe(df_results[df_cols], use_container_width=True)

        display_pdb_path = result_select_widget.selectbox(
            label = "Visualize",
            options = df_results['fpath'],
            format_func = fpath_to_name,
        )

        with open(display_pdb_path, 'r') as f:
            pdb_str = f.read()
        xyzview = py3Dmol.view(width=380, height=380)
        xyzview.addModelsAsFrames(pdb_str)
        xyzview.setStyle({'cartoon':{'color':'spectrum'}})
        xyzview.zoomTo()
        with result_molecule_display:
            stmol.showmol(xyzview, width=380, height=380)

    
if __name__ == '__main__':
    main()
