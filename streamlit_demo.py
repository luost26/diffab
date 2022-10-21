import sys
sys.path.append('./diffab-repo')
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
import gzip
import tarfile
import torch
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
DESIGN_MODES['denovo_dock'] = 'De novo design (with HDOCK)'
DESIGN_MODES['opt'] = 'Optimization'
DESIGN_MODES['fixbb'] = 'Fix-backbone'

MODE_CONFIG = {
    'denovo': './configs/test/codesign_multicdrs.yml',
    'denovo_dock': './configs/test/codesign_multicdrs.yml',
    'opt': './configs/test/abopt_singlecdr.yml',
    'fixbb': './configs/test/fixbb.yml',
}

GPU_AVAILABLE = torch.cuda.is_available()
DEFAULT_NUM_SAMPLES = 5 if GPU_AVAILABLE else 1
DEFAULT_NUM_DOCKS = 3


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


def run_design(pdb_path, config_path, output_dir, docking, display_widget, num_docks=DEFAULT_NUM_DOCKS):
    if docking:
        cmd = f"python design_dock.py --antigen {pdb_path} --config {config_path} --num_docks {num_docks} "
    else:
        cmd = f"python design_pdb.py {pdb_path} --config {config_path} "
    cmd += f"--batch_size 1 --out_root {output_dir} "

    if GPU_AVAILABLE:
        cmd += "--device cuda"
    else:
        cmd += "--device cpu"

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


@st.cache
def renumber_antibody_cached(in_pdb, out_pdb, file_id):
    return renumber_antibody(
        in_pdb, out_pdb, return_other_chains=True
    )


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

    with tarfile.open(os.path.join(result_dir, 'generated.tar.gz'), 'w:gz') as tar:
        for record in records:
            info = tar.gettarinfo(record['fpath'])
            info.name = record['name']
            tar.addfile(
                tarinfo = info,
                fileobj = open(record['fpath'], 'rb'),
            )
    
    records = pd.DataFrame(records)

    return records, fpath_to_name


def main():
    # Temporary workspace directory
    if 'tempdir_path' not in st.session_state:
        tempdir_path = tempfile.mkdtemp(prefix='streamlit')
        st.session_state.tempdir_path = tempdir_path
    else:
        tempdir_path = st.session_state.tempdir_path
    # Page layout
    st.set_page_config(layout="wide")
    st.markdown(
        "# DiffAb \n\n"
        "Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures (NeurIPS 2022) \n\n"
        "[[Paper](https://www.biorxiv.org/content/10.1101/2022.07.10.499510.abstract)] "
        "[[Code](https://github.com/luost26/diffab)]"
    )
    left_col, right_col = st.columns(2)

    # Step 1: Upload PDB or choose an example
    uploaded_file = None
    with left_col:
        uploaded_file = st.file_uploader(
            'Antigen structure or antibody-antigen complex',
            # disabled=True
        )

        if uploaded_file is None:
            st.session_state.submit = False
            st.session_state.done = False

            with st.expander("Don't know what to upload? Try these examples", expanded=True):
                with open('./data/examples/7DK2_AB_C.pdb', 'r') as f:
                    st.download_button(
                        'RBD + Antibody Complex', 
                        data = f,
                        file_name='RBD_AbAg.pdb',
                    )
                with open('./data/examples/Omicron_RBD.pdb', 'r') as f:
                    st.download_button(
                        'RBD Antigen Only (Much slower)', 
                        data = f,
                        file_name = 'RBD_AgOnly.pdb',
                    )
                st.text('Please upload the downloaded PDB file to run the demo.')

    # Step 1.2: Retrieve uploaded PDB
    if uploaded_file is not None:
        pdb_path = os.path.join(tempdir_path, 'structure.pdb')
        renum_path = os.path.join(tempdir_path, 'structure_renumber.pdb')
        with open(pdb_path, 'w') as f:
            f.write(uploaded_file.getvalue().decode())
        H_chains, L_chains, Ag_chains = renumber_antibody_cached(
            in_pdb = pdb_path,
            out_pdb = renum_path,
            file_id = uploaded_file.id
        )
        H_chain = H_chains[0] if H_chains else None
        L_chain = L_chains[0] if L_chains else None
        docking = H_chain is None and L_chain is None

    # Step 2: Design options
    if uploaded_file is not None:
        with left_col:
            st.dataframe(pd.DataFrame({
                'Heavy': {'Chain': H_chain},
                'Light': {'Chain': L_chain},
                'Antigen': {'Chain': ','.join(Ag_chains)},
            }))

            if docking:
                st.warning('No antibodies detected. Will try to run docking (very slow).')

            # form = st.form('design_form')
            form = st.container()
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

                if docking:
                    num_docks = st.slider(
                        'Number of docking poses', 
                        min_value=1, max_value=10, value=DEFAULT_NUM_DOCKS,
                    )
                else:
                    num_docks = 0
                num_designs = st.slider(
                    'Number of samples',
                    min_value=1, max_value=10, value=DEFAULT_NUM_SAMPLES,
                )

                if not GPU_AVAILABLE:
                    st.warning('No GPU available. Sampling might be very slow.')

                btn_placeholder = st.empty()
                submit = btn_placeholder.button('Run', key="run_btn_real")
                st.session_state.submit = st.session_state.submit or submit
                if submit:
                    st.session_state.done = False
                    btn_placeholder.empty()

    # Step 3: Prepare configuration and run design
    if uploaded_file is not None and st.session_state.submit:

        with left_col:
            output_display = st.empty()

        with right_col:
            result_molecule_display = st.empty()
            result_select_widget = st.empty()
            result_table_display = st.empty()
            result_download_btn = st.empty()

        if not st.session_state.done:
            output_display.code('[INFO] Your job has been submitted. Please wait...\n')

            config, config_path = get_config(
                save_dir = tempdir_path,
                mode = design_mode,
                cdrs = cdr_choices,
                num_samples = num_designs,
            )

            run_design(
                pdb_path = renum_path,
                config_path = config_path,
                output_dir = tempdir_path,
                docking = docking,
                display_widget = output_display,
                num_docks = num_docks,
            )
            st.session_state.done = True
    
            result_dir = os.path.join(tempdir_path, 'design')
            df_cols = ['name'] + list(CDR_OPTIONS.values())
            df_results, fpath_to_name = gather_results(result_dir)
            st.session_state.results = (df_results, fpath_to_name)

    # Step 5: Show results:
    if st.session_state.submit and st.session_state.done:
        result_dir = os.path.join(tempdir_path, 'design')
        df_results, fpath_to_name = st.session_state.results

        df_cols = ['name'] + list(CDR_OPTIONS.values())
        result_table_display.dataframe(df_results[df_cols])

        display_pdb_path = result_select_widget.selectbox(
            label = "Visualize",
            options = df_results['fpath'],
            format_func = dict_to_func(fpath_to_name),
        )

        with open(os.path.join(result_dir, 'generated.tar.gz'), 'rb') as f:
            result_download_btn.download_button(
                label = "Download PDBs",
                data = f,
                file_name = "generated.tar.gz",
            )

        if not os.path.exists(display_pdb_path):
            display_pdb_path = df_results['fpath'][0]
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
