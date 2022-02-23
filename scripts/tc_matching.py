'''
    Matches trigger cells to reconstructed clusters and gen particles, and
    converts input root files to dataframes.
'''

import os
import sys
import argparse
import subprocess
import warnings
from itertools import chain
from pathlib import Path

from tqdm import tqdm
import pickle
import yaml
import awkward
import uproot
import numpy as np
import h5py
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def set_indices(df):
    # modifies multiindex from uproot so that the leading index corresponds the
    # event number
    index = df.index
    event_numbers = df['event']
    new_index = [(e, ix[1]) for e, ix in zip(event_numbers, index)]
    df.set_index(pd.MultiIndex.from_tuples(new_index, names=['event', 'id']), inplace=True)
    df.drop('event', axis=1, inplace=True)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='config/tc_matching_cfg.yaml',
                        help='File specifying configuration of matching process.'
                        )
    parser.add_argument('--job_id', 
                        default=0, 
                        type=int, 
                        help='Index of the output job.'
                        )
    parser.add_argument('--input_file', 
                        default=None, 
                        type=str, 
                        help='File with list of input files.  If this option is used, it will override the value provided in the configuration file.'
                        )
    parser.add_argument('--output_dir', 
                        default=None, 
                        type=str, 
                        help='Directory to write output file to.  If this option is used, it will override the value provided in the configuration file.'
                        )
    parser.add_argument('--max_events', 
                        default=None, 
                        type=int, 
                        help='Maximum number of events to process.  Useful for testing...'
                        )
    parser.add_argument('--is_batch', 
                        action='store_true',
                        help='Use this if running with a (lpc condor?) batch system to enable xrd to open remote files.'
                        )
    args = parser.parse_args()
   
    # Load configuration file
    with open(args.config, 'r') as config_file: 
        config = yaml.safe_load(config_file)
    
    # Unpack options from configuration file
    dr_threshold    = config['dr_threshold']
    gen_tree_name   = config['gen_tree']
    match_only      = config['match_only']
    reached_ee      = config['reached_ee']
    backend         = config['backend']
    frontend_algos  = config['frontends']
    ntuple_template = config['ntuple_template']
    output_dir      = config['output_destination']

    branches_gen    = config['branches_gen']
    branches_cl3d   = config['branches_cl3d']
    branches_tc     = config['branches_tc']

    # baseline cuts (move these to the configuration file?)
    gen_cuts     = f'(genpart_reachedEE == {reached_ee}) and (genpart_gen != -1)'
    tc_cuts      = ''
    cluster_cuts = 'cl3d_pt > 5.'

    if args.input_file:
        with open(args.input_file) as f:
            file_list = f.read().splitlines()
    else:
        if type(config['input_files']) == str:
            with open(config['input_files'], 'r') as f:
                file_list = f.read().splitlines()
        else:
            file_list = config['input_files']

    if args.output_dir:
        output_dir = args.output_dir

    if args.is_batch: # having some issues with xrd on condor
        # override some user options when running over batch
        uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource
        output_dir = 'data'

    print('Getting gen particles and trigger cells...')

    # read root files
    df_gen_list = []
    df_tc_list = []
    layer_labels = [f'cl3d_layer_{n}_pt' for n in range(36)]
    for filename in tqdm(file_list, desc='Processing files and retrieving data...'):
        tqdm.write(filename)

        # get gen particles
        uproot_file = uproot.open(filename)
        gen_tree = uproot_file[gen_tree_name]
        df_gen = pd.concat([df for df in gen_tree.iterate(branches_gen, library='pd', step_size=5000, entry_stop=args.max_events)])
        df_gen.query(gen_cuts, inplace=True)
        df_gen_list.append(df_gen)

        # get trigger cells (do this for the most inclusive algorithm, same as gen tree)
        tc_tree = uproot_file[gen_tree_name]
        df_tc = pd.concat([df for df in tc_tree.iterate(branches_tc, library='pd', step_size=500, entry_stop=args.max_events)])
        if tc_cuts != '':
            df_tc.query(tc_cuts, inplace=True)
        df_tc_list.append(df_tc)

    print('Finished extracting data.  Carrying out trigger cell and cluster gen matching...')

    # concatenate dataframes for each algorithm after running over all files
    # clean particles that are not generator-level (genpart_gen) or didn't
    # reach endcap (genpart_reachedEE)
    df_tc = pd.concat(df_tc_list)
    set_indices(df_tc)

    df_gen = pd.concat(df_gen_list)
    set_indices(df_gen)
    df_gen_pos, df_gen_neg = [df for _, df in df_gen.groupby(df_gen['genpart_exeta'] < 0)]

    output_name = f'{output_dir}/output_{args.job_id}.pkl'
    outfile = open(output_name, 'wb')
    #store = pd.HDFStore(output_name, mode='w')
    output_dict = dict(gen=df_gen, tc=df_tc)

    ### matching loop deleted, but might be useful to reintroduce here ###

    ### save files to savedir in HDF (temporarily use pickle files because of problems with hdf5 on condor)
    pickle.dump(output_dict, outfile)
    outfile.close()
    print(f'Writing output to {output_name}')

    #store['gen'] = df_gen
    #store.close()
