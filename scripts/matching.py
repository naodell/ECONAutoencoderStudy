#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import sys
import uproot4
from datetime import date
import argparse
import subprocess
from itertools import chain

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

workdir=os.getcwd()

'''
Matching 3d clusters with gen information
'''

def xrd_prefix(filepaths):
    prefix = ''
    allow_prefetch = False
    if not isinstance(filepaths, (list, tuple)):
        filepaths = [filepaths]
    filepath = filepaths[0]
    if filepath.startswith('/eos/cms'):
        prefix = 'root://eoscms.cern.ch/'
    elif filepath.startswith('/eos/user'):
        prefix = 'root://eosuser.cern.ch/'
    elif filepath.startswith('/eos/uscms'):
        prefix = 'root://cmseos.fnal.gov/'
    elif filepath.startswith('/store/'):
        # remote file
        import socket
        host = socket.getfqdn()
        if 'cern.ch' in host:
            prefix = 'root://xrootd-cms.infn.it//'
        else:
            prefix = 'root://cmseos.fnal.gov//'
        allow_prefetch = True
    expanded_paths = [(prefix + '/' + f if prefix else f) for f in filepaths]
    return expanded_paths, allow_prefetch

# compute deltar between 3d cluster and genparticle
# this is done by looking a the delta-phi and delta-eta
def deltar(df):
    df['deta']=df['cl3d_eta']-df['genpart_exeta']
    df['dphi']=np.abs(df['cl3d_phi']-df['genpart_exphi'])
    sel=df['dphi']>np.pi
    df['dphi']-=sel*(2*np.pi)
    return(np.sqrt(df['dphi']*df['dphi']+df['deta']*df['deta']))
    
def matching(event):
    return event.cl3d_pt==event.cl3d_pt.max()

# get all branches from tree
# the pT per layer is a vector of a vector
def openroot(files, algo_trees, gen_tree):
    gens = []
    algos = {}
    branches_gen=['event','genpart_pid',
                  'genpart_exphi', 'genpart_exeta','genpart_gen',
                  'genpart_reachedEE', 'genpart_pt', 'genpart_energy'
                  ]
    branches_cl3d=['event','cl3d_pt','cl3d_eta','cl3d_phi',
                   'cl3d_showerlength','cl3d_coreshowerlength',
                   'cl3d_firstlayer','cl3d_maxlayer',
                   'cl3d_seetot','cl3d_spptot','cl3d_szz', 'cl3d_srrtot',
                   'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 
                   'cl3d_layer10', 'cl3d_layer50', 'cl3d_layer90', 
                   'cl3d_ntc67', 'cl3d_ntc90'
                   ]
    
    for filename in files:
        print(filename)
        uproot_file = uproot4.open(filename)
        gens.append(uproot_file[gen_tree].arrays(branches_gen, library='pd'))
        for algo_name, algo_tree in algo_trees.items():
            if not algo_name in algos:
                algos[algo_name] = []

            tree  = uproot_file[algo_tree]

            print(algo_tree)
            df_cl = tree.arrays(branches_cl3d, library='pd')

            # Trick to read layers pTs, which is a vector of vector
            #df_cl['cl3d_layer_pt'] = list(chain.from_iterable(tree.arrays(['cl3d_layer_pt'])[b'cl3d_layer_pt'].tolist()))
            algos[algo_name].append(df_cl)
        
    df_algos = {}
    df_gen = pd.concat(gens)
    for algo_name, dfs in algos.items():
        df_algos[algo_name] = pd.concat(dfs)
    return (df_gen, df_algos)

def get_output_name(md, jobid):
    info = md['jobs'][jobid]
    return '{samp}_{idx}.hdf5'.format(samp=md['name'], idx=info['idx'])

def preprocessing(md):
    files          = md['jobs'][args.jobid]['inputfiles']
    threshold      = md['threshold']
    algo_trees     = md['algo_trees']
    gen_tree       = md['gen_tree']
    bestmatch_only = md['bestmatch_only']
    reachedEE      = md['reachedEE']

    output_name = get_output_name(md, args.jobid)
    
    # read root files
    gen, algo  = openroot(files, algo_trees, gen_tree)
    n_rec      = {}
    algo_clean = {}
    
    # clean particles that are not generator-level (genpart_gen) or didn't reach endcap (genpart_reachedEE)
    sel       = gen['genpart_reachedEE']==reachedEE
    gen_clean = gen[sel]
    sel       = gen_clean['genpart_gen']!=-1
    gen_clean = gen_clean[sel]

    # split df_gen_clean in two, one collection for each endcap (eta>0 and eta<=0), same for 3d clusters
    sel1    = gen_clean['genpart_exeta']<=0
    sel2    = gen_clean['genpart_exeta']>0
    gen_neg = gen_clean[sel1]
    gen_pos = gen_clean[sel2]

    gen_pos.set_index('event', inplace=True)
    gen_neg.set_index('event', inplace=True)
    
    for algo_name, df_algo in algo.items():

        # split clusters in two, one collection for each endcap
        sel1     = df_algo['cl3d_eta']<=0
        sel2     = df_algo['cl3d_eta']>0
        algo_neg = df_algo[sel1]
        algo_pos = df_algo[sel2]

        # set the indices
        algo_pos.set_index('event', inplace=True)
        algo_neg.set_index('event', inplace=True)

        # merging gen columns and cluster columns
        algo_pos_merged = gen_pos.join(algo_pos, how='left', rsuffix='_algo')
        algo_neg_merged = gen_neg.join(algo_neg, how='left', rsuffix='_algo')

        # compute deltar
        algo_pos_merged['deltar'] = deltar(algo_pos_merged)
        algo_neg_merged['deltar'] = deltar(algo_neg_merged)
        
        # keep track of the unmatched values (NaN)
        sel           = pd.isna(algo_pos_merged['deltar'])
        unmatched_pos = algo_pos_merged[sel]
        sel           = pd.isna(algo_neg_merged['deltar'])
        unmatched_neg = algo_neg_merged[sel]

        if unmatched_pos.shape[0] > 0:
            unmatched_pos.loc[:,'matches'] = False

        if unmatched_neg.shape[0] > 0:
            unmatched_neg.loc[:,'matches'] = False
        
        # select deltar under threshold (threshold is set in configuration file)
        sel             = algo_pos_merged['deltar']<=threshold
        algo_pos_merged = algo_pos_merged[sel]
        sel             = algo_neg_merged['deltar']<=threshold
        algo_neg_merged = algo_neg_merged[sel]
        
        # matching
        group                         = algo_pos_merged.groupby('event')
        n_rec_pos                     = group['cl3d_pt'].size()
        algo_pos_merged['best_match'] = group.apply(matching).array
        group                         = algo_neg_merged.groupby('event')
        n_rec_neg                     = group['cl3d_pt'].size()
        algo_neg_merged['best_match'] = group.apply(matching).array
        
        # keep matched clusters only
        if bestmatch_only:
            sel             = algo_pos_merged['best_match']==True
            algo_pos_merged = algo_pos_merged[sel]
        
            sel             = algo_neg_merged['best_match']==True
            algo_neg_merged = algo_neg_merged[sel]
    
        # remerge with NaN values
        algo_pos_merged = pd.concat([algo_pos_merged, unmatched_pos], sort=False).sort_values('event')
        algo_neg_merged = pd.concat([algo_neg_merged, unmatched_neg], sort=False).sort_values('event')
        
        n_rec[algo_name]      = n_rec_pos.append(n_rec_neg)
        algo_clean[algo_name] = pd.concat([algo_neg_merged,algo_pos_merged], sort=False).sort_values('event')

        algo_clean[algo_name]['matches']=algo_clean[algo_name]['matches'].replace(np.nan, True)
        #  algo_clean[algo_name].drop(columns=['best_match'], inplace=True)

    #save files to savedir in HDF
    store = pd.HDFStore(output_name, mode='w')
    store['gen_clean'] = gen_clean
    for algo_name, df in algo_clean.items():
        store[algo_name] = df
    store.close()

    cmd = 'xrdcp --silent -p -f {outputname} {outputdir}/{outputname}'.format(
        outputname=output_name, outputdir=xrd_prefix(md['joboutputdir'])[0][0])        
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metadata',
                        default='metadata.json',
                        help='Path to the metadata file. Default:%(default)s')
    parser.add_argument('jobid', type=int, help='Index of the output job.')

    args = parser.parse_args()
   
    # Loading configuration parameters
    import json
    with open(args.metadata) as fp:
        md = json.load(fp)
    
    preprocessing(md)

