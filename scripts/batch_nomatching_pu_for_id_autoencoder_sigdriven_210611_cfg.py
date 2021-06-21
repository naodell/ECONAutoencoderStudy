from glob import glob
import pickle
import numpy as np

# Flag to test locally
local = False

# Clustering option
clustering_script = 'clusters2hdf.py'
clustering_option = 0

files_batch = {
    'MinBias': glob('/eos/uscms/store/user/dnoonan/MinBias_TuneCP5_14TeV-pythia8/crab_AE_minBias_3_23_2/210605_202018/0000/ntuple_*.root')
}

# if local test, use only one input file
if local:
    files_batch['MinBias'] = files_batch['MinBias'][:1]

eos_output_dir = '/eos/uscms/store/user/cmantill/HGCAL/study_autoencoder/'
job_output_dir = '3_22_1/pu_for_id_signaldriven/'

file_per_batch = {'MinBias': 6}

# List of ECON algorithms
fes = ['Threshold0', 'Threshold', 'Mixedbcstc',
        'AutoEncoderTelescopeMSE', 'AutoEncoderStride',
        'AutoEncoderQKerasTTbar', 'AutoEncoderQKerasEle',
        ]

ntuple_template = 'Floatingpoint{fe}Dummy{be}Genclustersntuple/HGCalTriggerNtuple'
algo_trees = {}
for fe in fes:
    be = 'Histomaxxydr015'
    algo_trees[fe] = ntuple_template.format(fe=fe, be=be)

# Preselection pT cut, after calibration
pt_cut = 20
# Store all clusters passing the preselection
store_max_only = False

# Load energy calibration/correction data
data_dir = '/eos/uscms/store/user/cmantill/HGCAL/data/'
data_tag = '210430'
with open('%s/layer_weights_photons_autoencoder_%s.pkl'%(data_dir,data_tag), 'rb') as f:  
    calibration_weights = pickle.load(f)
with open('%s/lineareta_electrons_autoencoder_%s.pkl'%(data_dir,data_tag), 'rb') as f:
    correction_cluster = pickle.load(f)

additive_correction = True
correction_inputs = ['cl3d_abseta']

# No ID selection is applied in the preprocessing
bdts = None
working_points = {}
for name in algo_trees.keys():
    working_points[name] = -999