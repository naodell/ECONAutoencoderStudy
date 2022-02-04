from glob import glob
import pickle
import numpy as np

# Flag to test locally
local = False

# clustering option
clustering_script = 'clusters2hdf.py'
clustering_option = 0

files = {
    'MinBias': glob('/eos/uscms/store/user/cmantill/HGCAL/AE_Jun11/MinBias_TuneCP5_14TeV-pythia8/crab_AE_minBias_3_23_2/210611_191016/0000/ntuple_*.root')
}

# if local test, use only one input file
if local:
    files['MinBias'] = files_batch['MinBias'][:1]

eos_output_dir = '/eos/uscms/store/user/naodell/hgcal/econ/study_autoencoder/'
job_output_dir = '3_22_1/pu_discri_signaldriven/'

file_per_batch = {'MinBias': 1}

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

# Clusters with a corrected pT below 0GeV are cut
# Note: there's a 5GeV pT cut applied at the ntuple level on the uncorrected pT
pt_cut = 0

# Store only the maximum pT cluster (corrected pT)
# Note: it is the maximum pT cluster among the clusters passing the ID
store_max_only = True

# Load energy calibration/correction data
data_dir = 'data'
data_tag = '210611'
with open('%s/layer_weights_photons_autoencoder_%s.pkl'%(data_dir,data_tag), 'rb') as f:
    calibration_weights = pickle.load(f)

for key,item in calibration_weights.items():
    if isinstance(item, np.ndarray):
        calibration_weights[key] = item.tolist()

with open('%s/lineareta_electrons_autoencoder_%s.pkl'%(data_dir,data_tag), 'rb') as f:
    correction_cluster = pickle.load(f)

additive_correction = True
correction_inputs = ['cl3d_abseta']

# Load identification data
with open('%s/xgboost_electron_pu_autoencoder_%s.pkl'%(data_dir,data_tag), 'rb') as f:
    boosters = pickle.load(f)
bdts = boosters['extended']

# Apply ID selection
# 99% signal efficiency working point
with open('%s/xgboost_threshold_electron_pu_autoencoder_%s.pkl'%(data_dir,data_tag), 'rb') as f:
    thresholds = pickle.load(f)

with open('%s/xgboost_tpr_electron_pu_autoencoder_%s.pkl'%(data_dir,data_tag), 'rb') as f: 
    tprs = pickle.load(f)

wp = 0.99
working_points = {}
for name in algo_trees.keys():
    working_points[name] = np.interp(wp, tprs['extended'][name], thresholds['extended'][name])
