from glob import glob
import itertools

# Flag to test locally
local = False

# clustering
clustering_script = 'matching.py'
clustering_option = 1 # 1 for matching

# DeltaR matching threshold
threshold = 0.05

# Input files
files = {
    'photons': glob('/eos/uscms/store/user/dnoonan/SinglePhoton_PT2to200/crab_AE_photons_3_23_2/210605_201932/0000/*.root'),
    'electrons': glob('/eos/uscms/store/user/dnoonan/SingleElectron_PT2to200/crab_AE_electrons_3_23_2/210605_201806/0000/*.root'),
    'pions': [],
}
if local:
    files['photons'] = files['photons'][:1] if len(files['photons'])>0 else []
    files['electrons'] = files['electrons'][:1] if len(files['electrons'])>0 else []
    files['pions'] = files['pions'][:1] if len(files['pions'])>0 else []

# Pick one of the different algos trees to retrieve the gen information
gen_tree = 'FloatingpointThreshold0DummyHistomaxxydr015GenmatchGenclustersntuple/HGCalTriggerNtuple'

# STore only information on the best match
bestmatch_only = True

# Output directories
eos_output_dir = '/eos/uscms/store/user/naodell/HGCAL/study_autoencoder/'
job_output_dir = '3_22_1/electron_photon_signaldriven/'

file_per_batch = {
    'electrons': 5,
    'pions': 2,
    'photons': 2,
}

algo_trees = {}
# List of ECON algorithms
fes = ['Threshold0', 'Threshold', 'Mixedbcstc',
       'AutoEncoderTelescopeMSE', 'AutoEncoderStride',
       'AutoEncoderQKerasTTbar', 'AutoEncoderQKerasEle',
]

ntuple_template = 'Floatingpoint{fe}Dummy{be}GenmatchGenclustersntuple/HGCalTriggerNtuple'
algo_trees = {}
for fe in fes:
    be = 'Histomaxxydr015'
    algo_trees[fe] = ntuple_template.format(fe=fe, be=be)
