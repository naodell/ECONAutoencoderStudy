#!/bin/bash

jobid=$1
infile=$2
process=$3
output_dir=$4
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt
NAME=hgcalPythonEnv

echo $jobid 
echo $infile
echo $process
echo $output_dir

# setup local environment
tar xzf source.tar.gz
rm source.tar.gz
source /cvmfs/cms.cern.ch/cmsset_default.sh
source $LCG/setup.sh
source $NAME/bin/activate

mkdir data
ls -lh
python -V
#mv ${infile} data/photons_nopu_ntuples.txt
#python scripts/matching.py --config config/matching_cfg.yaml --job_id ${jobid} --input_file ${infile} --output_dir ./data --is_batch
python scripts/tc_matching.py --config config/tc_matching_cfg.yaml --job_id ${jobid} --input_file ${infile} --is_batch
ls data

#python -c 'import pandas as pd; store = pd.HDFStore("test.hdf5", mode="w");store.close()'
#xrdcp -f data/output_${jobid}.hdf5 ${output_dir}/output_${process}_${jobid}.hdf5
xrdcp -f data/output_${jobid}.pkl ${output_dir}/output_${process}_${jobid}.pkl

status=$?
exit $status
