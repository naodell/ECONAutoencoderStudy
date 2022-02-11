#!/bin/bash

matching=$1
jobid=$2

echo $1
echo $2

# remove old
#source /cvmfs/cms.cern.ch/cmsset_default.sh
#rm *.tgz
#
#xrdcp -f root://cmseos.fnal.gov//store/user/$USER/CMSSW_11_3_0.tgz ./CMSSW_11_3_0.tgz
#tar -zxvf CMSSW_11_3_0.tgz
#rm *.tgz
#mkdir CMSSW_11_3_0/src
#cd CMSSW_*/src
#scram b ProjectRename
#eval `scramv1 runtime -sh`
#export PYTHONPATH=PYTHONPATH:"${CMSSW_BASE}/lib/${SCRAM_ARCH}"
#cd ../../
#
#xrdcp root://cmseos.fnal.gov//store/user/cmantill/uproot4env.tar.gz .
#tar -zxf uproot4env.tar.gz
#source uproot4env/bin/activate
#export PYTHONPATH=${PWD}:${PYTHONPATH}
#
#python3 -c "import uproot4; print(uproot4.__version__)"
#python -c "import uproot4; print(uproot4.__version__)"



if [ $matching == 1 ]
then
    python matching.py $jobid
else
    python3 clusters2hdf.py -m metadata.pkl $jobid
fi

status=$?

ls -l

exit $status

