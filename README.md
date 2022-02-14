# ECON-T algorithm testing scripts

Scripts and notebooks in this package should be run with python 3 (they have been tested with python 3.7). The main dependencies are:
- scipy
- numpy
- pandas
- uproot (version 4)
- scikit-learn
- xgboost

## Setup (tentative instructions for upcoming pull request)

The analysis code here is written to run in standalone python, i.e., no CMSSW dependence.  The current environment configuration may change.  To source the environment run the following:

```
source scripts/setup.sh
```

This will setup the python virtual environment `hgcalPythonEnv` and create a directory with all the necessary libraries.  To activate the environment, run:

```
source hgcalPythonEnv/bin/activate
```

## Running matching script

The first step in analyzing HGCal data is to produce ntuples using CMSSW to be processed here.  The matching script (`scripts/matching.py`) associates generator level particles to clusters that are produced under various different readout algorithm scenarios defined in the ntuple configuration.  The matching is done by finding the generator particle that is closest in dR to each cluster subject to the requirements:

   * cluster pt > 5 GeV
   * dR < 0.05
   * ...

The output is a dictionary of dataframes, one for each algorithm that is specified and one for generator particles, stored in a pickle file.  The default configuration for running the script is in `config/matching_cfg.yaml`.   An alternative configuration can be passed using the `--config` option when executing the script).  The configuration allows for specification of all necessary information for running the scripts such as input files, output directory, variables to store in the output, etc.  

To run the script with default configuration, run the following:

```
python scripts/matching.py
```

This will run over the single photon samples specificed in `data/photons_nopu_ntuples.txt`, and will produce a file in the `data` directory, `output_0.pkl`, that contains the dictionary of dataframes.

To run over multiple files using condor, use `scripts/condor_submit.py`.  The default configuration is in `config/batch_cfg.yaml` which is mainly useful for specifying the location of input files and where the output should be transferred to on exit.  Both the input and output should be located somewhere on EOS for now.  The commands that are run on the batch node are specified in `scripts/batch_executable.sh`.  The script is mainly responsible for sourcing the computing environment configuration and executing the matching script (or, eventually, whatever script you might want to run).

## Setup for juptyer notebooks
For running the notebooks that analyze the pandas dataframes.

If you are able, you can create a conda environment locally:
```
conda create -n econ-ae python=3.7 # note that 3.7 is important to have dataframe compatibility (otherwise run dataframes w. python 3.8)
conda activate econ-ae
pip install numpy pandas scikit-learn scipy matplotlib uproot coffea jupyterlab tables
pip install "xgboost==1.3.3"
```

or you can use JupyterHub. For this, point your browser to:
https://jupyter.accre.vanderbilt.edu/

Click the "Sign in with Jupyter ACCRE" button. On the following page, select CERN as your identity provider and click the "Log On" button. Then, enter your CERN credentials or use your CERN grid certificate to authenticate. Select a "Default ACCRE Image v5" image and then select either 1 core/2 GB or memory or 8 cores/8GB memory. Unless you are unable to spawn a server, we recommend using the 8 core/8GB memory servers as notebooks 4 and 5 require a lot of memory. Once you have selected the appropriate options, click "Spawn".

Now you should see the JupyterHub home directory. Click on "New" then "Terminal" in the top right to launch a new terminal.

Once any of these steps are done (conda or jupyter in vanderbilt) then you can clone the repository:
```
git clone git@github.com:cmantill/ECONAutoencoderStudy.git
```

And then, download the input data (processed with the configuration files in `fragments`), e.g.:
```
cd notebooks/
mkdir data/
mkdir img/
scp -r cmslpc-sl7.fnal.gov:/eos/uscms/store/user/cmantill/HGCAL/study_autoencoder/3_22_1/ data/
```

For the 2nd and so on notebooks you will need python 3.8, as well as scikit-learn (0.24.1 ?).

## Description of input data for physics studies

We first need to simulate the HGCAL trigger cells using the ECON-T algorithms. For this we use simulated datasets (photons, electrons, pileup).
Full documentation can be found [here](https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerPrimitivesSimulation).
This repository is used to simulate all the ECON-T algorithms in the CMS official simulation (cmssw).

We are currently running with TPG release v3.23.3_1130.
A short history of releases is the following:
- v3.22.1. (had bug on normalization fixed by danny [here](https://github.com/PFCal-dev/cmssw/commit/65625ee12e0c1a527820d20aeaaa656cf6f4df48#diff-0003f7b8caf7041ba5afce04bcfa74b1a2593d991fc3b5b84294d5ee9e680ae4)
- v3.23.3_1130 (fixes in for AE)

### Configuration files
To run this we need configuration files. These are the following:
- For 200PU electron gun ( /SingleElectron_PT2to200/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3_ext2-v2/GEN-SIM-DIGI-RAW) and 0PU photon gun (/SinglePhoton_PT2to200/Phase2HLTTDRWinter20DIGI-NoPU_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW):
`produce_ntuple_std_ae_xyseed_reduced_genmatch_v11_cfg.py`

- For 200PU MinBias (/MinBias_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3/GEN-SIM-DIGI-RAW/):
`produce_ntuple_std_ae_xyseed_reduced_pt5_v11_cfg.py`

### AutoEncoder implementation 
The implementation of the AutoEncoder (AE) in CMSSW is in [HGCalConcentratorAutoEncoderImpl.cc](https://github.com/PFCal-dev/cmssw/blob/v3.23.3_1130/L1Trigger/L1THGCal/src/concentrator/HGCalConcentratorAutoEncoderImpl.cc):
- The [`select` function](https://github.com/PFCal-dev/cmssw/blob/v3.23.3_1130/L1Trigger/L1THGCal/src/concentrator/HGCalConcentratorAutoEncoderImpl.cc#L122-L174) gets called once per event per wafer.
- It first loops over the trigger cells, remaps from the TC U/V coordinates to the 0-47 indexing we have been using for the training, then fills the mipPt list.
```  
for (const auto& trigCell : trigCellVecInput) {
    ...
    modSum += trigCell.mipPt();
}
```
- Then it normalizes the mipPt list, and quantizes it. 
- Puts stuff into tensors and [runs the encoder with tensorflow](https://github.com/PFCal-dev/cmssw/blob/v3.23.3_1130/L1Trigger/L1THGCal/src/concentrator/HGCalConcentratorAutoEncoderImpl.cc#L198-L225)
- [Runs the decoder](https://github.com/PFCal-dev/cmssw/blob/v3.23.3_1130/L1Trigger/L1THGCal/src/concentrator/HGCalConcentratorAutoEncoderImpl.cc#L227-L248)
- Loops over decoded values, and [puts them back into trigger cell objects](https://github.com/PFCal-dev/cmssw/blob/v3.23.3_1130/L1Trigger/L1THGCal/src/concentrator/HGCalConcentratorAutoEncoderImpl.cc#L256-L304) (which is what the backend code is expecting and uses)
- There are different configuration options, to allow multiple trainings for different number of eLinks in the [hgcalConcentratorProducer](https://github.com/PFCal-dev/cmssw/blob/v3.23.3_1130/L1Trigger/L1THGCal/python/hgcalConcentratorProducer_cfi.py#L184-L226)

[comment]: <> (Danny's config /uscms/home/dnoonan/work/HGCAL/CMSSW_11_2_0_pre5/src/L1Trigger/L1THGCalUtilities/test/NewTrainings_QKeras_cfg.py)
[comment]: <> (it requires the models dir /uscms/home/dnoonan/work/HGCAL/CMSSW_11_2_0_pre5/src/L1Trigger/L1THGCalUtilities/test/AEmodels)

## Notebooks:

- `electron_photon_calibration_autoencoder_210611.ipynb`:
   - Derives layer weight correction factors with 0PU unconverted photons
   - Derives $\eta$ dependent linear energy correction (this is an additive correction) with 200PU electrons
   - Produces energy scale and resolution plots, in particular differentially vs $|\eta|$ and $p_T$

- `electron_pu_bdt_tuning_autoencoder_210611.ipynb`: 
   - Finds the set of hyperparameters to be used later in the training of BDT (discriminator between electrons and PileUp). XGBOOST is used to train the BDTs.
      - Scans the L1 and L2 regularization parameters. 
      - Scans the learning rate. 
      - Scans the BDT tree depth. 
   - Checks the behaviour of the BDT as a function of the number of boosting steps. 
   - Checks for overtraining with a final set of hyperparameters. The notebook focuses on the limitation of overtraining rather than optimal performance. This hyperparameter tuning is currently done by hand, and some automatization could be implemented. 

- `electron_pu_autoencoder_210611.ipynb`: 
   - Performs the final BDT ID training on the full sample
   - Computes the signal efficiencies as a function of $\eta$ and $p_T$, for a 99% inclusive signal efficiency working point.
- `electron_turnon_autoencoder_210611.ipynb`: 
   - Computes the trigger efficiency turn on curves (using the energy corrections and the BDT ID)
   - The turn-on curves are finally used to extract the L1 $\to$ offline threshold mappings, which will be used to compare L1 rates as a function of the so-called offline threshold. In our case this offline threshold is defined as the gen-level $p_T$ at which the turnon reaches 95% efficiency.

- `egamma_rates_autoencoder_210611.ipynb`: 
   - Extracts the rate and plots the rate as a function of the offline threshold.
   - These are the final plots used to compare the different algorithms.

