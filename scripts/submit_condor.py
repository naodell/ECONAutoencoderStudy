#!/usr/bin/env python
import os
import subprocess
import datetime
import argparse
import textwrap
import yaml
from pathlib import Path

def get_current_time():
    now = datetime.datetime.now()
    currentTime = '{0:02d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

    return currentTime

def xrd_prefix(filepath):
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
    filepath = f'{prefix}/{filepath}' 

    return filepath

def make_file_batches(process, batch_config):
    file_location = batch_config['location']
    files_per_job = batch_config['files_per_job']

    file_list = [xrd_prefix(f'{file_location}/{f}') for f in os.listdir(file_location) if f[-4:] == 'root']
    if len(file_list) == 0:
        return None

    n_files = len(file_list)
    file_split = [file_list[i:i + files_per_job] for i in range(0, n_files, files_per_job)]

    return file_split

def prepare_submit(process, batches, output_dir, executable):

    # Writing the batch config file
    batch_filename = f'.batch_tmp_{process}'
    batch_tmp = open(batch_filename, 'w')
    batch_tmp.write(textwrap.dedent('''\
        Universe              = vanilla
        Should_Transfer_Files = YES
        WhenToTransferOutput  = ON_EXIT
        want_graceful_removal = true
        transfer_output_files = ""
        Notification          = Never
        Requirements          = OpSys == "LINUX"&& (Arch != "DUMMY" )
        request_disk          = 2000000
        request_memory        = 32000
        \n
    '''))

    output_dir = xrd_prefix(str(output_dir))
    for i, batch in enumerate(batches):

        ## make file with list of inputs ntuples for the analyzer
        input_file = open(f'input_{process}_{i}.txt', 'w')
        input_file.writelines(f'{f}\n' for f in batch)
        input_file.close()

        ### set output directory
        batch_tmp.write(textwrap.dedent(f'''\
            Executable            = {executable}
            Arguments             = {i} input_{process}_{i}.txt {process} {output_dir}
            Transfer_Input_Files  = source.tar.gz, input_{process}_{i}.txt
            Output                = reports/{process}_{i}_$(Cluster)_$(Process).stdout
            Error                 = reports/{process}_{i}_$(Cluster)_$(Process).stderr
            Log                   = reports/{process}_{i}_$(Cluster)_$(Process).log
            Queue
            \n
        '''))

    batch_tmp.close()

    cmd = f'condor_submit {batch_filename}'
    return cmd
        
def prepare_output(output_dir, prefix):
    time_string = get_current_time()
    output_dir = Path(f'{output_dir}/{prefix}_{time_string}')
    output_dir.mkdir(parents=True)

    stage_dir = Path(f'batch/{prefix}_{time_string}')
    stage_dir.mkdir(parents=True)

    report_dir = stage_dir / Path('reports')
    report_dir.mkdir()

    return output_dir, stage_dir
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', 
                        type=str,
                        help="Provide configuration file for submission of batch jobs."
                      )
    args = parser.parse_args()

    # Load configuration file
    with open(args.config, 'r') as config_file: 
        config = yaml.safe_load(config_file)

    current_dir = os.getcwd()
    executable = config['executable']
    output_dir, stage_dir = prepare_output(config['output_dir'], config['prefix'])
    os.system(f'tar czf {stage_dir}/source.tar.gz . --exclude="*.hdf5" --exclude="batch" --exclude="data" --exclude-vcs')
    os.system(f'cp {current_dir}/{executable} {stage_dir}/.')
    os.chdir(stage_dir)

    print('Preparing to submit jobs:\n')
    for process, batch_config in config['inputs'].items():
        batches = make_file_batches(process, batch_config)
        if not batches:
            print('No files found for process {process}!')
            continue

        cmd = prepare_submit(process, batches, output_dir, executable.split('/')[-1])
        print(f'{cmd}')
        subprocess.call(cmd, shell=True)
    
    os.chdir(current_dir)
