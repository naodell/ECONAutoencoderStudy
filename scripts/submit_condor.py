#!/usr/bin/env python
import os
from datetime import date
from datetime import datetime
import subprocess
import time
import optparse
import json
import pickle

def job_version(workdir):
    version_date = "v_1_"+str(date.today())
    if os.path.isdir(workdir):
        dirs= [f for f in os.listdir(workdir) if os.path.isdir(os.path.join(workdir,f)) and f[:2]=='v_']
        version_max = 0
        for d in dirs:
            version = int(d.split("_")[1])
            if version > version_max: 
                version_max = version
        version_date = "v_"+str(version_max+1)+"_"+str(date.today())
    return version_date

def batch_files(files, file_per_batch):
    batches={}
    j=0
    batches[j]=[]
    for i, filename in enumerate(files):
        filename = filename.replace('/eos/uscms/','root://cmseos.fnal.gov//')
        batches[j].append(filename)
        if i%file_per_batch == 0 and i+1<len(files):
            j+=1
            batches[j]=[]
    return batches
        
def prepare_metadata(name, batches, output_dir, eos_output_dir, param, reachedEE): 
    md = dict()
    md['algo_trees'] = param.algo_trees
    md['clustering'] = param.clustering_script
    md['clustering_option'] = param.clustering_option
    md['joboutputdir'] = eos_output_dir
    md['name'] = name
    md['inputfiles'] = []
    md['jobs'] = []

    for idx,b in enumerate(batches.keys()):
        md['inputfiles'].append(batches[b])
        md['jobs'].append({'idx': idx, 'inputfiles': batches[b]})

    if param.clustering_option == 1:
        md['bestmatch_only'] = param.bestmatch_only
        md['reachedEE'] = reachedEE
        md['gen_tree'] = param.gen_tree
        md['threshold'] = param.threshold

        metadata_file_name = '{0}/metadata.json'.format(output_dir)
        with open(metadata_file_name,'w') as f:
            json.dump(md, f, ensure_ascii=True, indent=2, sort_keys=True)
    else:
        md['bdts'] = param.bdts
        md['working_points'] = param.working_points
        md['correction_cluster'] = param.correction_cluster
        md['correction_inputs'] = param.correction_inputs
        md['calibration_weights'] = param.calibration_weights
        md['store_max_only'] = param.store_max_only
        md['additive_correction'] = param.additive_correction
        md['pt_cut'] = param.pt_cut

        metadata_file_name = '{0}/metadata.pkl'.format(output_dir)
        with open(metadata_file_name, 'wb') as f:
            pickle.dump(md, f)

    return md,metadata_file_name

def prepare_submit(name, batches, job_dir, md, md_name):
    current = os.getcwd()
    njobs = len(md['jobs'])
    jobids = [str(jobid) for jobid in range(njobs)]
    jobids_file = os.path.join(job_dir, 'submit.txt')

    with open(jobids_file, 'w') as f:
        f.write('\n'.join(jobids))

    # prepare the list of files to transfer
    script_file_name = os.path.join(os.path.dirname(__file__), 'run_processor.sh')
    clustering_file_name =  os.path.join(os.path.dirname(__file__), md['clustering'])
    files_to_transfer = [md_name, script_file_name, clustering_file_name]
    files_to_transfer = [os.path.abspath(f) for f in files_to_transfer]

    # condor jdl
    condordesc = '''\
    universe              = vanilla
    requirements          = (Arch == "X86_64") && (OpSys == "LINUX")
    request_memory        = {request_memory}
    request_disk          = 10000000
    executable            = {scriptfile}
    arguments             = {clustering} $(jobid)
    transfer_input_files  = {files_to_transfer}
    output                = {jobdir}/logs/$(jobid).out
    error                 = {jobdir}/logs/$(jobid).err
    log                   = {jobdir}/logs/$(jobid).log
    use_x509userproxy     = true
    Should_Transfer_Files = YES
    initialdir            = {initialdir}
    WhenToTransferOutput  = ON_EXIT
    want_graceful_removal = true
    periodic_release      = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 10*60)
    {transfer_output}
    queue jobid from {jobids_file}
    '''.format(scriptfile=os.path.abspath(script_file_name),
               files_to_transfer=','.join(files_to_transfer),
               clustering=md['clustering_option'],
               jobdir=os.path.abspath(job_dir),
               # when outputdir is on EOS, disable file transfer as file is manually copied to EOS in processor.py
               initialdir=os.path.abspath(job_dir),
               transfer_output='transfer_output_files = ""',
               jobids_file=os.path.abspath(jobids_file),
               request_memory=4000
           )

    condorfile = os.path.join(job_dir, 'submit.cmd')
    with open(condorfile, 'w') as f:
        f.write(condordesc)

    cmd = 'condor_submit {condorfile}'.format(condorfile=condorfile)
    return cmd
        
def prepare_jobs(param, batches, key):
    files = param.files[key]
    output_dir = param.job_output_dir

    job_output_dir = 'condor/'+param.job_output_dir
    eos_output_dir = param.eos_output_dir + param.job_output_dir

    version=job_version(output_dir)
    out_dir=job_output_dir+'/'+version+'/'+key
    out_dir_eos=eos_output_dir+'/'+version+'/'+key

    if not os.path.exists(out_dir): os.makedirs(out_dir)
    if not os.path.exists(out_dir+'/logs'): os.makedirs(out_dir+'/logs')

    md,md_name = prepare_metadata(key, batches, out_dir, out_dir_eos, param, reachedEE=2)
    cmd = prepare_submit(key, batches, out_dir, md, md_name)

    return cmd
    
def main(parameters_file):
    # Loading configuration file
    import importlib
    parameters = importlib.import_module(parameters_file)
    local = parameters.local
    cmds = ''

    for key,filelist in parameters.files.items():
        files = filelist
        file_per_batch = parameters.file_per_batch[key]

        if len(files)>0:
            batches = batch_files(files, file_per_batch)
      
            # Preparing jobs working directory
            cmd  = prepare_jobs(parameters, batches, key)

            cmds += cmd + ' \n  '

    print('Run the following command to submit the jobs:\n  %s' % cmds)
    
if __name__=='__main__':
    parser = optparse.OptionParser()
    parser.add_option("--cfg",type="string", dest="param_file", help="select the parameter file")
    (opt, args) = parser.parse_args()
    parameters=opt.param_file
    main(parameters)
    
