#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name -
#BSUB -J DDPM_train_no_cfg
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 01:00
# specify system resources
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
### -- NVlink required only for multigpu training [sxm2]
### BSUB -R "select[sxm2]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process" 
#BSUB -o batch_output/train_%J.out
#BSUB -e batch_output/train_%J.err
# -- end of LSF options --

source ~/.ADLCV_venv/bin/activate

python -u classifier_train.py