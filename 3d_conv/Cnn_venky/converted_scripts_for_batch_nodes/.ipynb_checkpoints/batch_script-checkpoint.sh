#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH --job-name=ice_cube_cnn_train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=haswell
#SBATCH --account=nstaff
#SBATCH --image=custom:pdsf-chos-sl64:v5
#################


echo "--start date" `date` `date +%s`
source activate v_py3
python --version
which python
### Actual script to run
##shifter python train.py 
source deactivate v_py3
echo "--end date" `date` `date +%s`