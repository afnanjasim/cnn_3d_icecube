#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --job-name=ice_cube_cnn_train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=haswell
#SBATCH --account=nstaff
#################


echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME
conda activate v_py3
python --version
which python
export HDF5_USE_FILE_LOCKING=FALSE
### Actual script to run
python Cnn_train.py --test --train --typeofdata $1 --model_list $2
conda deactivate
echo "--end date" `date` `date +%s`

