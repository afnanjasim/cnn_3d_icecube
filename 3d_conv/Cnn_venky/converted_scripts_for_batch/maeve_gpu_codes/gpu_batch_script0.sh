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
export HDF5_USE_FILE_LOCKING=FALSE
# Limit to one GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
### Actual script to run
echo $2
python3 Cnn_train.py --test --train --typeofdata $1 --model_list $2 --epochs 100
#python3 Cnn_train.py --typeofdata $1 --model_list $2
echo "--end date" `date` `date +%s`
