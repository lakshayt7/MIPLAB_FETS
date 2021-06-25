#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-36:00
#SBATCH --output=%N-%j.out


module load python/3.7 cuda cudnn
SOURCEDIR=~/scratch/FETS/Plain/

source ~/anaconda3/etc/profile.d/conda.sh


mkdir -p new_env
tar -xzf $SOURCEDIR/fets_env.tar.gz -C new_env
./new_env/bin/python
source new_env/bin/activate
conda-unpack

# list of collaborators for each subjob

collaborators_list="0102030405060708091011121314151617"


# path is where we save the models both collaborator and aggregate  

path=/home/lakshayt/scratch/FETS/Plain/parallel_models/test_run/

mkdir $SLURM_TMPDIR/Data
tar xf /home/lakshayt/scratch/FETS/Data/FETS.tar -C $SLURM_TMPDIR/Data

# Loop through the number of rounds

lockdir=/home/lakshayt/scratch/FETS/Plain/scripts/data.lock
rm -r lockdir

for cy in {1..20}
do
    # Turns the iterator (integer?) into a padded string with 4 characters
    # This matches the naming convention followed in the rest of the script
    printf -v CYCLE_ID "%04d" $cy
    
    # Record the cycle start time
    START_TIME=$(date +"%T")
    echo "Cycle $CYCLE_ID started at : $START_TIME"
    
    sleep 10

    # Schedule each subjob for running and pass it the collaborators it needs to train as well as other other stuff like path to data

    sbatch --wait --export=round_num=$cy,save_path=$path /home/lakshayt/scratch/FETS/Plain/scripts/parallel_subrunner.sh

    sleep 120
    
    AGG_TIME=$(date +"%T")
    echo "Training for round $CYCLE_ID finished at : $AGG_TIME"
    # aggregation code which aggregates and validates

    python ~/scratch/FETS/Plain/aggregation.py $cy $SLURM_TMPDIR/Data/ $path $collaborators_list
    
    # Record the cycle end time
    END_TIME=$(date +"%T")
    echo "  Cycle $CYCLE_ID ended at : $END_TIME"
done
