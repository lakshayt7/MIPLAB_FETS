#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=2  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=3200M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
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

collaborators_list=(
    "01020304"
    "05060708"
    "09101112"
    "1314151617"
)

# path is where we save the models both collaborator and aggregate  

path=/home/lakshayt/scratch/FETS/Plain/parallel_models/test_run/

# Loop through the number of rounds

for cy in {1..2}
do
    # Turns the iterator (integer?) into a padded string with 4 characters
    # This matches the naming convention followed in the rest of the script
    printf -v CYCLE_ID "%04d" $cy
    
    # Record the cycle start time
    START_TIME=$(date +"%T")
    echo "Cycle $CYCLE_ID started at : $START_TIME"
    
    sleep 10

    # Schedule each subjob for running and pass it the collaborators it needs to train as well as other other stuff like path to data
    for run in {1..4}
    do
        if (( $run == 4 ))
        then
            sbatch --wait --export=round_num=$cy,save_path=$path,collaborators=${collaborators_list[$run-1]} /home/lakshayt/scratch/FETS/Plain/scripts/parallel_subrunner.sh
        else
            sbatch --export=round_num=$cy,save_path=$path,collaborators=${collaborators_list[$run-1]} /home/lakshayt/scratch/FETS/Plain/scripts/parallel_subrunner.sh
        fi
        # Sleep put in for giving gap between job scheduling
        sleep 180
    done

    sleep 120
    
    # aggregation code which aggregates and validates

    python aggregation.py $cy $path $collaborators_list
    
    # Record the cycle end time
    END_TIME=$(date +"%T")
    echo "  Cycle $CYCLE_ID ended at : $END_TIME"
done
