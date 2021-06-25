#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-2:00
#SBATCH --output=%N-%j.out
#SBATCH --array=1-4
#SBATCH --requeue  

collaborators_list=(
    "01020304"
    "05060708"
    "09101112"
    "1314151617"
)


module load python/3.7 cuda cudnn
SOURCEDIR=~/scratch/FETS/Plain/scripts



lockdir=/home/lakshayt/scratch/FETS/Plain/scripts/data.lock

while true; do
    if mkdir "$lockdir"
    then
        echo 'successfully acquired lock for tar\n'
        mkdir $SLURM_TMPDIR/Data
        tar xf /home/lakshayt/scratch/FETS/Data/FETS.tar -C $SLURM_TMPDIR/Data
        # Remove lockdir when the script finishes, or when it receives a signal
        trap 'rm -rf "$lockdir"' 0    # remove directory when script finishes
        rm -r "$lockdir"
        break
    else
        echo 'cannot acquire lock, waiting %s\n' "$lockdir"
        sleep 120
    fi
done


source ~/anaconda3/etc/profile.d/conda.sh
source $SOURCEDIR/new_env/bin/activate

mkdir $SLURM_TMPDIR/Data
tar xf /home/lakshayt/scratch/FETS/Data/FETS.tar -C $SLURM_TMPDIR/Data

collaborator=${collaborators[$SLURM_ARRAY_TASK_ID-1]}



echo "ROUND NUMBER = "
echo $round_num 
echo "SUBJOB NUMBER"
echo $SLURM_ARRAY_TASK_ID
echo "COLLABORATORS TO BE TRAINED"
collaborator=${collaborators_list[$SLURM_ARRAY_TASK_ID-1]}
echo $collaborator
echo "SAVE PATH"
echo $save_path
echo "DATA PATH"
echo $SLURM_TMPDIR/Data/


ctr=0
while [ $ctr -le 2 ]
do
    python ~/scratch/FETS/Plain/part_main.py $collaborator $SLURM_TMPDIR/Data/ $round_num $save_path 
    if [[ $? = 0 ]]; then
        echo "success finished for $SLURM_ARRAY_TASK_ID"
        break
    else
        echo "failure retrying  $SLURM_ARRAY_TASK_ID time $ctr"
        ((ctr++))
        sleep 120
    fi
done
