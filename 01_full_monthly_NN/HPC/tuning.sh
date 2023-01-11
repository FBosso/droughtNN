#!/bin/bash
#SBATHC --job-name=test
#SBATHC -N 1
#SBATHC --time 1:00:00
#SBATHC -N 1
#SBATHC --cpus-per-task 2
#SBATHC --mem=16G
#SBATHC --partition thin
#SBATHC --output=out/$SLURM_ARRAY_TASK_ID.out
#SBATHC --error=errors/$SLURM_ARRAY_TASK_ID.err

cp -r $HOME/code/droughtNN/01_full_monthly_NN/datasets "$TMPDIR"
cp -t $HOME/code/droughtNN/01_full_monthly_NN/HPC "$TMPDIR"

FILES=$(ls $TMPDIR/datasets)

for FILE in $FILES
do
	python keras-tuner_model_selection.py $FILE $TMPDIR "/scratch-shared/models"
done