#!/bin/bash

# Hyperparameter ranges
LEARNING_RATES=("1.3e-3")
CORRECTOR_LEARNING_RATES=("1.3e-3")

BATCH_SIZES=(8)
NGRAM_SIZES=(64)
PADDINGS=(0)  # Add different padding values if needed
LEVELS=("sentence")
EMBEDDING_MODELS=("text-embedding-ada-002" "gtr-t5-base")
TEMPERATURES=(0.1)
STRIDES=(5)
MARGINS=(1.0)
ALPHAS=(0.25)
N_STEPS=(1)
# Number of epochs
EPOCHS=100

# Subjects list, including the combination of all subjects
SUBJECTS=("Subject1 Subject2 Subject3")

# Define the path to your training script
TRAIN_SCRIPT="train.py"

# Path to save logs
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

# WandB project name
WANDB_PROJECT="neuro2semantic_project"

# Bands list
BANDS=("highgamma")  # Add more bands if needed

# Trials to leave out for evaluation
LEAVE_OUTS=("5 8 15 19 24 28")

# Loop through each combination of hyperparameters and subject configurations
for NSTEP in "${N_STEPS[@]}"; do
    for CLR in "${CORRECTOR_LEARNING_RATES[@]}"; do
        for MARG in "${MARGINS[@]}"; do 
            for ALPHA in "${ALPHAS[@]}"; do
                for STRIDE in "${STRIDES[@]}"; do
                    for LEAVE_OUT in "${LEAVE_OUTS[@]}"; do
                        for SUBJECT in "${SUBJECTS[@]}"; do
                            for LR in "${LEARNING_RATES[@]}"; do
                                for BS in "${BATCH_SIZES[@]}"; do
                                    for BAND in "${BANDS[@]}"; do
                                        for NGRAM in "${NGRAM_SIZES[@]}"; do
                                            for PADDING in "${PADDINGS[@]}"; do
                                                for LEVEL in "${LEVELS[@]}"; do
                                                    for EMBEDDING_MODEL in "${EMBEDDING_MODELS[@]}"; do
                                                        for TEMP in "${TEMPERATURES[@]}"; do 
                                                        
                                                            # Replace spaces with underscores for trial indices
                                                            LEAVE_OUT_STR=$(echo ${LEAVE_OUT// /_})
                                                            
                                                            # Define a unique run name based on hyperparameters
                                                            RUN_NAME="level_${LEVEL}_emb_${EMBEDDING_MODEL}_clr_${LR}_nsteps_${NSTEP}_alpha_${ALPHA}_leave_out_${LEAVE_OUT_STR}_epochs_${EPOCHS}_stride_${STRIDE}"
                                                            
                                                            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=neuro2semantic_${RUN_NAME}
#SBATCH --output=${LOG_DIR}/${RUN_NAME}.out
#SBATCH --error=${LOG_DIR}/${RUN_NAME}.err
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=24:00:00

# Load environment or activate conda environment
source /path/to/conda.sh
conda activate your_env_name


# Run the training script with specified parameters
python $TRAIN_SCRIPT \
  --learning_rate $LR \
  --corrector_lr $CLR \
  --batch_size $BS \
  --num_epochs $EPOCHS \
  --wandb_run_name $RUN_NAME \
  --wandb_project $WANDB_PROJECT \
  --subjects $SUBJECT \
  --bands $BAND \
  --ngram $NGRAM \
  --padding $PADDING \
  --level $LEVEL \
  --embedding_model_name $EMBEDDING_MODEL \
  --temperature $TEMP \
  --stride $STRIDE \
  --leave_out_trials $LEAVE_OUT \
  --n_steps $NSTEP \
  --margin $MARG \
  --alpha $ALPHA
EOT
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
