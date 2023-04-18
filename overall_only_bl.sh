#!/bin/bash

# fetches the directory where the shell file resides
file_path=$(realpath "$0")
dir_path=$(dirname "$file_path")


#setting up the defaults
LANG='as,bn,en,gu,hi,kn,ml,mr,or,pa,ta,te'
GPUS=4
MODEL_DIR=$dir_path   #optional
PRE_PYTHON="/home2/aditya_hari/miniconda3/envs/old_py/bin/python"  #change required
PYTHON="/home2/aditya_hari/miniconda3/envs/multisent/bin/python"  #change required
SCRATCH_DIR=/tmp/XAlign
mkdir -p $SCRATCH_DIR
CHECKPOINT_PATH=$SCRATCH_DIR/checkpoint   #change required


BATCH_SIZE=4
MT5_BATCH_SIZE=4
TEST_MT5_BATCH_SIZE=4
TEST_BATCH_SIZE=4
PRE_EPOCHS=7
LR=1e-3
IS_TRIAL=1

# seq length related configuration
SRC_MAX_SEQ_LENGTH=200
TGT_MAX_SEQ_LENGTH=200
#transformer model to use
MODEL_NAME='google/mt5-small'
PRETRAINED=1
MT5_CHECKPOINT='None'


FIN_SAVE_DIR='/scratch/lol_check'
mkdir -p $FIN_SAVE_DIR


ONLINE_SYNC=1  #control w&b online syncronization, 0 means inactive

DATASET_DIR=$SCRATCH_DIR/datasets

printf "\n\n"
#dynamically set above default values through shell arguments


while [ $# -gt 0 ]; do
  case "$1" in
    --gpus=*)
      GPUS="${1#*=}"
      ;;
    --checkpoint_path=*)
      CHECKPOINT_PATH="${1#*=}"/checkpoint
      ;;
    --fin_save=*)
      FIN_SAVE_DIR="${1#*=}"/checkpoint
      ;;
    --model_dir=*)
      MODEL_DIR="${1#*=}"
      ;;
    --python=*)
      PYTHON="${1#*=}"
      ;;
    --pre_python=*)
      PRE_PYTHON="${1#*=}"
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      TEST_BATCH_SIZE=$BATCH_SIZE
      ;;
    --mt5_batch_size=*)
      MT5_BATCH_SIZE="${1#*=}"
      TEST_MT5_BATCH_SIZE=$BATCH_SIZE
      ;;
    --test_batch_size=*)
      TEST_BATCH_SIZE="${1#*=}"
      ;;
    --pre_epochs=*)
      PRE_EPOCHS="${1#*=}"
      ;;
    --epochs=*)
      EPOCHS="${1#*=}"
      ;;
    --lr=*)
      LR="${1#*=}"
      ;;
    --src_max_seq_len=*)
      SRC_MAX_SEQ_LENGTH="${1#*=}"
      ;;
    --tgt_max_seq_len=*)
      TGT_MAX_SEQ_LENGTH="${1#*=}"
      ;;
    --model_name=*)
      MODEL_NAME="${1#*=}"
      ;;
    --pretrained=*)
      PRETRAINED="${1#*=}"
      ;;
    --online=*)
      ONLINE_SYNC="${1#*=}"
      ;;
    --lang=*)
      LANG="${1#*=}"
      ;;
    --dataset_dir=*)
      DATASET_DIR="${1#*=}"
      ;;  
    --mt5_checkpoint=*)
      MT5_CHECKPOINT="${1#*=}"
      ;;
    --isTrial=*)
      IS_TRIAL="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument. please check argument $1 *\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done

# api key weight & Biases (uncomment when using online logger and replace <value> with API)
# export WANDB_API_KEY=<your_key>  


#########################################################
#print argument captures in shell script
echo "<< ----------- Experiment configurations -------------"
echo "GPUS : $GPUS"
echo "CHECKPOINT_PATH : $CHECKPOINT_PATH"
echo "FIN_SAVE_DIR : $FIN_SAVE_DIR"
echo "MODEL_DIR : $MODEL_DIR"
echo "PYTHON : $PYTHON"
echo "BATCH_SIZE : $BATCH_SIZE"
echo "TEST_BATCH_SIZE : $TEST_BATCH_SIZE"
echo "PRE_EPOCHS : $PRE_EPOCHS"
echo "LR : $LR"
echo "SRC_MAX_SEQ_LENGTH : $SRC_MAX_SEQ_LENGTH"
echo "TGT_MAX_SEQ_LENGTH : $TGT_MAX_SEQ_LENGTH"
echo "MODEL_NAME : $MODEL_NAME"
echo "PRETRAINED : $PRETRAINED"
echo "ONLINE_SYNC : $ONLINE_SYNC"
echo "DATASET_DIR : $DATASET_DIR"
echo "LANGUAGE : $LANG"
echo "CHECKPOINT_PATH : $CHECKPOINT_PATH"
echo "EPOCHS : $EPOCHS"
echo "mt5 checkpoint : $MT5_CHECKPOINT"
echo "--------------------------------------------------- >>"
printf "\n"

# execute training
$PRE_PYTHON $MODEL_DIR/generation/prompt_uni/main.py --dataset_path $DATASET_DIR --epochs $PRE_EPOCHS --gpus $GPUS --batch_size $MT5_BATCH_SIZE --eval_batch_size $TEST_MT5_BATCH_SIZE --src_max_seq_len $SRC_MAX_SEQ_LENGTH --tgt_max_seq_len $TGT_MAX_SEQ_LENGTH --checkpoint_path $CHECKPOINT_PATH --learning_rate $LR --model_name $MODEL_NAME --online_mode $ONLINE_SYNC --use_pretrained $PRETRAINED --lang $LANG --verbose --enable_script_unification 1

# inference
#$PYTHON $MODEL_DIR/main.py --dataset_path $DATASET_DIR --epochs $EPOCHS --gpus $GPUS --batch_size $BATCH_SIZE --eval_batch_size $TEST_BATCH_SIZE --src_max_seq_len $SRC_MAX_SEQ_LENGTH --tgt_max_seq_len $TGT_MAX_SEQ_LENGTH --checkpoint_path $CHECKPOINT_PATH --learning_rate $LR --model_name $MODEL_NAME --online_mode $ONLINE_SYNC --use_pretrained $PRETRAINED --lang $LANG --verbose --inference --enable_script_unification 1
#!/bin/bash


LANG='all'
export NUM_NODES=1
export NUM_GPUS_PER_NODE=3
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

echo "copying the checkpoint here cp $CHECKPOINT_PATH/checkpoint/*.pt $CHECKPOINT_PATH/"
cp $CHECKPOINT_PATH/*/*ckpt $CHECKPOINT_PATH/

MODEL_NAME='google/mt5-small'
PRETRAINED=1
PRE_N_EPOCHS=$((PRE_EPOCHS-1))
MT5_CHECKPOINT="${CHECKPOINT_PATH}/epoch=${PRE_N_EPOCHS}.ckpt" 

ONLINE_SYNC=1  #control w&b online syncronization, 0 means inactive


printf "\n\n"
#dynamically set above default values through shell arguments
# api key weight & Biases (uncomment when using online logger and replace <value> with API)
# export WANDB_API_KEY=<your_key>


torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    rl_only_bl/train_ddp.py --langs $LANG --num_epochs $EPOCHS --dataset_dir $DATASET_DIR --save_dir $FIN_SAVE_DIR --max_source_length $SRC_MAX_SEQ_LENGTH --max_target_length $TGT_MAX_SEQ_LENGTH --is_mt5 1  --model_gpus 0,1 --train_batch_size $BATCH_SIZE --val_batch_size $TEST_BATCH_SIZE --test_batch_size $TEST_BATCH_SIZE --exp_name multisent_mt5_rl --world_size 3 --mt5_checkpoint $MT5_CHECKPOINT --isTrial $IS_TRIAL



