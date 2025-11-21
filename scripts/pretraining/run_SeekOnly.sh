#!/usr/bin/env bash

PROJ_ROOT=/home/jhair/Research/DOING/dream_replay_bind_dev

export PYTHONPATH=${PROJ_ROOT}
source activate py311gpu
cd ${PROJ_ROOT}

# Dataset parameters
DATA_PATH=/data/datasets/caltech256/256_ObjectCategories_splits
NUM_CLASSES=256
CHANNELS=3 # 4
### View encoder parameters
lr_venc=7.3e-4 #0.0008
wd_venc=0.05 #0.05
venc_drop_path=0.0125
### Action encoder parameters
lr_act_enc=7.3e-4 #0.0008
wd_act_enc=0 #0.01
act_enc_dim=64
act_enc_n_layers=2
act_enc_n_heads=4
### Seek parameters
lr_seek=7.3e-4 #0.0008
wd_seek=0
seek_dim=192
seek_n_layers=8
seek_n_heads=8
seek_dropout=0
### Bind parameters
lr_bind=7.3e-4 #0.0008
wd_bind=0.001
bind_dim=128
bind_n_layers=2
bind_n_heads=4
bind_dropout=0
### Generator parameters
lr_gen=7.3e-4 #0.0008
wd_gen=0
### Classifier parameters
lr_classifier=0.01
wd_classifier=0
### Training parameters
epochs=100
warmup_epochs=5
episode_batch_size=88
num_views=4
coeff_mse=1.0
coeff_bce=0.0 # 0 means SeekOnly
workers=32
save_dir=output/Pretraining_caltech256_SeekBind
print_frequency=10
seed=0
zoom_min=0.08
zoom_max=0.5

RUN_NAME=SeekOnly_mse${coeff_mse}_bce${coeff_bce}_ch${CHANNELS}_cosineLR_nohflip_cropzoom${zoom_min}_${zoom_max}_firstcenteranysize_bs${episode_batch_size}

LOGS_FOLDER=${save_dir}/${RUN_NAME}

### if folder for logs doens't exist, create one so I can save log there
if [ ! -d ${LOGS_FOLDER} ]; then
    mkdir -p ${LOGS_FOLDER}
fi

### Run file
python -u main_pretraining.py \
    --data_path  ${DATA_PATH} \
    --num_classes ${NUM_CLASSES} \
    --channels ${CHANNELS} \
    --lr_venc ${lr_venc} \
    --wd_venc ${wd_venc} \
    --venc_drop_path ${venc_drop_path} \
    --lr_act_enc ${lr_act_enc} \
    --wd_act_enc ${wd_act_enc} \
    --act_enc_dim ${act_enc_dim} \
    --act_enc_n_layers ${act_enc_n_layers} \
    --act_enc_n_heads ${act_enc_n_heads} \
    --lr_seek ${lr_seek} \
    --wd_seek ${wd_seek} \
    --seek_dim ${seek_dim} \
    --seek_n_layers ${seek_n_layers} \
    --seek_n_heads ${seek_n_heads} \
    --seek_dropout ${seek_dropout} \
    --seek_gain_fields \
    --lr_bind ${lr_bind} \
    --wd_bind ${wd_bind} \
    --bind_dim ${bind_dim} \
    --bind_n_layers ${bind_n_layers} \
    --bind_n_heads ${bind_n_heads} \
    --bind_dropout ${bind_dropout} \
    --lr_gen ${lr_gen} \
    --wd_gen ${wd_gen} \
    --lr_classifier ${lr_classifier} \
    --wd_classifier ${wd_classifier} \
    --epochs ${epochs} \
    --warmup_epochs ${warmup_epochs} \
    --episode_batch_size ${episode_batch_size} \
    --num_views ${num_views} \
    --coeff_mse ${coeff_mse} \
    --coeff_bce ${coeff_bce} \
    --workers ${workers} \
    --print_frequency ${print_frequency} \
    --seed ${seed} \
    --zoom_min ${zoom_min} \
    --zoom_max ${zoom_max} \
    --save_dir ${LOGS_FOLDER} > ${LOGS_FOLDER}/log.log