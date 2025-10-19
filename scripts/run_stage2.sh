which_python=$(which python)
export PYTHONPATH=${PYTHONPATH}:${which_python}:.
echo "PYTHONPATH: ${PYTHONPATH}"

export MASTER_PORT=$((54000 + $RANDOM % 10000))
# export MASTER_ADDR=localhost
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" # 

# global lr = batch size × base learning rate × number of GPUs = 0.00064
epoch=3 # this is set to 3 but we actually train 2 epochs
batch_size=8
lr=1e-5
input_mask3d_dim=128
input_dim=1024
img_input_dim=1536
max_obj_num=150
lora_r=16
lora_alpha=16
config=""
max_grad_norm=0.01
seed=42
max_txt_len=64

# new ideas
add_box_emb=False
add_pos_emb=False
add_scene_token=False
add_sid_eid=False
add_objlabel=False
add_layer_norm=False
add_mask_token=False
add_mask3d_token=True

# adversarial, v2 is corresponding diversified data
# Hybrid Object Probing Evaluation (HOPE): partial_od_scale
# Hybrid Referring Object Classification (HROC): partial_objalign_scale
# Partial Factual 3D Visual Grounding (PF3DVG): partialref_scale, partialrefv2
# Faithful 3D Question Answering (3DFQA): groundedqa, groundedqav2

# diverse, v2 is corresponding diversified data
# captioning: nr3d_caption, scannet_caption, scannet_region_caption, nr3d_captionv2
# grounding: nr3d, sr3d+, scanreferv2, multi3drefv2, nr3dv2, sr3d+v2
# question answering: obj_align, scanqav2, sqa3dv2

# first stage: RIG data
# train_tag="partialref_scale#groundedqa#partial_objalign_scale#partial_od_scale#partialrefv2#groundedqav2#obj_align#nr3d_caption#scannet_caption#scannet_region_caption#nr3d#sr3d+#scanreferv2#multi3drefv2#nr3dv2#sr3d+v2#nr3d_captionv2#scanqav2#sqa3dv2"

# second stage: benchmark data
train_tag="scanrefer#multi3dref#scan2cap#scanqa#sqa3d"
val_tag="scanrefer#multi3dref#scan2cap#scanqa#sqa3d"

do_eval_during_train=False
resume=False
evaluate=False
debug=False
if [ $debug = "True" ]; then
    enable_wandb=False
    gpu_num=1
    do_save=False
    other_info="debug"
else
    enable_wandb=False
    gpu_num=8
    do_save=True
    other_info=""
fi

# default: 4gpu-cuda18-bs32-maxgrad0.01-3Esave2E-Lora16w16-lr5
# do not put / at the end of OUTPUT_DIR, bc of wandb
OUTPUT_DIR=/data/kangweitai/3D/project/chat3d/rift-pretrain-robust-crcv-4g16bs-bj4g16bs-v2
pretrained_path=/data/kangweitai/3D/project/chat3d/rift-pretrain-robust-crcv-4g16bs/ckpt_01.pth # for evaluation


mkdir -p ${OUTPUT_DIR}

torchrun --nnodes=1 --nproc_per_node=$gpu_num --master_port=$MASTER_PORT \
    tasks/train.py \
    "$(dirname $0)/${config}config.py" \
    output_dir "$OUTPUT_DIR" \
    scheduler.epochs "$epoch" \
    optimizer.lr "$lr" \
    model.add_scene_token "$add_scene_token" \
    pretrained_path "$pretrained_path" \
    evaluate "$evaluate" \
    wandb.enable "$enable_wandb" \
    gpu_num "$gpu_num" \
    do_save "$do_save" \
    batch_size "$batch_size" \
    train_tag "$train_tag" \
    val_tag "$val_tag" \
    segmentor "$segmentor" \
    pc_encoder "$pc_encoder" \
    model.input_dim "$input_dim" \
    model.max_obj_num "$max_obj_num" \
    lora.lora_r "$lora_r" \
    lora.lora_alpha "$lora_alpha" \
    model.add_pos_emb "$add_pos_emb" \
    optimizer.max_grad_norm "$max_grad_norm" \
    seed "$seed" \
    do_eval_during_train "$do_eval_during_train" \
    resume "$resume" \
    model.add_box_emb "$add_box_emb" \
    model.add_sid_eid "$add_sid_eid" \
    model.add_objlabel "$add_objlabel" \
    model.add_layer_norm "$add_layer_norm" \
    model.add_mask_token "$add_mask_token" \
    model.input_mask3d_dim "$input_mask3d_dim" \
    model.img_input_dim "$img_input_dim" \
    model.add_mask3d_token "$add_mask3d_token" \
    model.max_txt_len "$max_txt_len" \
