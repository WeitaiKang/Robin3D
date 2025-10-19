# ========================= data ==========================
scene_val_attr_file = '/data/kangweitai/3D/chat3d-anno/scannet/scannet_val_attributes.pt' # gt
scene_train_attr_file = '/data/kangweitai/3D/chat3d-anno/scannet/scannet_train_attributes.pt' # gt

file_dir ="mask3d150sort-uni3d-dinov2Giant-iou50-oneprompt"
anno_root = f"/data/kangweitai/3D/chat3d-anno/{file_dir}/"

pred_root = "/data/kangweitai/3D/mask3d_uni3d_dinov2/mask3d150sort_uni3d_dinov2G/" # feature dir 
pc_encoder = "uni3d"
segmentor = "mask3d"
version = ""
seg_train_attr_file = f"{pred_root}/scannet_{segmentor}_train_attributes{version}.pt"

# train, val
seg_mask3d_feat_pos_file = f"{pred_root}/scannet_{segmentor}_{segmentor}_feat_pos.pt"
seg_mask3d_feat_file = f"{pred_root}/scannet_{segmentor}_{segmentor}_feats.pt"
seg_feat_file = f"{pred_root}/scannet_{segmentor}_{pc_encoder}_feats.pt"
seg_img_feat_file = f"{pred_root}/scannet_{segmentor}_videofeats{version}.pt"
seg_val_attr_file = f"{pred_root}/scannet_{segmentor}_val_attributes{version}.pt"

# test
# seg_mask3d_feat_pos_file = f"{pred_root}/scannet_{segmentor}_{segmentor}_feat_pos_test.pt"
# seg_mask3d_feat_file = f"{pred_root}/scannet_{segmentor}_{segmentor}_feats_test.pt"
# seg_feat_file = f"{pred_root}/scannet_{segmentor}_{pc_encoder}_feats_test.pt"
# seg_img_feat_file = f"{pred_root}/scannet_{segmentor}_videofeats{version}_test.pt"
# seg_val_attr_file = f"{pred_root}/scannet_{segmentor}_test_attributes{version}.pt"

train_tag = 'scanqa'
val_tag = 'scanqa'

train_file_dict = {
    'scanrefer': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'scan2cap': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'nr3d_caption': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/nr3d_caption_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'obj_align': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/obj_align_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'multi3dref': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'scanqa': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanqa_train.json",
        scene_train_attr_file
    ],
    'sqa3d': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sqa3d_train.json",
        scene_train_attr_file,
    ],
    'scannet_caption': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scannet_caption_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'scannet_region_caption': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scannet_region_caption_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'partialref': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/partialref_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'groundedqa': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/groundedqa_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'pointedcap': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/pointedcap_{segmentor}_train{version}.json",
        scene_train_attr_file,
    ],
    'partial_objalign': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/partial_obj_align_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'partial_od': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/partial_od_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'nr3d': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/nr3d_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'sr3d+': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sr3d+_{segmentor}_train{version}.json",
        scene_train_attr_file
    ],
    'partialrefv2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/partialref_train_rephrase_merged.json",
        scene_train_attr_file
    ],
    'pointedcapv2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/pointedcap_mask3d_train_rephrase.json",
        scene_train_attr_file
    ],
    'groundedqav2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/groundedqa_{segmentor}_train{version}_rephrase.json",
        scene_train_attr_file
    ],
    'scanreferv2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_train{version}_rephrase.json",
        scene_train_attr_file
    ],
    'multi3drefv2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_train{version}_rephrase.json",
        scene_train_attr_file
    ],
    'nr3dv2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/nr3d_{segmentor}_train{version}_rephrase.json",
        scene_train_attr_file
    ],
    'sr3d+v2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sr3d+_{segmentor}_train{version}_rephrase.json",
        scene_train_attr_file
    ],
    'nr3d_captionv2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/nr3d_caption_{segmentor}_train{version}_rephrase.json",
        scene_train_attr_file
    ],
    'scanqav2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/scanqa_train_rephrase.json",
        scene_train_attr_file
    ],
    'sqa3dv2': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/sqa3d_train_rephrase.json",
        scene_train_attr_file,
    ],
    'pointedcapv3': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/pointedcap_mask3d_train2.json",
        scene_train_attr_file
    ],
    'partialref_scale': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/partialref_{segmentor}_train{version}_scale.json",
        scene_train_attr_file
    ],
    'partial_objalign_scale': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/partial_obj_align_{segmentor}_train{version}_scale.json",
        scene_train_attr_file
    ],
    'partial_od_scale': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_train_attr_file,
        f"{anno_root}/partial_od_{segmentor}_train{version}_scale.json",
        scene_train_attr_file
    ],
}

val_file_dict = {
    'scanqa': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanqa_val.json",
        scene_val_attr_file
    ],
    'scanrefer': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scanrefer_{segmentor}_val{version}.json",
        scene_val_attr_file
    ],
    'scan2cap': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/scan2cap_{segmentor}_val{version}.json",
        scene_val_attr_file
    ],
    'sqa3d': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/sqa3d_val.json",
        scene_val_attr_file
    ],
    'sqa3d_test': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/sqa3d_test.json",
        scene_val_attr_file
    ],
    'multi3dref': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/multi3dref_{segmentor}_val{version}.json",
        scene_val_attr_file
    ],
    'nr3d': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/nr3d_{segmentor}_val{version}.json",
        scene_val_attr_file
    ],
    'sr3d': [
        seg_mask3d_feat_pos_file,
        seg_mask3d_feat_file,
        seg_feat_file,
        seg_img_feat_file,
        seg_val_attr_file,
        f"{anno_root}/sr3d_{segmentor}_val{version}.json",
        scene_val_attr_file
    ],
}


num_workers = 8
batch_size = 32


# ========================= model ==========================
model = dict(
    llama_model_path="/data/kangweitai/LLM/vicuna-7b-v1.5/", 
    input_mask3d_dim=128,
    input_dim=1024,
    img_input_dim=1024,
    attr_dim=512,
    scene_dim=256,
    pos_dim=128,
    encoder_num_layers=3,
    low_resource=False,
    system_path="prompts/system.txt",
    instruction_path="prompts/instruction.txt",
    max_txt_len=128,
    end_sym="</s>",
    role=("USER", "ASSISTANT"),
    add_scene_token=False,
    add_img_token=True,
    use_lora=True,
    train_emb=True,
    train_img_proj=True,
    no_obj=False,
    max_obj_num=100,
    bidirection=False,
    add_pos_emb=False,
    feat_fusion=False,
    add_box_emb=False,
    add_sid_eid=False,
    add_objlabel=False,
    add_layer_norm=False,
    add_mask_token=False,
    add_mask3d_token=False,
)

lora = dict(
    lora_target_modules=[
      "q_proj",
      "v_proj",
      "k_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ],
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.05
)

optimizer = dict(
    opt="adamW",
    lr=5e-3,
    opt_betas=[0.9, 0.999],  # default
    weight_decay=0.02,
    scaler_enable=False,
    max_grad_norm=-1,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(
        enable=False,
        module_names=["model.embed_tokens"],
        lr=[5e-4],
        wd=[0.02]
    ),
)

scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="tmp",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="geep",
)
dist_url = "env://"
device = "cuda"

# ========================= others ==========================
output_dir = "outputs/tmp"  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
# eval_freq = 500
seed = 42

save_latest = True # False
do_save = True
auto_resume = False # True
pretrained_path = ""
img_projector_path = ""

debug=False
gpu_num=1
do_eval_during_train=True