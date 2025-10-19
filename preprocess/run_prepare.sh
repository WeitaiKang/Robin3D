#! /bin/bash

scannet_dir="/data_ext1/kangweitai/3D/scannet/"
version=""
#输入目录
segment_result_dir="/data/kangweitai/3D/mask3d_uni3d_dinov2/Mask3D/eval_output/instance_evaluation_weitai_0/"
inst_seg_dir=""
#输出目录
processed_data_dir="/data/kangweitai/3D/mask3d_uni3d_dinov2/Mask3D/eval_output/"
class_label_file="/data/kangweitai/3D/scannet/meta_data/scannetv2-labels.combined.tsv"
save_feat_dir="/data/kangweitai/3D/mask3d_uni3d_dinov2/mask3d150sort_uni3d_dinov2G/"

segmentor="mask3d"

train_iou_thres=0.50
save_dir="/data/kangweitai/3D/chat3d-anno/mask3d150sort-uni3d-dinov2Giant-iou50-oneprompt/"


# python preprocess/prepare_scannet_attributes.py \
#     --scannet_dir "$scannet_dir" \

python preprocess/prepare_mask3d_data.py \
    --scannet_dir "$scannet_dir" \
    --output_dir "$processed_data_dir" \
    --segment_dir "$segment_result_dir" \
    --inst_seg_dir "$inst_seg_dir" \
    --class_label_file "$class_label_file" \
    --apply_global_alignment \
    --num_workers 16 \
    --parallel \
    --save_feat_dir "$save_feat_dir"

python preprocess/prepare_scannet_mask3d_attributes.py \
    --scan_dir "$processed_data_dir" \
    --segmentor "$segmentor" \
    --save_dir "$save_feat_dir"

# python preprocess/prepare_scanrefer_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_scan2cap_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_objalign_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_nr3dcaption_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_multi3dref_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_scanqa_annos.py \
#     --save_dir "$save_dir"

# python preprocess/prepare_sqa3d_annos.py \
#     --save_dir "$save_dir"

# python preprocess/prepare_scannet_caption_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_scannet_region_caption_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_groundedqa_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_pointedcap_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_partialref_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_partialobjalign_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_partialod_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_nr3d_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_sr3d_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"
    
# python preprocess/prepare_taskqa_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_scanrefer_rephrase_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_multi3dref_rephrase_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_nr3d_rephrase_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_nr3dcaption_rephrase_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_pointedcap_rephrase_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"

# python preprocess/prepare_sr3d_rephrase_annos.py \
#     --segmentor "$segmentor" \
#     --version "$version" \
#     --train_iou_thres "$train_iou_thres" \
#     --save_dir "$save_dir" \
#     --attr_dir "$save_feat_dir"