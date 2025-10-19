import datetime
import logging
import time
from os.path import join

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import ConcatDataset
import torch.multiprocessing
import sys
sys.path.append('.')
from dataset import MetaLoader, create_dataset, create_loader, create_sampler
from dataset.dataset_train import train_collate_fn
from dataset.dataset_val import val_collate_fn
from models.chat3d import Chat3D
from tasks.shared_utils import get_media_types, setup_model
from utils.basic_utils import (MetricLogger, SmoothedValue, setup_seed)
from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb
from utils.eval import calc_scanrefer_score, clean_answer, \
    calc_scan2cap_score, calc_scanqa_score, calc_sqa3d_score, calc_multi3dref_score, \
    calc_partialref_score, calc_pointedcap_score, calc_taskqa_score, calc_groundedqa_score, \
    calc_nr3d_score, calc_sr3d_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
# from utils.capeval.cider.cider import Cider


import numpy as np
from tqdm import tqdm

import json
import os

logger = logging.getLogger(__name__)
max_bleus = [0.] * 4

tokenizer = PTBTokenizer()
scorers = [
    (Bleu(), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), 
    # (Meteor(), "METEOR"),
    # (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
    # (Spice(), "SPICE")
]
scorers_scanqa = [
    (Bleu(), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), 
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
]

max_global_step = 200000000


def train(
    model,
    model_without_ddp,
    train_loaders,
    val_loaders,
    optimizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
    do_eval=True
):
    model.train()
    model_without_ddp.llama_model.config.use_cache = False

    metric_logger = MetricLogger(delimiter="  ")
    eval_metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    loss_names = ["loss", "obj_norm", "obj_img_norm", "objid_norm"]
    media_types = get_media_types(train_loaders)

    # tot_param = sum(p.numel() for p in model_without_ddp.parameters())
    # trainable_param = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    # print(f"Total Params: {tot_param / 1e6}M")
    # print(f"Trainable Params: {trainable_param / 1e6}M")
    # exit()

    for name in loss_names:
        metric_logger.add_meter(
            f"{name}", SmoothedValue(window=1, fmt="{value:.6f}")
        )

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    accum_iter = 1
    optimizer.zero_grad()
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (media_type, batch) in enumerate(iterator):
        for k in batch.keys():
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].to(device)
        loss_dict = model(batch)
        loss = loss_dict["loss"] / accum_iter
        
        scaler.scale(loss).backward()

        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            if config.optimizer.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
        scheduler.step()

        # logging
        for name in loss_names:
            if name not in loss_dict:
                continue
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{name}": value})
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1
        if global_step > max_global_step:
            return global_step

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")

    if is_main_process() and config.do_save and not config.debug:
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_without_ddp.named_parameters()
            }
        state_dict = model_without_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
        }
        torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))
    
    if do_eval:
        val_metrics = evaluate_all(model, model_without_ddp, val_loaders, epoch, global_step, device, config)
        if is_main_process():
            for k, v in val_metrics.items():
                if k not in eval_metric_logger.meters:
                    eval_metric_logger.add_meter(k, SmoothedValue(window=1, fmt="{value:.4f}"))
            eval_metric_logger.update(**val_metrics)
    
    return global_step


def evaluate_all(
    model,
    model_without_ddp,
    val_loaders,
    epoch,
    global_step,
    device,
    config
):
    logger.info("Start evaluating...")
    model.eval()
    model_without_ddp.llama_model.config.use_cache = True
    val_scores = {}
    for val_loader in val_loaders:
        new_val_scores = evaluate(model, val_loader, epoch, global_step, device, config)
        val_scores = {**val_scores, **new_val_scores}
        if is_main_process() and config.wandb.enable:
            log_dict_to_wandb(new_val_scores, step=global_step, prefix="val/")
    
    logger.info(f"[epoch={epoch}, global steps={global_step}] Val Results:")
    for k, v in val_scores.items():
        logger.info(f"{k}: {v}")
    
    model.train()
    if config.distributed:
        model.module.llama_model.config.use_cache = False
    return val_scores


def evaluate(
    model,
    val_loader,
    epoch,
    global_step,
    device,
    config
):
    eval_name = val_loader.dataset.datasets[0].dataset_name
    anno_len = len(val_loader.dataset.datasets[0].anno)
    if is_main_process():
        print(f'-- eval {eval_name} annotation len: {anno_len} --')
    logger.info(f"Evaluating {eval_name}...")
    if config.distributed:
        val_loader.sampler.set_epoch(epoch)

    save_preds = []
    logger.info(f"{eval_name}: batch-size={val_loader.batch_size} length(val_loader)={len(val_loader)}")
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        for k in batch.keys():
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].to(device)
        with torch.no_grad():
            pred = model(batch)
            batch_size = len(pred)
            for bi in range(batch_size):
                index = batch["indexes"][bi]
                scene_id = batch["scene_id"][bi]
                obj_id = batch["obj_ids"][bi]
                qid = batch["qid"][bi]
                pred_id = int(batch['pred_ids'][bi])
                type_info = batch['type_infos'][bi]
                tmp_pred = pred[bi]
                save_preds.append({
                    "scene_id": scene_id,
                    "gt_id": obj_id,
                    'pred_id': pred_id,
                    "qid": qid,
                    "pred": tmp_pred,
                    "ref_captions": batch["ref_captions"][bi],
                    "type_info": type_info,
                    "eval_name_index": f"{eval_name}_{index}"
                })

    if len(save_preds) > 0:
        save_preds = sorted(save_preds, key=lambda x: f"{x['scene_id']}_{x['qid']}")
        with open(os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_rank{get_rank()}_{eval_name}.json"),
                  "w") as f:
            json.dump(save_preds, f, indent=4)

    if config.distributed:
        dist.barrier()
    if is_main_process():
        save_preds = dict()
        for rank in range(config.gpu_num):
            path = os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_rank{rank}_{eval_name}.json")
            if os.path.exists(path):
                preds = json.load(open(path, "r"))
                for pred in preds:
                    save_preds[pred["eval_name_index"]] = pred
                os.remove(path)
        save_preds = list(save_preds.values())
        save_preds = sorted(save_preds, key=lambda x: f"{x['scene_id']}_{x['qid']}")
        with open(os.path.join(config.output_dir, f"preds_epoch{epoch}_step{global_step}_{eval_name}.json"), "w") as f:
            json.dump(save_preds, f, indent=4)

        assert len(save_preds) == anno_len, f"len(save_preds)={len(save_preds)} != anno_len={anno_len}"
        print(f'--- {eval_name} eval quantity: {len(save_preds)} ---')

    val_scores = {}
    if is_main_process() and len(save_preds) > 0:
        if eval_name == 'scanqa':
            val_scores = calc_scanqa_score(save_preds, tokenizer, scorers_scanqa, config)
        elif eval_name == 'scanrefer':
            val_scores = calc_scanrefer_score(save_preds, config)
        elif eval_name == "scan2cap":
            val_scores = calc_scan2cap_score(save_preds, tokenizer, scorers, config)
        elif eval_name == "sqa3d" or eval_name == "sqa3d_test":
            val_scores = calc_sqa3d_score(save_preds, tokenizer, scorers, config, eval_name)
        elif eval_name == 'multi3dref':
            val_scores = calc_multi3dref_score(save_preds, config)
        elif eval_name == 'partialref':
            val_scores = calc_partialref_score(save_preds, tokenizer, scorers, config)
        elif eval_name == 'pointedcap':
            val_scores = calc_pointedcap_score(save_preds, tokenizer, scorers, config)
        elif eval_name == 'taskqa':
            val_scores = calc_taskqa_score(save_preds, tokenizer, scorers, config)
        elif eval_name == 'groundedqa':
            val_scores = calc_groundedqa_score(save_preds, tokenizer, scorers, config)
        elif eval_name == 'nr3d':
            val_scores = calc_nr3d_score(save_preds, config)
        elif eval_name == 'sr3d':
            val_scores = calc_sr3d_score(save_preds, config)
        else:
            raise NotImplementedError
        print(json.dumps(val_scores, indent=4))

    return val_scores


def setup_dataloaders(config):
    # train datasets, create a list of data loaders
    train_datasets, val_datasets = create_dataset(config)

    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_samplers = create_sampler(
            train_datasets, [True] * len(train_datasets), num_tasks, global_rank
        )
        val_samplers = create_sampler(
            val_datasets, [False] * len(val_datasets), num_tasks, global_rank
        )
    else:
        train_samplers = [None] * len(train_datasets)
        val_samplers = [None] * len(val_datasets)

    train_loaders = create_loader(
        train_datasets,
        train_samplers,
        batch_size=[config.batch_size] * len(val_datasets),
        num_workers=[config.num_workers] * len(train_datasets),
        is_trains=[True] * len(train_datasets),
        collate_fns=[train_collate_fn] * len(train_datasets),
    )
    val_loaders = create_loader(
        val_datasets,
        val_samplers,
        batch_size=[config.batch_size] * len(val_datasets),
        num_workers=[config.num_workers] * len(val_datasets),
        is_trains=[False] * len(val_datasets),
        collate_fns=[val_collate_fn] * len(val_datasets),
    )

    return train_loaders, val_loaders


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    # torch.autograd.set_detect_anomaly(True)
    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, val_loaders = setup_dataloaders(config)

    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs
    torch.backends.cudnn.benchmark = True

    model_cls = eval(config.model.get('model_cls', 'Chat3D'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        find_unused_parameters=False,
    )
    # if is_main_process() and config.wandb.enable:
    #     wandb.watch(model)
    
    start_time = time.time()
    if not config.evaluate:
        logger.info("Start training")
        for epoch in range(start_epoch, config.scheduler.epochs):
            # if epoch == config.scheduler.epochs - 1:
            #     break
            global_step = train(
                model,
                model_without_ddp,
                train_loaders,
                val_loaders,
                optimizer,
                epoch,
                global_step,
                device,
                scheduler,
                scaler,
                config,
                do_eval=config.do_eval_during_train
            )
            if global_step > max_global_step:
                break
            dist.barrier()

    if config.evaluate:
        evaluate_all(model, model_without_ddp, val_loaders, start_epoch - 1, global_step, device, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()


if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
