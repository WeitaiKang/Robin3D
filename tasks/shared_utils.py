import copy
import logging
import os
import os.path as osp
from os.path import join

import torch
from torch.utils.data import ConcatDataset, DataLoader

from utils.optimizer import create_optimizer
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def get_media_types(datasources):
    """get the media types for for all the dataloaders.

    Args:
        datasources (List): List of dataloaders or datasets.

    Returns: List. The media_types.

    """
    if isinstance(datasources[0], DataLoader):
        datasets = [dataloader.dataset for dataloader in datasources]
    else:
        datasets = datasources
    media_types = [
        dataset.datasets[0].media_type
        if isinstance(dataset, ConcatDataset)
        else dataset.media_type
        for dataset in datasets
    ]

    return media_types


def setup_model(
    config, model_cls, find_unused_parameters=False
):
    logger.info("Creating model")
    config = copy.deepcopy(config)

    model = model_cls(config=config)

    model = model.to(torch.device(config.device))
    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.gpu],
            find_unused_parameters=find_unused_parameters,  # `False` for image-only task
            gradient_as_bucket_view=True
        )
    optimizer = create_optimizer(config.optimizer, model, config)
    scheduler = create_scheduler(config.scheduler, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.optimizer.scaler_enable, growth_interval=100)

    start_epoch = 0
    global_step = 0

    if osp.isfile(config.pretrained_path):
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        state_dict = checkpoint["model"]

        # state_dict = {k: v for k, v in state_dict.items() 
        #               if ('llama_model' in k)} # and 'OBJ_lm_head' not in k
        # print(f'---- siri!')

        if config.resume:
            print(f'--- resume training from {config.pretrained_path}')
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
        keys_to_delete = []
        for name, param  in state_dict.items():
            if name not in model_without_ddp.state_dict():
                continue
            if param.size() != model_without_ddp.state_dict()[name].size():
                keys_to_delete.append(name)
        for key in keys_to_delete:
            del state_dict[key]
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"Loaded checkpoint from {config.pretrained_path}.")
    else:
        logger.warning("No pretrained checkpoint provided, training from scratch.")

    return (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        global_step,
    )
