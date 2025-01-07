import os
import logging
import shutil
import model.FocusMae as model_focus_mae 
import model.PatchTST as model_patchtst

import model.FocusMergeMae as model_focusmerge_mae

import model.PatchTSMixer as model_patchtsmixer

from dataset import PretrainDataset
from torch.utils.data import DataLoader
from datetime import datetime
from util.schedule import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
import pytz
import torch.optim as optim
import model.st_mem.st_mem as model_st_mem
def init_logger(model_name, dataset_name, exp_id):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 创建文件处理程序
    root = f'./ckpt/pre_train/{dataset_name}/{model_name}/{exp_id}'
    os.makedirs(root, True)
    path = os.path.join(root, f"{exp_id}_exp.log")
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 添加文件处理程序到logger
    logger.addHandler(file_handler)
    return logger

def run_pretrain(args):
    # logger and ckpt ready
    exp_id = f'{datetime.now().astimezone(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d%H%M")}'
    logger = init_logger(args.model_name, args.dataset_name, exp_id)
    ckpt_dir = 'ckpt/pre_train/{}/{}/{}'.format(args.dataset_name, args.model_name, exp_id)
    shutil.copy('pretrain.py', ckpt_dir)
    shutil.copy('.vscode/launch.json', ckpt_dir)
    shutil.copy(f'script/pretrain/{args.dataset_name}/run_{args.model_name.lower()}_pretrain.sh', ckpt_dir)

    # make model
    if args.model_name == "FocusMae":
        model = model_focus_mae.mae_prefer_custom(args)
        shutil.copy('model/FocusMae.py', ckpt_dir)
    elif args.model_name == "PatchTST":
        model = model_patchtst.patchtst_prefer_custom(args)
        shutil.copy('model/PatchTST.py', ckpt_dir)
    elif args.model_name == "FocusMergeMae":
        model = model_focusmerge_mae.mae_prefer_custom(args)
        shutil.copy('model/FocusMergeMae.py', ckpt_dir)
    elif args.model_name == "PatchTSMixer":
        model = model_patchtsmixer.patchtsmixer_prefer_custom(args)
        shutil.copy('model/PatchTSMixer.py', ckpt_dir)
    elif args.model_name == "ST-MEM":
        model = model_st_mem.st_mem_prefer_custom(args)
        shutil.copytree('model/st_mem', os.path.join(ckpt_dir, 'st_mem'))
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")
        
    model = model.to(args.device)
    # make data
    train_dataset = PretrainDataset(args.train_data_path)
    val_dataset = PretrainDataset(args.val_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=False)
    
    early_stopping = EarlyStopping(patience=args.early_stop_patience, save_dir=ckpt_dir)
    
    # ready train
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.scheduler_patience,  # 增加patience
        factor=args.scheduler_factor,   # 更温和的衰减
        min_lr=args.scheduler_min_lr,
        verbose=True  # 打印学习率变化
    )

    step = 0
    all_loss = 0
    # 添加epoch属性到模型
    model.current_epoch = 0
    
    for epoch in range(args.max_epoch_num):
        model.current_epoch = epoch  # 更新当前epoch
        
        logger.info(f'epoch={epoch}')
    
        # 训练阶段
        train_losses = []
        val_losses = []
        # train
        prog_iter = tqdm(train_dataloader, desc="training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):
            model.train()
            batch = batch.to(args.device)
            optimizer.zero_grad()
            if args.model_name == "ST-MEM":
                loss= model.forward(batch)['loss']
            else:
                loss= model.forward_loss(batch)
            loss.backward()
            optimizer.step()
            all_loss += loss    
            step += 1
            if step % args.val_every_n_steps == 0:
                logger.info(f'batch_iter={step // args.val_every_n_steps} train_loss={all_loss}')
                all_loss = 0
                # validate
                model.eval()
                prog_iter = tqdm(val_dataloader, desc="validating", leave=False)
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter):

                        if args.model_name == 'simclr':
                            batch = torch.cat(batch, dim=0)
                            batch = batch.to(args.device)
                            logits, labels, loss = model(batch)
                            top1, top5 = model.accuracy(logits, labels, topk=(1, 5))
                            avg_top1 += top1[0]
                            avg_top5 += top5[0]
                        else:
                            if args.model_name == 'ContraMaeTest':
                                batch = torch.cat(batch, dim=0)
                            batch = batch.to(args.device)
                            loss= model.forward_loss(batch)
                        batch = batch.to(args.device)
                        loss= model.forward_loss(batch)
                        if args.model_name == "ST-MEM":
                            loss= model.forward(batch)['loss']
                        else:
                            loss= model.forward_loss(batch)
                        all_loss += loss
                logger.info(f'batch_iter={step // args.val_every_n_steps} val_loss={all_loss}')
                scheduler.step(all_loss)
                early_stopping.check(all_loss, model)
                if early_stopping.early_stop:
                    return
                all_loss = 0

# if __name__ == '__main__':
#     args = parse_args()
#     print(args)
#     pre_train(args)
