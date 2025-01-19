import os
import torch
import pytz
import shutil
import torch.optim as optim
import model.FocusMae as model_focus_mae
import model.FocusMergeMae as model_focusmerge_mae
import model.PatchTST as model_patchtst
import model.PatchTSMixer as model_patchtsmixer
import model.Mae as model_mae
import model.RMae as model_rmae
from tqdm import tqdm
from dataset import PhysionetDataset
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report 
from model.classifier import MlpHeadV1, Classifier
from util.schedule import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import model.st_mem.encoder.st_mem_vit as model_st_mem_vit
from sklearn.metrics import accuracy_score
<<<<<<< HEAD


=======
>>>>>>> 2d66fc41ececdbd34e3e1cf83f167359a6c77daf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)
torch.cuda.manual_seed(41)
print(device)


def infer(model, data_loader, task ,device):
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    prog_iter = tqdm(data_loader, task, leave=False)
    y_true_list = []
    y_pred_list = []
    y_pred_prob_list = []
    all_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)
            input_y = input_y.reshape(-1)
            pred = model(input_x)
            y_true_list.extend(input_y.tolist())
            y_pred_list.extend(torch.max(pred, dim=1)[1].tolist())
            y_pred_prob_list.extend(torch.softmax(pred, dim=1).tolist())
            loss = loss_func(pred, input_y)
            all_loss += loss

    acc = accuracy_score(y_true_list, y_pred_list)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_list, y_pred_list, average='macro')
    
    acc=accuracy_score(y_true_list, y_pred_list)
    auroc = roc_auc_score(y_true_list, y_pred_prob_list, multi_class='ovr')

    print(classification_report(y_true_list, y_pred_list))
    print(f'accuracy = {acc}, macro_p = {macro_p}, macro_r = {macro_r}, macro_f1 = {macro_f1}, AUROC = {auroc}')      
<<<<<<< HEAD
    return all_loss, micro_f1, macro_p, macro_r, macro_f1, auroc
=======
    return all_loss, acc, macro_p, macro_r, macro_f1, auroc
>>>>>>> 2d66fc41ececdbd34e3e1cf83f167359a6c77daf


def get_model(args):
    pre_train_model, classifier_head = None, None

    if args.model_name == 'FocusMae':
        pre_train_model = model_focus_mae.mae_prefer_custom(args)
    elif args.model_name == 'Mae':
        pre_train_model = model_mae.mae_prefer_custom(args)
    elif args.model_name == 'RMae':
        pre_train_model = model_rmae.mae_prefer_custom(args)
    elif args.model_name == 'FocusMergeMae':
        pre_train_model = model_focusmerge_mae.mae_prefer_custom(args)
    elif args.model_name == 'PatchTST':
        model = model_patchtst.patchtst_prefer_custom(args)
        if args.task == 'finetune' and args.pretrain_model_freeze:
            for _, p in model.model.model.named_parameters():
                p.requires_grad = False
        model = model.to(args.device)
        return model
    elif args.model_name == 'PatchTSMixer':
        model = model_patchtsmixer.patchtsmixer_prefer_custom(args)
        if args.task == 'finetune' and args.pretrain_model_freeze:
            for _, p in model.model.model.named_parameters():
                p.requires_grad = False
        model = model.to(args.device)
        return model
    elif args.model_name == 'ST-MEM':
        model = model_st_mem_vit.ST_MEM_ViT(
            seq_len=args.signal_length,
            patch_size=args.patch_length,
            num_leads=args.num_input_channels,
            num_classes=args.class_n,
            width=args.embed_dim,
            depth=args.encoder_depth,
            heads=args.encoder_num_heads,
            mlp_dim=args.mlp_ratio * args.embed_dim,
        )
        model = model.to(args.device)

        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        if args.task == 'finetune':
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("encoder."):
                    new_key = k[len("encoder."):]
                    new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        

        if args.task == 'finetune' and args.pretrain_model_freeze:
            for _, p in model.named_parameters():
                p.requires_grad = False
            for _, p in model.head.named_parameters():
                p.requires_grad = True

        return model
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    if args.classifier_head_name == 'mlp_v1':

        classifier_head = MlpHeadV1(pretrain_out_dim=768  , class_n=args.class_n)
        model = Classifier(pre_train_model=pre_train_model, classifier_head=classifier_head)
        
    if args.task == 'finetune' and args.classifier_head_name is not None and not args.multi_stage_finetune:
        if args.ckpt_path != "":
            checkpoint = torch.load(args.ckpt_path, map_location='cpu')
            model.pre_train_model.load_state_dict(checkpoint, strict=True)
        if args.pretrain_model_freeze:
            for name, p in model.pre_train_model.named_parameters():
                p.requires_grad = False  
    elif args.task == 'test' or args.multi_stage_finetune:
        print("multi stage finetune")
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)

    model = model.to(args.device)
    return model


def run_finetune(args):

    # make ckpt_dir
    ckpt_dir = 'ckpt/classifier/{}/{}/{}'.format(args.dataset_name, args.model_name, datetime.now().astimezone(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d%H%M"))
    os.makedirs(ckpt_dir, True)
    shutil.copy('finetune_test.py', ckpt_dir)
    if args.model_name == 'ST-MEM':
        shutil.copytree(f'model/st_mem', os.path.join(ckpt_dir, 'st_mem'))
    else:
        shutil.copy(f'model/{args.model_name}.py', ckpt_dir)
    shutil.copy('.vscode/launch.json', ckpt_dir)
    shutil.copy(f'script/finetune/{args.dataset_name}/run_{args.model_name.lower()}_finetune.sh', ckpt_dir)

    model = get_model(args)
    # make data


    train_dataset = PhysionetDataset(args.train_data_path, args.data_standardization, args.dataset_name)
    val_dataset = PhysionetDataset(args.val_data_path, args.data_standardization, args.dataset_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=False)
    
   
    # ready train
    early_stopping = EarlyStopping(patience=args.early_stop_patience, save_dir=ckpt_dir)
    batch_iter = 0

    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    loss_func = torch.nn.CrossEntropyLoss()
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.scheduler_patience, factor=args.scheduler_factor, min_lr=args.scheduler_min_lr)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_epoch_num,
        eta_min=args.scheduler_min_lr
    )
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    for epoch in range(args.max_epoch_num):
        # # if args.task == 'finetune':
        # #     for name, p in model.pre_train_model.named_parameters():
        # #         if 'fc' in name:
        # #             p.requires_grad = True
        # #         else:
        # #             p.requires_grad = False
        # # print(epoch)
        # # 解冻模型的最后两层
        # for param in model.pre_train_model.blocks[-1:].parameters():
        #     param.requires_grad = True
        all_loss = 0
        prog_iter = tqdm(train_dataloader, desc="training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):
            model.train()
            input_x, input_y = tuple(t.to(args.device) for t in batch)
            input_y = input_y.reshape(-1)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            # # 梯度裁剪，避免梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            all_loss += loss.item()
            
        # validate
        print(all_loss)
        all_loss, micro_f1, macro_p, macro_r, macro_f1, auroc = infer(model, val_dataloader, "validating", args.device)
        batch_iter += 1
        scheduler.step(auroc)
        early_stopping(all_loss, auroc, model)
        all_loss = 0
        if early_stopping.early_stop:
            return

def run_test(args):
    model = get_model(args)

    test_dataset = PhysionetDataset(args.test_data_path, args.data_standardization, args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=False)

    all_loss, micro_f1, macro_p, macro_r, macro_f1, auroc = infer(model, test_dataloader, "testing", args.device)