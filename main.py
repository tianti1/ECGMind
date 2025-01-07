import torch
import argparse
from pretrain import run_pretrain
from finetune_test import run_finetune, run_test

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain model parameter configuration')
    # task
    parser.add_argument('--task', type=str, default='pretrain', choices=['pretrain', 'finetune', 'test'], help='Task type')
    # dataset
    parser.add_argument('--dataset_name', type=str, default='p1-2_people2000_segmentall_sample_step100_data', help='Dataset name')
    parser.add_argument('--train_data_path', type=str, default='/root/data/p1-2_people2000_segmentall_sample_step100_data/index_0-42k_step:1.txt', help='Training data path')
    parser.add_argument('--val_data_path', type=str, default='/root/data/p1-2_people2000_segmentall_sample_step100_data/index_42-60k_step:1.txt', help='Validation data path')  
    parser.add_argument('--test_data_path', type=str, default='/root/data/p1-2_people2000_segmentall_sample_step100_data/index_60-80k_step:1.txt', help='Test data path')
    parser.add_argument('--data_standardization', type=str, default='true', choices=['true', 'false'], help='Data standardization')
    # train
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--max_epoch_num', type=int, default=1000, help='Maximum training epoch number')
    parser.add_argument('--val_every_n_steps', type=int, default=40, help='Validate every n steps')
    parser.add_argument('--early_stop_patience', type=int, default=50, help='Early stop patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Scheduler patience')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Scheduler factor')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6, help='Scheduler minimum learning rate')
    parser.add_argument('--pretrain_model_freeze', type=str, default='true', choices=['true', 'false'], help='Freeze pretrain model')
    parser.add_argument('--ckpt_path', type=str, default='', help='Checkpoint path')
    parser.add_argument('--classifier_head_name', type=str, default='mlp_v1', help='Classifier head name')
    parser.add_argument('--class_n', type=int, default=4, help='Number of classes')
    # model
    parser.add_argument('--model_name', type=str, default='PatchTST', help='Model name')
    parser.add_argument('--num_input_channels', type=int, default=1, help='Number of input channels, [PatchTST]')
    parser.add_argument('--signal_length', type=int, default=2500, help='Signal length, [focusmae, PatchTST]')
    parser.add_argument('--patch_length', type=int, default=50, help='Patch length, [focusmae, PatchTST]')
    parser.add_argument('--patch_stride', type=int, default=50, help='Patch stride, [PatchTST]')
    parser.add_argument('--patch_size', type=int, default=50, help='Patch size, [focusmae]')
    parser.add_argument('--embed_dim', type=int, default=256, help='Backbone output dimension, [focusmae, PatchTST]')
    parser.add_argument('--encoder_depth', type=int, default=48, help='Encoder depth, [focusmae]')
    parser.add_argument('--encoder_num_heads', type=int, default=16, help='Encoder number of heads, [focusmae]')
    parser.add_argument('--decoder_embed_dim', type=int, default=64, help='Decoder dimension, [focusmae]')
    parser.add_argument('--decoder_depth', type=int, default=8, help='Decoder depth, [focusmae]')
    parser.add_argument('--decoder_num_heads', type=int, default=8, help='Decoder number of heads, [focusmae]')
    parser.add_argument('--mlp_ratio', type=int, default=2, help='MLP ratio, [focusmae]')
    parser.add_argument('--norm_layer', type=str, default='LayerNorm', help='Norm layer, [focusmae]')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Mask ratio, [focusmae, PatchTST]')
    parser.add_argument('--mask_type', type=str, default='random', help='Mask type [focusmae, PatchTST]')
    parser.add_argument('--all_encode_norm_layer', type=str, default='LayerNorm', help='Encoder norm layer, [focusmae]')
    parser.add_argument('--use_cls_token', type=str, default='true', choices=['true', 'false'], help='Use cls token, [PatchTST]')
    parser.add_argument('--num_layers', type=int, default=48, help='Number of layers, [PatchTSMixer]')
    parser.add_argument('--self_attn', type=str, default='true', choices=['true', 'false'], help='Use self attention, [PatchTSMixer]')
    parser.add_argument('--use_positional_encoding', type=str, default='true', choices=['true', 'false'], help='Use positional encoding, [PatchTSMixer]')
    parser.add_argument('--self_attn_heads', type=int, default=16, help='Number of heads, [PatchTSMixer]')


    args = parser.parse_args()
    
    args.data_standardization = args.data_standardization.lower() == 'true'
    args.pretrain_model_freeze = args.pretrain_model_freeze.lower() == 'true'
    args.use_cls_token = args.use_cls_token.lower() == 'true'
    args.self_attn = args.self_attn.lower() == 'true'
    args.use_positional_encoding = args.use_positional_encoding.lower() == 'true'
    
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)
    print(args.device)
    
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.task == 'pretrain':
        run_pretrain(args)
    elif args.task == 'finetune':
        run_finetune(args)
    elif args.task == 'test':
        run_test(args)
    else:
        raise ValueError(f'Invalid task: {args.task}')
