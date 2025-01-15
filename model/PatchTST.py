import torch
import torch.nn as nn
from transformers import PatchTSTForPretraining, PatchTSTForClassification, PatchTSTConfig


class PatchTST(nn.Module):
    name = "PatchTST"
    
    def __init__(self, args):
        super().__init__()
        if args.task == "pretrain":
            self.model = PatchTSTForPretraining(PatchTSTConfig(
                num_input_channels=args.num_input_channels,
                context_length=args.signal_length,
                patch_length=args.patch_length,
                patch_stride=args.patch_stride,
                mask_type=args.mask_type,
                random_mask_ratio=args.mask_ratio,
                use_cls_token=args.use_cls_token,
                d_model=args.embed_dim,
                num_hidden_layers=args.encoder_depth,
                num_attention_heads=args.encoder_num_heads,
            ))
        elif args.task == "finetune":
            self.model = PatchTSTForClassification(PatchTSTConfig(
                num_input_channels=args.num_input_channels,
                context_length=args.signal_length,
                patch_length=args.patch_length,
                patch_stride=args.patch_stride,
                mask_type=args.mask_type,
                random_mask_ratio=args.mask_ratio,
                use_cls_token=args.use_cls_token,
                num_targets=args.class_n,
                d_model=args.embed_dim,
                num_hidden_layers=args.encoder_depth,
                num_attention_heads=args.encoder_num_heads,
            ))
            checkpoint = torch.load(args.ckpt_path, map_location='cpu')
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("model.model."):
                    new_key = k[len("model.model."):]
                    new_state_dict[new_key] = v
            self.model.model.load_state_dict(new_state_dict, strict=True)
        elif args.task == "test":
            self.model = PatchTSTForClassification(PatchTSTConfig(
                num_input_channels=args.num_input_channels,
                context_length=args.signal_length,
                patch_length=args.patch_length,
                patch_stride=args.patch_stride,
                mask_type=args.mask_type,
                random_mask_ratio=args.mask_ratio,
                use_cls_token=args.use_cls_token,
                num_targets=args.class_n,
                d_model=args.embed_dim,
                num_hidden_layers=args.encoder_depth,
                num_attention_heads=args.encoder_num_heads,
            ))
            checkpoint = torch.load(args.ckpt_path, map_location='cpu')
            self.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        x = x.transpose(2, 1)
        outputs = self.model(past_values=x)
        logits = outputs.prediction_logits
        return logits
    
    def forward_loss(self, x: torch.Tensor):
        x = x.transpose(2, 1)
        outputs = self.model(past_values=x)
        loss = outputs.loss
        return loss
    
    def forward_feature(self, x):
        x = x.transpose(2, 1)
        model_output = self.model.model(past_values=x)
        return model_output.last_hidden_state
    
def patchtst_prefer_custom(args):
    model = PatchTST(args)
    return model