import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_dir = ''):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_loss = None
        self.max_f1 = None
        self.early_stop = False
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, val_loss, val_f1, model, always_save=False):
        print("min_loss:{}, max_f1:{}, val_loss:{}, val_f1:{}".format(self.min_loss, self.max_f1, val_loss, val_f1))
        if always_save:
            self.save_checkpoint(model, 'max_f1', val_f1, always_save)
            self.counter += 1
            if self.counter >= 30:
                self.early_stop = True
        else:
            if self.min_loss is None:
                self.min_loss = val_loss
                self.max_f1 = val_f1
                self.save_checkpoint(model, 'max_f1', self.max_f1)
            elif val_loss < self.min_loss - self.delta:
                self.min_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
    
            if val_f1 > self.max_f1:
                self.max_f1 = val_f1
                self.counter = 0
                self.save_checkpoint(model, 'max_f1', self.max_f1)
            
    def check(self, loss, model):
        if self.min_loss is None:
            self.min_loss = loss
            self.save_checkpoint(model, 'min_val_loss', self.min_loss)
        elif loss < self.min_loss - self.delta:
            self.min_loss = loss
            self.counter = 0
            self.save_checkpoint(model, 'min_val_loss', self.min_loss)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                    
    def save_checkpoint(self, model, score_name, score_value, always_save=False):
        if not always_save:
            for filename in os.listdir(self.save_dir):
                if filename.startswith("max") or filename.startswith("min"):
                    file_path = os.path.join(self.save_dir, filename)
                    os.remove(file_path)
        path = os.path.join(self.save_dir, f"{score_name}={score_value}.pth")
        torch.save(model.state_dict(), path)

