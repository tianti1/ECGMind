class EmptyConfig:
    pass

# pretrain
#################################################################################################
class PreTrainConfig:
    data_standardization = True
    batch_size = 512
    epoch_num = 1000
    train_data_path = '/root/data/p1-2_people2000_segmentall_sample_step100_data/index_0-42k_step:1.txt'
    val_data_path = '/root/data/p1-2_people2000_segmentall_sample_step100_data/index_42-60k_step:1.txt'
    val_every_n_steps = 40

class FocusMaePreTrainConfig(PreTrainConfig):
    winsize = 1200
    patch_size = 30


class FocusMergeMaePreTrainConfig(PreTrainConfig):
    winsize = 1200
    patch_size = 30



# finetune
#################################################################################################
class FineTuneConfig:
    train_data_name = "physionet_index_62-72k"
    val_data_name = "physionet_index_76-82k"
    train_data_path = "/root/data/p1-2_people2000_segmentall_sample_step100_data/index_62-74k_step:1.txt"
    val_data_path = "/root/data/p1-2_people2000_segmentall_sample_step100_data/index_76-82k_step:1.txt"
    class_n = 4
    epoch_num = 100
    batch_size = 128
    pretrain_model_freeze = True

class focusMaeFineTuneConfig(FineTuneConfig):

    ckpt_path = "/root/ecg_ai/FocusECG/FocusECG/ckpt/pre_train/p1-2_people2000_segmentall/focusmae/202412262158/min_val_loss=21.37238883972168.pth"
    pretrain_model_freeze = True
    winsize = 1200
    patch_size = 30
    


# test
#################################################################################################
class TestConfig:
    test_data_name = "index_84-120k_step:1"
    test_data_path = "/root/data/p1-2_people2000_segmentall_sample_step100_data/index_84-120k_step:1.txt"
    class_n = 4
    epoch_num = 100
    batch_size = 256


class focusMaeTestConfig(TestConfig):
    ckpt_path="/root/ecg_ai/FocusECG/FocusECG/ckpt/classifier/physionet_index_62-72k/physionet_index_76-82k/focusmae+mlp_v1/normal/202412280106/max_f1=0.8528546700103565.pth"
    winsize = 1200
    patch_size = 30


