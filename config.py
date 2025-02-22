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
    


# test
#################################################################################################
class TestConfig:
    test_data_name = "index_84-120k_step:1"
    test_data_path = "/root/data/p1-2_people2000_segmentall_sample_step100_data/index_84-120k_step:1.txt"
    class_n = 4
    epoch_num = 100
    batch_size = 256



