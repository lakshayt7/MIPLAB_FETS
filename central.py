import yaml
import pandas as pd
import sys
import copy

DATA_PATH = sys.argv[1] + 'MICCAI_FeTS2021_TrainingData/'
from gandlf_csv_adapter import *

TOTAL_ROUNDS = 5
pardir = DATA_PATH
split_subdirs_path = "/home/lakshayt/scratch/FETS/Notebook/Challenge/Task_1/openfl-workspace/fets_challenge_workspace/partitioning_single.csv"
percent_train = 0.8
federated_simulation_train_val_csv_path = "/home/lakshayt/scratch/FETS/Plain/split.csv"
data_config_path = "/home/lakshayt/scratch/FETS/Plain/split.csv"
collaborator_names = construct_fedsim_csv(pardir, split_subdirs_path, percent_train, federated_simulation_train_val_csv_path)
collaborator_data_loaders = {}


df_paths = pd.read_csv('split.csv')
df_paths.columns = ['0', '1', '2', '3', '4', '5', 'TrainOrVal', 'InstitutionName']
df_paths.to_csv('split_new.csv')



data_loader = FeTSChallengeDataLoader(
data_path = str(int('01')),
federated_simulation_train_val_csv_path = 'split_new.csv',
training_batch_size = 1,
q_samples_per_volume = 1,
q_max_length = 1,
patch_sampler = 'uniform',
psize = [128, 128, 128],
divisibility_factor= 16,
data_usage = 'train-val',
q_verbose = False,
split_instance_dirname = 'fets_phase2_split_1',
np_split_seed = 8950,
allow_auto_split = True,
class_list = ['4', '1||4', '1||2||4'],
data_augmentation = {
    'noise' : 
    {
    'mean' : 0.0,
    'std' : 0.1,
    'probability' : 0.2
    }
    ,
    'rotate_90':
    {
    'axis' : [1, 2, 3],
    'probability' : 0.5
    }
    ,
    'rotate_180':
    {
    'axis': [1, 2, 3],
    'probability': 0.5
    },
    'flip':
    {
    'axis': [0, 1, 2],
    'probability': 1.0
    }
},
data_preprocessing = {
    'crop_external_zero_planes': None,
    'normalize_nonZero_masked': None
},
federated_simulation_institution_name = '__USE_DATA_PATH_AS_INSTITUTION_NAME__'                                                                                            
)



print('Model Initialized')
params_dict = {'epochs_per_round': 0.5, 'num_batches': None , 'learning_rate':0.005}


model = Model_Simple( data_loader = data_loader,
                base_filters = 30,
                lr = 0.005,
                loss_function = 'mirrored_brats_dice_loss',
                opt = 'adam',
                use_penalties = False,
                device = 'cuda',
                validate_with_fine_grained_dice = True,
                sigmoid_input_multiplier = 20.0,
                validation_function = 'fets_phase2_validation',
                validation_function_kwargs = { 
                  'challenge_reduced_output': True
                }
            )
train_losses = {}
val_metrics = {}
for r in range(20):
    save_path, loss = model.train_batches(col_name = 1, round_num = r+1, params_dict = params_dict)
    train_losses[r] = loss
    print('Training Loss for epoch = ' + str(r*0.5))
    print(train_losses[r])
    out = model.validate(col_name = 1, round_num = r+1)
    val_metrics[r] = out
    print('Validation Metrics for epoch = ' + str(r*0.5))
    print(val_metrics[r])
    for key, value in val_metrics[r].items():
        print(key, ' : ', value)
print(train_losses)
print(val_metrics)
