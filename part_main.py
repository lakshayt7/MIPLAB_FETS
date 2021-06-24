import yaml
import pandas as pd
import sys
import copy
import numpy as np
import textwrap
import os
from gandlf_csv_adapter import *

col_list = sys.argv[1]
DATA_PATH = sys.argv[2] + 'MICCAI_FeTS2021_TrainingData/'
round_num = sys.argv[3]
save_path = sys.argv[4]

aggregate_path = save_path+"|aggregate"+'|'+str(int(round_num)-1)+'.pth'

print('AGGREGATE PATH = ')
print(aggregate_path)

col_list = textwrap.wrap(col_list, 2)

print('list of collaborators = ')
print(col_list)
print('DATA PATH')
print(DATA_PATH)



model_paths = {}


print('round number = ')
print(round_num)

pardir = DATA_PATH
split_subdirs_path = "/home/lakshayt/scratch/FETS/Notebook/Challenge/Task_1/openfl-workspace/fets_challenge_workspace/partitioning_1.csv"
percent_train = 0.8
federated_simulation_train_val_csv_path = "/home/lakshayt/scratch/FETS/Plain/split.csv"
collaborator_names = construct_fedsim_csv(pardir, split_subdirs_path, percent_train, federated_simulation_train_val_csv_path)
collaborator_data_loaders = {}


df_paths = pd.read_csv('~/scratch/FETS/Plain/split.csv')
df_paths.columns = ['0', '1', '2', '3', '4', '5', 'TrainOrVal', 'InstitutionName']
df_paths.to_csv('~/scratch/FETS/Plain/split_new.csv')


for col in collaborator_names:

    collaborator_data_loaders[col]  = FeTSChallengeDataLoader(
    data_path = str(int(col)),
    federated_simulation_train_val_csv_path = '~/scratch/FETS/Plain/split_new.csv',
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
print('Collaborator Names')
print(collaborator_names)
params_dict = {'epochs_per_round': 0.5, 'num_batches': None , 'learning_rate':0.005}


for col in col_list:
    print('Initializing model for collaborator ' + str(col))
    model = Model_Simple( data_loader = collaborator_data_loaders[col],
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
    if os.path.exists(aggregate_path):
        print('Loading Model from '+str(aggregate_path)+'for collabortor '+str(col))
        model.load_cpt(aggregate_path)
    else:
        print('Starting from scratch for collaborator'+str(col))
    new_path, loss = model.train_batches(col_name = col, round_num = round_num, params_dict = params_dict, save_path = save_path+'|'+str(round_num)+'|'+col+'.pth')
    print('Model Trained for collaborator '+str(col))
    print('Loss for collaborator = ')
    print(loss)

print('Model Trained')
