import os
import argparse
import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import pandas as pd

import qamodels
import models
from utils.utils import *
from models import all_models
from utils.utils_inference import *

parser = argparse.ArgumentParser(
    description="Computing the Scores for Different Question Types"
)

parser.add_argument(
    '--use_cuda', type=bool, default=True, help = "Use gpu"
)

parser.add_argument(
    '--gpu', type=int, default=1, help = "On which gpu to run code"
)

parser.add_argument(
    '--preprocessing', type=bool, default=False, help = "Do preprocessing of data"
)

parser.add_argument(
    '--get_curvature', type=bool, default=False, help = "Do preprocessing and get the curvature"
)

# Exporting enviromental variables

qa_data_path = os.environ['DATA_PATH']
checkpoints = os.environ['CHECKPOINTS']
kge_path = os.environ['KGE_PATH']
config_files = os.environ['CONFIG_FILES']
inference_path = os.environ['INFERENCE_PATH']

#-------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

    kge_data_path = "{}/data".format(kge_path)  
    model_path = "{}/fbwq".format(checkpoints)
    device = torch.device(0 if args.use_cuda else 'cpu')

    ##  Prepross all the mentioned data
    if args.preprocessing:
        for kg_type in ['full', 'half']:
            dataset_path = "{}/fbwq_{}".format(kge_data_path, 'half')
            cmd = "python {}/preprocess/preprocess_default.py {}".format(kge_data_path, dataset_path)
            os.system(cmd)

    if args.get_curvature:
        cmd = "python qtype_curv.py --dataset fbwq"
        os.system(cmd)
    
    ##  Read the Curvatures
    full_file = "{}/fbwq_curv_full.txt".format(inference_path)
    half_file = "{}/fbwq_curv_half.txt".format(inference_path)
    ##  ipath referring to the different inferencial path, key: ipath, value: curvature
    ipath_dict_full = read_curvature_fbwq(full_file)
    ipath_dict_half = read_curvature_fbwq(half_file)

    fbwq_data_path = '{}/QA_data/WebQuestionsSP'.format(qa_data_path)
    pruning_file = '{}/pruning_test.txt'.format(fbwq_data_path)

    ## Counting how many models to evaluate
    count = 0
    for entry in os.scandir(model_path):
        if entry.path.endswith(".pt"):
            f_list = entry.path.split('_')
            dataset = f_list[1].split('/')[-1]
            kg_type, model, dim = f_list[2:5]
            hops = f_list[-1].split('.')[0]
            
            ##  Models to evaluate
            if model not in ['AttH', 'TransE']:
                continue
            count+=1
        else:
            continue
    
    ##  Counting how many train data for each ipath
    ipath_count = count_train_fbwq(fbwq_data_path)
    
    outFile = open('{}/fbwq_relation_performance.csv'.format(inference_path),'w')
    outFile.write('file name,dataset,kg type,model,dimension,hops,inference path,curvature,accuracy,total correct,test size,train size\n')
    outFile.close()

    ##  Reading QA dataset
    train_data, _, test_data, check_length = read_qa_dataset('fbwq', None, qa_data_path, None)
    train_samples = process_text_file(train_data, check_length)
    
    progress = 1
    for entry in os.scandir(model_path):
        if entry.path.endswith(".pt"):
            f_list = entry.path.split('_')
            dataset = f_list[1].split('/')[-1]
            kg_type, model, dim = f_list[2:5]
            hops = f_list[-1].split('.')[0]

            file_name = '{}_{}_{}_{}_{}.pt'.format(dataset, kg_type, model, dim, hops)

            if model not in ['AttH', 'TransE']:
                continue
            
            print("Working on {}/{} th Model".format(progress, count))

            if dataset == 'MetaQA':
                qa_nn_type = 'LSTM'
            else:
                qa_nn_type = 'RoBERTa'

            dataset_path = "{}/{}_{}".format(kge_data_path, dataset, kg_type)

            ##  Get entity2idx and word2idx
            entity2idx, word2idx = prepare_mapping(dataset_path, train_samples)
            
            ##  Loading QA Model
            print('Loading QA model from {}'.format(file_name))
            qa_model = torch.load('{}/{}'.format(model_path, file_name))
            qa_model.to(device)

            current = 1
            for ipath in ipath_dict_full.keys():
                print("Working on {}/{}th inferential path".format(current, len(ipath_dict_full)))
                test_samples = process_text_fbwq(test_data, check_length, ipath, pruning_file)
                print("Calculating Score for {}".format(ipath))
                if len(test_samples) != 0:
                    qa_model.eval()
                    _, accuracy, _, total_correct = calculate_valid_loss_inference(qa_model, test_samples, qa_nn_type, device, entity2idx, word2idx, measure_time = False)
                with open('{}/fbwq_relation_performance.csv'.format(inference_path),'a') as outFile:
                    if kg_type == 'full':
                        outFile.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:.4f},{9},{10},{11}\n'.format(
                            file_name, dataset, kg_type, model, dim, hops, ipath, ipath_dict_full[ipath], accuracy if len(test_samples)!= 0 else 0, total_correct if len(test_samples)!= 0 else 0, len(test_samples),ipath_count[ipath] if ipath in ipath_count.keys() else 0
                            ))
                    elif kg_type == 'half':
                        outFile.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:.4f},{9},{10},{11}\n'.format(
                            file_name, dataset, kg_type, model, dim, hops, ipath, ipath_dict_half[ipath], accuracy if len(test_samples)!=0 else 0, total_correct if len(test_samples)!= 0 else 0, len(test_samples), ipath_count[ipath] if ipath in ipath_count.keys() else 0
                            ))
                print("Accuracy is {}".format(accuracy if len(test_samples)!= 0 else '0 (Test sample size is 0'))
                current+=1
        
        else:
            continue
        progress += 1

    ##  Sort the data
    results_df = pd.read_csv('{}/fbwq_relation_performance.csv'.format(inference_path))
    results_df.sort_values(['dataset','kg type','model','dimension','curvature'], axis=0, inplace=True)
    results_df.to_csv('{}/fbwq_relation_performance.csv'.format(inference_path))
    


