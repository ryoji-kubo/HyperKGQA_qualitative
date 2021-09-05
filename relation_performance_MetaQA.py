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
    "--hops", type=str, default='3', help = "Number of edges to reach the answer"
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
    model_path = "{}/MetaQA".format(checkpoints)
    device = torch.device(0 if args.use_cuda else 'cpu')

    ##  Prepross all the mentioned data
    if args.preprocessing:
        for kg_type in ['full', 'half']:
            dataset_path = "{}/MetaQA_{}".format(kge_data_path, 'half')
            cmd = "python {}/preprocess/preprocess_default.py {}".format(kge_data_path, dataset_path)
            os.system(cmd)
            
    ##  Get the curvature
    if args.get_curvature:
        cmd = "python qtype_curv.py --dataset MetaQA --hops {}".format(args.hops)
        os.system(cmd)


    ##  Read the Curvatures
    full_file = "{}/qtype_curv_{}_full.txt".format(inference_path, args.hops)
    half_file = "{}/qtype_curv_{}_half.txt".format(inference_path, args.hops)
    qtype_dict_full = read_curvature(full_file)
    qtype_dict_half = read_curvature(half_file)

    ## Counting how many models to evaluate
    count = 0
    for entry in os.scandir(model_path):
        if entry.path.endswith(".pt"):
            f_list = entry.path.split('_')
            dataset = f_list[1].split('/')[-1]
            kg_type, model, dim = f_list[2:5]
            hops = f_list[-1].split('.')[0]
            
            ##  Models to evaluate
            if model not in ['AttH'] or hops != args.hops:
                continue
            count+=1
        else:
            continue
    
    testfiles_dict = create_testfiles(qa_data_path, args.hops, qtype_dict_full)
    qtype_count = count_train(qa_data_path, args.hops)

    outFile = open('{}/MetaQA_relation_performance_{}.csv'.format(inference_path, args.hops),'w')
    outFile.write('file name,dataset,kg type,model,dimension,hops,question type,relations,curvature,accuracy,test size,train size\n')
    outFile.close()

    progress = 1
    for entry in os.scandir(model_path):
        if entry.path.endswith(".pt"):
            f_list = entry.path.split('_')
            dataset = f_list[1].split('/')[-1]
            kg_type, model, dim = f_list[2:5]
            hops = f_list[-1].split('.')[0]

            file_name = '{}_{}_{}_{}_{}.pt'.format(dataset, kg_type, model, dim, hops)

            if model not in ['AttH'] or hops != args.hops:
                continue
            
            print("Working on {}/{} th Model".format(progress, count))

            if dataset == 'MetaQA':
                qa_nn_type = 'LSTM'
            else:
                qa_nn_type = 'RoBERTa'

            dataset_path = "{}/{}_{}".format(kge_data_path, dataset, kg_type)

            
            ##  Reading QA dataset
            train_data, _, _, check_length = read_qa_dataset(dataset, hops, qa_data_path, kg_type)
            train_samples = process_text_file(train_data, check_length)

            ##  Get entity2idx and word2idx
            entity2idx, word2idx = prepare_mapping(dataset_path, train_samples)
            
            ##  Loading QA Model
            print('Loading QA model from {}'.format(file_name))
            qa_model = torch.load('{}/{}'.format(model_path, file_name))
            qa_model.to(device)

            for question_type, relation_test_data in testfiles_dict.items():
                test_samples = process_text_file(relation_test_data, check_length)
                print("Calculating Score for {}".format(question_type))
                qa_model.eval()
                _, accuracy, _, _ = calculate_valid_loss_inference(qa_model, test_samples, qa_nn_type, device, entity2idx, word2idx, measure_time = False)
                with open('{}/MetaQA_relation_performance_{}.csv'.format(inference_path, args.hops),'a') as outFile:
                    if kg_type == 'full':
                        outFile.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:.4f},{9},{10}\n'.format(file_name, dataset, kg_type, model, dim, hops, question_type, qtype_dict_full[question_type], accuracy, len(test_samples), qtype_count[question_type]))
                    elif kg_type == 'half':
                        outFile.write('{0},{1},{2},{3},{4},{5},{6},{7},{8:.4f},{9},{10}\n'.format(file_name, dataset, kg_type, model, dim, hops, question_type, qtype_dict_half[question_type], accuracy, len(test_samples), qtype_count[question_type]))
                print("Accuracy is {}".format(accuracy))
        
        else:
            continue
        progress += 1

    ##  Sort the data
    results_df = pd.read_csv('{}/MetaQA_relation_performance_{}.csv'.format(inference_path, args.hops))
    results_df.sort_values(['dataset','kg type','model','dimension','curvature'], axis=0, inplace=True)
    results_df.to_csv('{}/MetaQA_relation_performance_{}.csv'.format(inference_path, args.hops))

