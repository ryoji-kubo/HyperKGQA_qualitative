from utils.utils import *
import torch
from transformers import *
from tqdm import tqdm
import time
import numpy as np

import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
from collections import defaultdict

import pprint

##  Get entity2idx and word2idx
def prepare_mapping(dataset_path, train_samples):
    entity_dict = '{}/entity_ids.del'.format(dataset_path)
    e = []
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e.append(ent_name)
    else:
        f.close()
     
    entity2idx = {}
    i = 0
    embedding_matrix = []
    for key in e:
        entity2idx[key] = i
        i += 1

    word2idx, _, _ = get_vocab(train_samples)

    return entity2idx, word2idx

def data_generator_lstm(data, entity2idx, word2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1].strip().split(' ')
        encoded_question = [word2idx[word.strip()] for word in question]
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long),torch.tensor(encoded_question, dtype=torch.long) , ans, torch.tensor(len(encoded_question), dtype=torch.long), data_sample[1]

def pad_sequence(arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

def tokenize_question(question, tokenizer):
    question = "<s> " + question + " </s>"
    question_tokenized = tokenizer.tokenize(question)
    question_tokenized = pad_sequence(question_tokenized, 64)
    question_tokenized = torch.tensor(tokenizer.encode(question_tokenized, add_special_tokens=False))
    attention_mask = []
    for q in question_tokenized:
        # 1 means padding token
        if q == 1:
            attention_mask.append(0)
        else:
            attention_mask.append(1)
    return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

def data_generator_bert(data, entity2idx):
        tokenizer_class = RobertaTokenizer
        pretrained_weights = 'roberta-base'
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, cache_dir='.')

        for i in range(len(data)):
            data_sample = data[i]
            head = entity2idx[data_sample[0].strip()]
            question = data_sample[1]
            question_tokenized, attention_mask = tokenize_question(question, tokenizer)
            if type(data_sample[2]) is str:
                ans = entity2idx[data_sample[2]]
            else:
                ans = []
                for entity in list(data_sample[2]):
                    if entity.strip() in entity2idx:
                        ans.append(entity2idx[entity.strip()])

            yield torch.tensor(head, dtype=torch.long), question_tokenized, ans, attention_mask, data_sample[1]

def calculate_valid_loss_inference(model, samples, qa_nn_type, device, entity2idx, word2idx, measure_time=False):
        answers = []
        candidates_with_scores = []
        recorded_time = []

        if qa_nn_type == 'LSTM':
            data_gen = data_generator_lstm(samples, entity2idx, word2idx)
        else:
            data_gen = data_generator_bert(samples, entity2idx)
        total_correct = 0

        for i in tqdm(range(len(samples))):
            start = time.time_ns()

            d = next(data_gen)

            head = d[0].to(device)
            question_tokenized = d[1].unsqueeze(0).to(device)
            ans = d[2]
            attention_mask = d[3].unsqueeze(0).to(device)

            scores = model.get_score_ranked(head, question_tokenized, attention_mask)
            top_2 = torch.topk(scores, k=2, largest=True, sorted=True)
            top_2_idx = top_2[1].tolist()[0]
            head_idx = head.tolist()

            if top_2_idx[0] == head_idx:
                pred_ans = top_2_idx[1]
            else:
                pred_ans = top_2_idx[0]

            if type(ans) is int:
                ans = [ans]

            is_correct = 0
            if pred_ans in ans:
                total_correct += 1
                is_correct = 1

            q_text = d[-1]
            answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))

            end = time.time_ns()
            recorded_time.append(end-start)


        # print(total_correct)
        accuracy = total_correct/len(samples)

        recorded_time = np.array(recorded_time)
        average_time_ms = 1/1000 * np.mean(recorded_time)

        if measure_time:
            print('Average Time: {:.2f} microseconds'.format(average_time_ms))

        return answers, accuracy, average_time_ms, total_correct

def read_curvature(file):
    qtype_curv_dict = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            index = line.find(',')
            qtype_curv_dict[line.strip().split(',')[0]] = line.strip()[index+1:]
    return qtype_curv_dict

##  read the fbwq curvature files, return dictionary holding inference path as key and its curvature as value
def read_curvature_fbwq(file):
    curv_dict = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            index = line.rindex(',', beg=0, end=len(line))
            curv_dict[line.strip().split(',')[0]] = line[index+1:].strip()
    return curv_dict

##create test files for different question types
def create_testfiles(qa_data_path, hops, qtype_dict):
    testfiles_dict = {}
    files = {}

    metaqa_data_path = '{}/QA_data/MetaQA'.format(qa_data_path)

    test_dp = '{}/qa_test_{}hop.txt'.format(metaqa_data_path, hops)
    test_qtype = '{}/qa_test_qtype_{}hop.txt'.format(metaqa_data_path, hops)

    for type in qtype_dict:
        testfiles_dict[type] = "{}/qa_test_{}hop_{}.txt".format(metaqa_data_path, hops, type)
        f = open(testfiles_dict[type], 'w')
        files[type] = f
    
    with open(test_dp, 'r') as testFile, open(test_qtype, 'r') as qtypeFile: 
        for test_line, qtype in zip(testFile, qtypeFile):
            for type in qtype_dict:
                if qtype.strip() == type:
                    files[type].write(test_line)

    for type in qtype_dict:
        files[type].close()

    return testfiles_dict

##  count from training file how many questions are there for each question type
def count_train(qa_data_path, hops):
    metaqa_data_path = '{}/QA_data/MetaQA'.format(qa_data_path)
    qtype_count = {}
    train_qtype = '{}/qa_train_qtype_{}hop.txt'.format(metaqa_data_path, hops)
    with open(train_qtype, 'r') as qtypeFile: 
        for qtype in qtypeFile.readlines():
            if qtype.strip() not in qtype_count.keys():
                qtype_count[qtype.strip()] = 1
            else:
                qtype_count[qtype.strip()]+=1
    return qtype_count

##  count from training file how many questions are there for each inferential path
def count_train_fbwq(fbwq_data_path, check_length = True):
    ipath_count = {}
    pruning_file = '{}/pruning_train.txt'.format(fbwq_data_path)
    training_file = '{}/qa_train_webqsp.txt'.format(fbwq_data_path)

    training_file = open(training_file, 'r')
    pruning_file = open(pruning_file, 'r')

    for pruning_line in pruning_file.readlines():
        pruning_line = pruning_line.strip().split('\t')
        
        if pruning_line[1] not in ipath_count.keys():
            ipath_count[pruning_line[1]] = 0

        match = False
        training_file.seek(0, 0)
        for data_line in training_file.readlines():
            data_line = data_line.strip().split('\t')
            
            ##  if not match skip
            if data_line[0] != pruning_line[0]:
                continue
            if check_length and len(data_line) != 2:
                # print('check_length error')
                continue
            
            match = True

            if match:
                ipath_count[pruning_line[1]] += 1
                break

    training_file.close()
    pruning_file.close()

    return ipath_count


def process_text_fbwq(text_file, check_length, ipath, pruning_file, split=False):
    
    data_array = []
    data_file = open(text_file, 'r')
    pruning_file = open(pruning_file, 'r')
    
    matches = 0
    for pruning_line in pruning_file.readlines():
        pruning_line = pruning_line.strip().split('\t')
        
        if pruning_line[1].strip() != ipath:
            continue
        
        match = False
        data_file.seek(0, 0)
        for data_line in data_file.readlines():
            data_line = data_line.strip().split('\t')
            
            ##  if not match skip
            if data_line[0] != pruning_line[0]:
                continue
            if check_length and len(data_line) != 2:
                # print('check_length error')
                continue
            
            match = True
            
            question = data_line[0].split('[')
            question_1 = question[0]
            question_2 = question[1].split(']')
            head = question_2[0].strip()
            question_2 = question_2[1]
            question = question_1 + 'NE' + question_2
            ans = data_line[1].split('|')
            data_array.append([head, question.strip(), ans])

            if match:
                matches += 1
                break

        ##  This is to check for questions that exist in pruning_test.txt but not in qa_test_webqsp.txt
        # if match == False:
        #     pprint.pprint(pruning_line)
    
    # print(matches)

    data_file.close()
    pruning_file.close()
    if split == False:
        return data_array
    data = []
    for line in data_array:
        head = line[0]
        question = line[1]
        tails = line[2]
        for tail in tails:
            data.append([head, question, tail])
            
    return data