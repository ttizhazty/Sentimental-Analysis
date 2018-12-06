import json
from tqdm import tqdm
import pandas as pd
import ast
import pickle
import csv
import re 
import numpy as np
import os
import pdb
import scipy
from collections import Counter

def data_convert_csv(data_path, input_file_name, output_file_name):
    if os.path.isfile(data_path + output_file_name):
        print(output_file_name + ' exists')
    else:
        print('converting json file into csv file')
        with open(data_path + input_file_name, encoding="utf8") as json_file:
            data = json_file.readlines()  # list of string  
        # store the data in csv file 
        with open(data_path+ output_file_name, 'w', encoding='utf-8') as f:
            # write the heading 
            review_dict = ast.literal_eval(data[0])
            heading = review_dict.keys()
            writer = csv.writer(f)
            writer.writerow(list(heading))
            for index in tqdm(range(len(data))):
                #if index > 300000:
                #    break
                review_dict = json.loads(data[index])
                writer = csv.writer(f)
                writer.writerow(review_dict.values())
        print('converting complete !!!')

def data_sample_csv(data_path, input_file_name, sample_file_name):
    if os.path.isfile(data_path+ sample_file_name):
        print(sample_file_name + ' exists')
    else:
        print('converting json file into csv file')
        with open(data_path + input_file_name, encoding="utf8") as json_file:
            data = json_file.readlines()  # list of string 
        print("finished json file load") 
        # store the data in csv file 
        with open(data_path+ sample_file_name, 'w', encoding="utf8") as f:
            # write the heading 
            review_dict = ast.literal_eval(data[0])
            heading = review_dict.keys()
            writer = csv.writer(f)
            writer.writerow(list(heading))
            for index in tqdm(range(len(data[:1000]))):
                review_dict = json.loads(data[index])
                writer = csv.writer(f)
                writer.writerow(review_dict.values())
        print('converting complete !!!')

def split_feature_label(data_path, sample_file_name, data_export, test_size):
    # if os.path.isfile(data_export + train_data_file_name) and os.path.isfile(data_export + test_data_file_name):
    #     print(train_data_file_name + ' exists !!!')
    #     print(test_data_file_name + ' exists !!!')
    if os.path.isfile(data_export + 'all_data.data'):
        print('raw data is exist !!!')
    else:
        df = pd.read_csv(data_path + sample_file_name, low_memory=True) 
        print('finished load data')
        num_samples = len(df['stars'])
        with open(data_export + 'all_data.data', 'w', encoding='utf8') as f:
            for i in tqdm(range(num_samples)):
                label = str(df['stars'][i])
                raw_text = df['text'][i]
                try:
                    f.write(label + '\t\t\t' + raw_text + '\n')
                except TypeError:
                    continue
        print('file wirtten has completed !!!')

    label_list, text_list = loadData(data_export + 'all_data.data')
    #analysis for training data:
    print(Counter(label_list))
    valid_label_list = []
    valid_text_list = []
    for i in range(len(text_list)):
        if 20 <= len(text_list[i].split(' ')) <= 60:
            valid_text_list.append(text_list[i])
            valid_label_list.append(label_list[i])
    print('number of valid samples: ', len(valid_text_list))
    print('valid data label distribution:', Counter(valid_label_list))
    label2text_dict = {}
    for j in range(len(valid_text_list)):
        if valid_label_list[j] not in label2text_dict:
            label2text_dict[valid_label_list[j]] = []
        label2text_dict[valid_label_list[j]].append(valid_text_list[j])
    pickle.dump(label2text_dict, open('./yelp_dataset/label2text.pkl', 'wb'))
    star_4_list = label2text_dict[4]
    star_2_list = label2text_dict[2]
    star_3_list = label2text_dict[3]
    star_1_list = label2text_dict[1]
    # print(len(star_4_list), len(star_3_list), len(star_2_list), len(star_1_list))


    new_star_4_list = upsample(star_4_list, 2) # 40234
    new_star_2_list = upsample(star_2_list, 2) # 7816
    new_star_3_list = upsample(star_3_list, 2) # 14052
    new_star_1_list = upsample(star_1_list, 6) # 15236
    print(len(new_star_4_list), len(new_star_3_list), len(new_star_2_list), len(new_star_1_list))
    star2text_pair_5 = [(5,x) for x in label2text_dict[5]] # 101860
    star2text_pair_4 = [(4,x) for x in new_star_4_list]
    star2text_pair_3 = [(3,x) for x in new_star_3_list]
    star2text_pair_2 = [(2,x) for x in new_star_2_list]
    star2text_pair_1 = [(1,x) for x in new_star_1_list]

    pickle.dump(star2text_pair_5, open('./yelp_dataset/positive_text.pkl','wb'))
    pickle.dump(star2text_pair_4 + star2text_pair_3 + star2text_pair_2, open('./yelp_dataset/neutral_text.pkl', 'wb'))
    pickle.dump(star2text_pair_1, open('./yelp_dataset/negative_text.pkl', 'wb'))

    balance_data_list = star2text_pair_5 + star2text_pair_4 + star2text_pair_3 + star2text_pair_2 + star2text_pair_1
    np.random.shuffle(balance_data_list)
    cut_idx = int((1 - test_size) * len(balance_data_list))
    label_list = []
    text_list = []
    for item in balance_data_list:
        label_list.append(item[0])
        text_list.append(item[1])
    train_label_list = label_list[: cut_idx]
    train_text_list = text_list[: cut_idx]
    test_label_list = label_list[cut_idx:]
    test_text_list = text_list[cut_idx:]
    print('----------------')
    # train_label_list_len = [len(x) for x in train_text_list]
    # print(Counter(train_label_list_len))
    # test_label_list_len = [len(x) for x in test_text_list]
    # print(Counter(test_label_list_len))

    print(len(label_list))
    print(len(text_list))
    '''
    train_text_list = []
    train_label_list = []
    test_text_list = []
    test_label_list = []
    for i in idx_list[:cut_idx]:
        train_text_list.append(text_list[i])
        train_label_list.append(label_list[i])
    for j in idx_list[cut_idx:]:
        test_text_list.append(text_list[j])
        test_label_list.append(label_list[j])
    '''
    return train_text_list, train_label_list, test_text_list, test_label_list

def upsample(list, times):
    tmp = []
    for _ in range(times):
        tmp += list
    return tmp

def loadData(file_path):
    print('preprocessing data by combine the mulit-line comments...')
    label_list = []
    text_list = []
    with open(file_path, 'r',encoding='utf-8') as f:
        f.readline()
        for line in tqdm(f):
            line = line.strip().split('\t\t\t')
            if len(line) == 1 and len(line[0]) <= 2:
                continue
            try:
                label = int(line[0])
                label_list.append(label)
                text = line[1]
                text_list.append(text)
            except (IndexError, ValueError):
                last_text = text_list.pop()
                last_text += line[0] # for here using line[0] since the unlabeled text has no label
                text_list.append(last_text)
            if len(text_list) != len(label_list):
                label_list.pop()
        return label_list, text_list