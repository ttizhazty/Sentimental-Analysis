import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import tree
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import os
import pickle
from collections import Counter
# import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as RFC

data_path = './yelp_dataset/data_split/'
train_file_list = os.listdir(data_path + 'train/emb_feature/')
test_file_list = os.listdir(data_path + 'test/emb_feature')
train_feature = []
train_label = []
for i in range(len(train_file_list)//2):
    if i % 10 == 0:
        with open(data_path + 'train/emb_feature/train_feature_emb_%d' %i + '.train', 'rb') as f:
            train_feature += pickle.load(f)
        with open(data_path + 'train/emb_label/train_label_emb_%d' %i + '.train', 'rb') as l:
            train_label += pickle.load(l)

lr_train_label = []
for label in train_label:
    label = label.index(1)
    lr_train_label.append(label)

print(len(train_feature), len(train_label))
train_feature = np.array(train_feature).reshape(len(train_feature), -1)
train_label = np.array(train_label)
lr_train_label = np.array(lr_train_label)

print(train_feature.shape, train_label.shape)
# lr_model = LinearRegression()
# lr_model.fit(train_feature, lr_train_label)
# model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=7))
# clt = model.fit(train_feature, train_label)
# lr_model = tree.DecisionTreeClassifier(max_depth=40, min_weight_fraction_leaf=0.3)
# lr_model.fit(train_feature, lr_train_label)
lr_model = RFC()
lr_model.fit(train_feature, lr_train_label)

#Xgb model
# params = {
#     'booster': 'gbtree',
#     'objective': 'reg: linear',
#     'gamma': 0.1,
#     'max_depth': 7,
#     'subsample': 0.8,
#     'min_child_weight': 3,
#     'eta': 0.05,
#     'silent': 0,
#     'nthread': 7
# }

# plst = list(params.items())
# xgb_train = xgb.DMatrix(train_feature, label=lr_train_label)
# watchlist = [(xgb_train, 'train')]
# xgb_model = xgb.train(plst, xgb_train, 200, watchlist)

test_feature = []
test_label = []
sample_count = 0
sen_count = 0
sen_correct_count = 0
lr_hit = 0
for i in range(len(test_file_list)//2):
    if i % 10 == 0:
        with open(data_path + 'test/emb_feature/test_feature_emb_%d' %i + '.test', 'rb') as f:
            test_feature = pickle.load(f)
        with open(data_path + 'test/emb_label/test_label_emb_%d' %i + '.test', 'rb') as l:
            test_label = pickle.load(l)
        with open(data_path + 'test/emb_feature/sample_idx_%d' %i + '.test', 'rb') as f_idx:
            test_idx = pickle.load(f_idx)
            test_idx[-1] -= 1

        test_feature = np.array(test_feature).reshape(len(test_feature), -1)
        # xgb_test = xgb.DMatrix(test_feature)
        lr_pred = lr_model.predict(test_feature)

        for j in range(len(lr_pred)):
            if lr_pred[j] < 0.5:
                lr_pred[j] = 0
            elif 0.5 <= lr_pred[j] < 1.5:
                lr_pred[j] = 1
            else:
                lr_pred[j] = 2
        file_hit = 0
        for k in range(len(lr_pred)):
            if lr_pred[k] == test_label[k].index(1):
                file_hit += 1
        lr_hit += file_hit
        sample_count += len(lr_pred)
        file_acc = file_hit / len(lr_pred)
        print('file_%d acc = ' %i, file_acc)

        file_sen_hit = 0
        for e in range(1, len(test_idx)):
            # print(test_idx[e - 1])
            # print(test_idx[e])
            # print(test_label[test_idx[e - 1]: test_idx[e]][0])
            try:
                sen_label = test_label[test_idx[e - 1]: test_idx[e]][0].index(1)
                sen_pred = Counter(lr_pred[test_idx[e - 1]: test_idx[e]]).most_common(1)[0][0]
                if sen_label == sen_pred:
                    sen_correct_count += 1
                    file_sen_hit += 1
                sen_count += 1
            except IndexError:
                continue
        print('file_%d sen level acc = ' % i, file_sen_hit / len(test_idx))

print('lr test acc = ', lr_hit / sample_count)
print('lr sen level acc = ', sen_correct_count / sen_count)