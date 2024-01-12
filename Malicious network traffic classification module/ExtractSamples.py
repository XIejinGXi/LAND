import os
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
import numpy as np
import collections
from PIL import Image
import csv
import random
import pandas as pd

class extractSamples(Dataset):



    def __init__(self, mode, batchsz, n_way, k_shot, k_query, startidx=0):
        """

        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.startidx = startidx  # index label not from 0, but from startidx
        self.mode = mode
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (
        mode, batchsz, n_way, k_shot, k_query))


        csvdata = self.loadCSV('data/cic-ids2017-' + self.mode + '.csv')

        self.data = []
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)
        self.cls_num = len(self.data)





    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        :return: {label:[value1, value2 ...]}
        """

        dictLabels = {}
        data = pd.read_csv(csvf, header=None)

        value = torch.Tensor(np.array(preprocessing.scale(data.iloc[:, :-1])))
        label = np.array(data.iloc[:, -1:])
        label = list(map(int, label))
        # label = ' '.join(map(str, label.ravel().tolist()))
        for i in range(len(label)):
            l = label[i]
            row = value[i]
            # append filename to current label
            if l in dictLabels.keys():
                dictLabels[l].append(row)
            else:
                dictLabels[l] = [row]
        return dictLabels




    def __getitem__(self, index):

        selected_cls = np.random.choice(self.cls_num, self.n_way, False)

        support_x = torch.FloatTensor(self.setsz, 1, 1, 78)
        support_y = torch.LongTensor(self.setsz)
        query_x = torch.FloatTensor(self.querysz, 1, 1, 78)
        query_y = torch.LongTensor(self.querysz)

        label = 0
        k1 = k2 = 0
        for cls in selected_cls:
            selected_v_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
            indexDtrain = np.array(selected_v_idx[:self.k_shot])
            indexDtest = np.array(selected_v_idx[self.k_shot:])

            for i in range(self.k_shot):
                support_x[k1] = self.data[cls][indexDtrain[i]]# self.data[cls][indexDtrain[i]]
                support_y[k1] = label
                k1 = k1 + 1
            for j in range(self.k_query):
                query_x[k2] = self.data[cls][indexDtest[j]]
                query_y[k2] = label
                k2 = k2 + 1
            label = label + 1

        return support_x, support_y, query_x, query_y





    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


