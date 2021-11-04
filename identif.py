import os
import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from PyNomaly import loop
from sklearn.cluster import KMeans

root_dir = r'/home/cowerling/文档/管网异常识别数据'

normal_data_file = os.path.join(root_dir, '正常数据.csv')
booster_data_file = os.path.join(root_dir, '爆管数据.csv')
error_data_file = os.path.join(root_dir, '错误数据.csv')
sample_data_file = os.path.join(root_dir, '样本数据.csv')

interval = 3
outliers_count = 10


def build_iforests(sample_file, interval, outliers_count):
    iforests = {}

    with open(sample_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader)
        header = next(reader)[1:interval+1]

        rows = list(reader)

        for i in tqdm(range(len(rows)), colour='white'):
            row = rows[i]

            label = row[0]
            sub_data = np.array([float(x) for x in row[1:]])
            x_data = sub_data.reshape((-1, interval))

            sample_count = x_data.shape[0]
            outliers_fraction = outliers_count * 1.0 / sample_count

            sub_iforests = {}

            for index in range(0, interval):
                sub_x_data = x_data[:, index].reshape((-1, 1))

                iforest = IsolationForest(max_samples=sub_x_data.shape[0],
                                          random_state=np.random.RandomState(42), contamination=outliers_fraction)
                iforest.fit(sub_x_data)

                sub_iforests[index] = iforest

            iforests[label] = sub_iforests

    print('build isolation forests completed.')

    return iforests


iforests = build_iforests(sample_data_file, interval, outliers_count)

with open(booster_data_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)

    next(reader)
    next(reader)

    for row in reader:
        label = row[0]
        iforest = iforests[label]

        sub_data = np.array([float(x) for x in row[1:]])
        sub_data = sub_data.reshape((-1, interval + 1))

        y_label = sub_data[:, 0]
        y_data = sub_data[:, 1:]

        for index in range(0, interval):
            sub_y_data = y_data[:, index].reshape((-1, 1))

            iforest_predict = -iforest[index].score_samples(sub_y_data)
            iforest_predict = iforest_predict.reshape((-1, 1))

            m = loop.LocalOutlierProbability(iforest_predict).fit()
            loOp_scores = m.local_outlier_probabilities

            k_means = KMeans(n_clusters=2)
            k_means.fit(iforest_predict)

            p = k_means.labels_ * loOp_scores

            print(p)
            print(y_label)
