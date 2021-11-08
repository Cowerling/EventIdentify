import os
import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from PyNomaly import loop
from sklearn.cluster import KMeans
import math

root_dir = r'./data'

normal_data_file = os.path.join(root_dir, '正常数据.csv')
booster_data_file = os.path.join(root_dir, '爆管数据.csv')
error_data_file = os.path.join(root_dir, '错误数据.csv')
sample_data_file = os.path.join(root_dir, '样本数据.csv')
test_data_file = os.path.join(root_dir, '测试数据.csv')

interval = 3
outliers_count = 10
threshold = 0.2


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


def build_normal_depository(normal_data_file, interval):
    normal_depository = {}

    with open(normal_data_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader)
        next(reader)

        for row in reader:
            label = row[0]

            data = np.array([float(x) for x in row[1:]])
            data = data.reshape((-1, interval))

            sub_normal_depository = {}

            for index in range(0, interval):
                sub_data = data[:, index].reshape((-1, 1))

                sub_normal_depository[index] = sub_data

            normal_depository[label] = sub_normal_depository

    return normal_depository


def detect_event(data_file, iforests, normal_depository, threshold):
    data_count = 0
    total_error_count = 0

    with open(data_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader)
        next(reader)

        for row in reader:
            label = row[0]

            sub_data = np.array([float(x) for x in row[1:]])
            sub_data = sub_data.reshape((-1, interval + 1))

            y_label = sub_data[:, 0]
            y_data = sub_data[:, 1:]

            all_monitor_event_list = np.zeros(y_data.shape[0])

            for index in range(0, interval):
                sub_y_data = y_data[:, index].reshape((-1, 1))
                normal_data_list = normal_depository[label][index]
                iforest = iforests[label][index]

                single_monitor_event_list = []

                for i in range(0, sub_y_data.shape[0]):
                    single_data = np.expand_dims(sub_y_data[i], 0)
                    check_data_list = np.concatenate((normal_data_list, single_data))

                    iforest_predict = -iforest.score_samples(check_data_list)
                    iforest_predict = iforest_predict.reshape((-1, 1))

                    m = loop.LocalOutlierProbability(iforest_predict).fit()
                    loOp_scores = m.local_outlier_probabilities

                    k_means = KMeans(n_clusters=2)
                    k_means.fit(iforest_predict)

                    p = k_means.labels_ * loOp_scores
                    p_last_index = p.shape[0] - 1

                    if np.argmax(p) == p_last_index and p[p_last_index] > threshold:
                        single_monitor_event_list.append(1)
                    else:
                        single_monitor_event_list.append(0)

                        normal_data_list = check_data_list[1:].copy()
                        # normal_data_list = check_data_list.copy()

                all_monitor_event_list += np.array(single_monitor_event_list)

            all_monitor_event_list = all_monitor_event_list > 0
            all_monitor_event_list = all_monitor_event_list.astype(np.int32)
            error_count = np.sum((all_monitor_event_list - y_label) != 0)
            total_error_count += error_count
            data_count += y_data.shape[0]

            print('时刻: {} 预测: {} 实际: {} 错误率: {}'.format(label, all_monitor_event_list, y_label,
                                                                   error_count / y_label.shape[0]))

    total_error = total_error_count / data_count
    return total_error


def get_all_monitor_data(data_file, interval):
    data = []
    time_labels = []

    with open(data_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader)
        header = next(reader)

        day_count = int(len(header) / interval)
        monitors = header[1: interval + 1]

        for row in reader:
            time_labels.append(row[0])
            data.append([float(x) for x in row[1:]])

    return data, day_count, time_labels, monitors


def get_monitor_data(monitor, day, data, monitors):
    index = monitors.index(monitor)
    n_data = np.array(data)[:, day * len(monitors) + index: day * len(monitors) + index + 1]

    return n_data.copy()


normal_monitor_data, normal_day_count, normal_time_labels, monitors = get_all_monitor_data(normal_data_file, interval)
test_monitor_data, test_day_count, test_time_labels, _ = get_all_monitor_data(test_data_file, interval)

day = 0
moment_length = 5
moment_count = len(normal_time_labels)
back_days = 15

monitor = '169'
k = 3

sequence_detect_result = []

for moment in range(moment_length - 1, moment_count):
    moment_test_monitor_data = get_monitor_data(monitor, day,
                                                test_monitor_data, monitors)[moment + 1 - moment_length: moment]
    mean_test = np.mean(moment_test_monitor_data)
    std_test = np.std(moment_test_monitor_data)

    d_list = []

    for i in range(0, back_days):
        moment_normal_monitor_data = get_monitor_data(monitor, normal_day_count - 1 - i,
                                                      normal_monitor_data, monitors)[moment + 1 - moment_length: moment]
        mean_normal = np.mean(moment_normal_monitor_data)
        std_normal = np.std(moment_normal_monitor_data)

        m = np.dot(moment_test_monitor_data, moment_normal_monitor_data)

        d = math.sqrt(math.fabs(2 * moment_length * (1 - (m - moment_length * mean_test * mean_normal)
                                                     / moment_length / std_test / std_normal)))
        d_list.append(d)

    d_min = np.min(d_list)
    d_mean = np.mean(d_list)
    d_std = np.std(d_list)

    d_threshold = d_mean + k * d_std

    sequence_detect_result.append(int(d_min > d_threshold))

iforests = build_iforests(sample_data_file, interval, outliers_count)
normal_depository = build_normal_depository(normal_data_file, interval)

total_error = detect_event(error_data_file, iforests, normal_depository, threshold)

print('>' * 20)
print('阈值: {} 总错误率:{}'.format(threshold, total_error))
print('<' * 20)
