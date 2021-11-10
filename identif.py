import os
import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from PyNomaly import loop
from sklearn.cluster import KMeans
import math
import itertools

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


def get_all_monitor_data(data_file, interval, has_label=False):
    data = {}
    time_labels = []
    labels = []

    offset = 0 if has_label is False else 1

    with open(data_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader)
        header = next(reader)

        day_count = int(len(header) / (interval + offset))
        monitors = header[1 + offset: interval + 1 + offset]

        for monitor in monitors:
            data[monitor] = []

        for row in reader:
            time_labels.append(row[0])

            sub_data = np.array([float(x) for x in row[1:]])
            sub_data = sub_data.reshape((-1, interval + offset))

            if has_label is True:
                labels.append(np.squeeze(sub_data[:, 0]).tolist())

            for index, monitor in enumerate(monitors):
                data[monitor].append(np.squeeze(sub_data[:, index + offset]).tolist())

    return data, day_count, time_labels, monitors, labels


def get_monitor_data(monitor, day, data):
    n_data = np.array(data[monitor])[:, day: day + 1]

    return n_data.copy()


def get_moment_monitor_data(monitor, day, data, start_moment, end_moment):
    moment_monitor_data = get_monitor_data(monitor, day, data)[start_moment: end_moment]
    moment_monitor_data = np.squeeze(moment_monitor_data)
    mean = np.mean(moment_monitor_data)
    std = np.std(moment_monitor_data)

    return moment_monitor_data, mean, std

def get_distance(first_moment_monitor_data, second_moment_monitor_data, first_mean, second_mean, first_std, second_std):
    length = first_moment_monitor_data.shape[0]
    m = np.dot(first_moment_monitor_data, second_moment_monitor_data)
    d = math.sqrt(math.fabs(2 * length * (1 - (m - length * first_mean * second_mean)
                                          / length / first_std / second_std)))

    return d


def sequence_detect_between_day(normal_monitor_data, normal_day_count, test_monitor_data, test_day_count,
                                time_labels, monitors,
                                moment_length):
    moment_count = len(time_labels)
    back_days = 15

    k = 3

    all_sequence_detect_result = []

    for day in range(0, test_day_count):
        sequence_detect_result = []

        for moment in range(moment_length - 1, moment_count):
            result = 0

            for monitor in monitors:
                moment_test_monitor_data, mean_test, std_test = get_moment_monitor_data(monitor,
                                                                                        day,
                                                                                        test_monitor_data,
                                                                                        moment + 1 - moment_length,
                                                                                        moment + 1)

                d_list = []
                moment_normal_monitor_data_list = []

                for i in range(0, back_days):
                    moment_normal_monitor_data, mean_normal, std_normal = get_moment_monitor_data(monitor,
                                                                                                  normal_day_count - 1 - i,
                                                                                                  normal_monitor_data,
                                                                                                  moment + 1 - moment_length,
                                                                                                  moment + 1)

                    moment_normal_monitor_data_list.append((moment_normal_monitor_data, mean_normal, std_normal))

                    d = get_distance(moment_test_monitor_data, moment_normal_monitor_data,
                                     mean_test, mean_normal, std_test, std_normal)
                    d_list.append(d)

                d_min = np.min(d_list)

                normal_d_list = []

                for pair in itertools.combinations([x for x in range(normal_day_count - back_days, normal_day_count)], 2):
                    first_day = pair[0]
                    second_day = pair[1]

                    first_moment_normal_monitor_data, first_mean_normal, first_std_normal = \
                    moment_normal_monitor_data_list[first_day]
                    second_moment_normal_monitor_data, second_mean_normal, second_std_normal = \
                    moment_normal_monitor_data_list[second_day]

                    normal_d = get_distance(first_moment_normal_monitor_data, second_moment_normal_monitor_data,
                                            first_mean_normal, second_mean_normal, first_std_normal, second_std_normal)
                    normal_d_list.append(normal_d)

                d_mean = np.mean(normal_d_list)
                d_std = np.std(normal_d_list)

                d_threshold = d_mean + k * d_std

                if d_min <= d_threshold:
                    normal_monitor_data[monitor][moment].append(moment_test_monitor_data[-1])
                    normal_monitor_data[monitor][moment] = normal_monitor_data[monitor][moment][1:]

                result += int(d_min > d_threshold)

            sequence_detect_result.append(result)

        all_sequence_detect_result.append(sequence_detect_result)

    all_sequence_detect_result = np.array(all_sequence_detect_result)
    all_sequence_detect_result = (all_sequence_detect_result.T > 0).astype(np.int32)

    return all_sequence_detect_result


def sequence_detect_between_monitor(normal_monitor_data, normal_day_count, test_monitor_data, test_day_count,
                                    time_labels, monitors,
                                    moment_length):
    moment_count = len(time_labels)
    back_days = 15

    k = 3

    all_sequence_detect_result = []

    for day in range(0, test_day_count):
        sequence_detect_result = []

        for moment in range(moment_length - 1, moment_count):
            result = 0

            for monitor_pair in itertools.combinations(monitors, 2):
                first_monitor = monitor_pair[0]
                second_monitor = monitor_pair[1]

                first_moment_test_monitor_data, first_mean_test, first_std_test = get_moment_monitor_data(first_monitor,
                                                                                                          day,
                                                                                                          test_monitor_data,
                                                                                                          moment + 1 - moment_length,
                                                                                                          moment + 1)

                second_moment_test_monitor_data, second_mean_test, second_std_test = get_moment_monitor_data(second_monitor,
                                                                                                             day,
                                                                                                             test_monitor_data,
                                                                                                             moment + 1 - moment_length,
                                                                                                             moment + 1)

                d_test = get_distance(first_moment_test_monitor_data, second_moment_test_monitor_data,
                                      first_mean_test, second_mean_test, first_std_test, second_std_test)

                d_normal_list = []

                for back_day in range(normal_day_count - back_days, normal_day_count):
                    first_moment_normal_monitor_data, first_mean_normal, first_std_normal = get_moment_monitor_data(
                        first_monitor,
                        back_day,
                        normal_monitor_data,
                        moment + 1 - moment_length,
                        moment + 1)

                    second_moment_normal_monitor_data, second_mean_normal, second_std_normal = get_moment_monitor_data(
                        second_monitor,
                        back_day,
                        normal_monitor_data,
                        moment + 1 - moment_length,
                        moment + 1)

                    d_normal = get_distance(first_moment_normal_monitor_data, second_moment_normal_monitor_data,
                                            first_mean_normal, second_mean_normal, first_std_normal, second_std_normal)

                    d_normal_list.append(d_normal)

                d_mean = np.mean(d_normal_list)
                d_std = np.std(d_normal_list)
                d_threshold = d_mean + k * d_std

                result += int(d_test > d_threshold)

            sequence_detect_result.append(result)

            if result == 0:
                for monitor in monitors:
                    normal_monitor_data[monitor][moment].append(test_monitor_data[monitor][moment][day])
                    normal_monitor_data[monitor][moment] = normal_monitor_data[monitor][moment][1:]

        all_sequence_detect_result.append(sequence_detect_result)

    all_sequence_detect_result = np.array(all_sequence_detect_result)
    all_sequence_detect_result = (all_sequence_detect_result.T > 0).astype(np.int32)

    return all_sequence_detect_result


normal_monitor_data, normal_day_count, normal_time_labels, monitors, _ = get_all_monitor_data(normal_data_file, interval)
test_monitor_data, test_day_count, test_time_labels, _, labels = get_all_monitor_data(booster_data_file, interval, True)

moment_length = 5

day_normal_monitor_data = normal_monitor_data.copy()
day_test_monitor_data = test_monitor_data.copy()

# all_sequence_detect_result = sequence_detect_between_day(day_normal_monitor_data, normal_day_count,
#                                                          day_test_monitor_data, test_day_count,
#                                                          normal_time_labels, monitors, moment_length)

monitor_normal_monitor_data = normal_monitor_data.copy()
monitor_test_monitor_data = test_monitor_data.copy()

all_sequence_detect_result = sequence_detect_between_monitor(monitor_normal_monitor_data, normal_day_count,
                                                             monitor_test_monitor_data, test_day_count,
                                                             normal_time_labels, monitors, moment_length)

print(all_sequence_detect_result)
print(all_sequence_detect_result.shape)

input()

iforests = build_iforests(sample_data_file, interval, outliers_count)
normal_depository = build_normal_depository(normal_data_file, interval)

total_error = detect_event(error_data_file, iforests, normal_depository, threshold)

print('>' * 20)
print('阈值: {} 总错误率:{}'.format(threshold, total_error))
print('<' * 20)
