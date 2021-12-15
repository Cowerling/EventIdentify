import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from PyNomaly import loop
from sklearn.cluster import KMeans


def build_iforests(sample_file, interval, outliers_count):
    iforests = {}

    with open(sample_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader)
        next(reader)[1:interval+1]

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


def detect_event(data_file, iforests, normal_depository, threshold, interval, k):
    with open(data_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        next(reader)
        next(reader)

        total_all_monitor_event_list = []
        total_all_mean_event_list = []

        for row in reader:
            label = row[0]

            sub_data = np.array([float(x) for x in row[1:]])
            sub_data = sub_data.reshape((-1, interval + 1))

            y_label = sub_data[:, 0]
            y_data = sub_data[:, 1:]

            all_monitor_event_list = []
            all_mean_event_list = []

            for index in range(0, interval):
                sub_y_data = y_data[:, index].reshape((-1, 1))
                normal_data_list = normal_depository[label][index]
                iforest = iforests[label][index]

                single_monitor_event_list = []
                single_mean_event_list = []

                for i in range(0, sub_y_data.shape[0]):
                    single_data = np.expand_dims(sub_y_data[i], 0)

                    normal_data_mean = np.mean(normal_data_list)
                    normal_data_std = np.std(normal_data_list)

                    if single_data < normal_data_mean - normal_data_std * k:
                        single_mean_event_list.append(-1)
                    elif single_data > normal_data_mean + normal_data_std * k:
                        single_mean_event_list.append(1)
                    else:
                        single_mean_event_list.append(0)

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

                single_monitor_event_list = np.array(single_monitor_event_list)
                single_monitor_event_list = np.expand_dims(single_monitor_event_list, axis=1)
                all_monitor_event_list.append(single_monitor_event_list)

                single_mean_event_list = np.array(single_mean_event_list)
                single_mean_event_list = np.expand_dims(single_mean_event_list, axis=1)
                all_mean_event_list.append(single_mean_event_list)

            all_monitor_event_list = np.concatenate(all_monitor_event_list, axis=1)
            all_mean_event_list = np.concatenate(all_mean_event_list, axis=1)

            total_all_monitor_event_list.append(all_monitor_event_list)
            total_all_mean_event_list.append(all_mean_event_list)

        total_all_monitor_event_list = np.array(total_all_monitor_event_list)

        total_all_mean_event_list = np.array(total_all_mean_event_list)

    return total_all_monitor_event_list, total_all_mean_event_list


def single_detect_event(sample_data_file, normal_data_file, estimate_data_file, interval, outliers_count, threshold, k):
    iforests = build_iforests(sample_data_file, interval, outliers_count)
    normal_depository = build_normal_depository(normal_data_file, interval)

    single_detect_monitor_result, single_detect_mean_result = detect_event(estimate_data_file, iforests,
                                                                           normal_depository, threshold, interval, k)

    return single_detect_monitor_result, single_detect_mean_result
