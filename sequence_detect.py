import csv
import numpy as np
import math
import itertools


def get_all_monitor_data(data_file, interval, has_label):
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

    return data, day_count, time_labels, monitors, np.array(labels).astype(np.int32)


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
                                moment_length,
                                k):
    moment_count = len(time_labels)
    back_days = 15

    all_sequence_detect_result = []

    for day in range(0, test_day_count):
        sequence_detect_result = []

        for moment in range(moment_length - 1, moment_count):
            result = []

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

                for pair in itertools.combinations([x for x in range(0, back_days - 1)], 2):
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

                result.append(int(d_min > d_threshold))

            sequence_detect_result.append(result)

        all_sequence_detect_result.append(sequence_detect_result)

    all_sequence_detect_result = np.array(all_sequence_detect_result)
    all_sequence_detect_result = all_sequence_detect_result.transpose((1, 0, 2))
    all_sequence_detect_result = np.concatenate((np.zeros((moment_length - 1, all_sequence_detect_result.shape[1], len(monitors))),
                                                 all_sequence_detect_result),
                                                axis=0)

    return all_sequence_detect_result


def sequence_detect_between_monitor(normal_monitor_data, normal_day_count, test_monitor_data, test_day_count,
                                    time_labels, monitors,
                                    moment_length,
                                    k):
    moment_count = len(time_labels)
    back_days = 15
    monitor_pair_list = list(itertools.combinations(monitors, 2))

    all_sequence_detect_result = []

    for day in range(0, test_day_count):
        sequence_detect_result = []

        for moment in range(moment_length - 1, moment_count):
            result = []

            for monitor_pair in monitor_pair_list:
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

                result.append(int(d_test > d_threshold))

            sequence_detect_result.append(result)

            if len(result) == 0:
                for monitor in monitors:
                    normal_monitor_data[monitor][moment].append(test_monitor_data[monitor][moment][day])
                    normal_monitor_data[monitor][moment] = normal_monitor_data[monitor][moment][1:]

        all_sequence_detect_result.append(sequence_detect_result)

    all_sequence_detect_result = np.array(all_sequence_detect_result)
    all_sequence_detect_result = all_sequence_detect_result.transpose((1, 0, 2))
    all_sequence_detect_result = np.concatenate((np.zeros((moment_length - 1, all_sequence_detect_result.shape[1], len(monitor_pair_list))),
                                                all_sequence_detect_result),
                                                axis=0)

    return all_sequence_detect_result


def sequence_detect_event(normal_data_file, estimate_data_file, interval, has_label, moment_length, k):
    normal_monitor_data, normal_day_count, normal_time_labels, monitors, _ = get_all_monitor_data(normal_data_file,
                                                                                                  interval, False)
    estimate_monitor_data, estimate_day_count, estimate_time_labels, _, labels = get_all_monitor_data(estimate_data_file,
                                                                                                      interval,
                                                                                                      has_label)

    day_normal_monitor_data = normal_monitor_data.copy()
    day_estimate_monitor_data = estimate_monitor_data.copy()
    monitor_normal_monitor_data = normal_monitor_data.copy()
    monitor_estimate_monitor_data = estimate_monitor_data.copy()

    all_sequence_detect_day_result = sequence_detect_between_day(day_normal_monitor_data, normal_day_count,
                                                                 day_estimate_monitor_data, estimate_day_count,
                                                                 normal_time_labels, monitors, moment_length,
                                                                 k)

    all_sequence_detect_monitor_result = sequence_detect_between_monitor(monitor_normal_monitor_data, normal_day_count,
                                                                         monitor_estimate_monitor_data, estimate_day_count,
                                                                         normal_time_labels, monitors, moment_length,
                                                                         k)

    return all_sequence_detect_day_result, all_sequence_detect_monitor_result, labels
