import numpy as np

from monitor import MonitorValue


def judge(day_count, moment_count, interval,
          single_detect_monitor_result, single_detect_mean_result,
          sequence_detect_day_result, sequence_detect_monitor_result,
          list_min_size, k1, k2, repair_size, rollback, under_mean_threshold, k3):
    all_result = []

    for day in range(0, day_count):
        start_doubt_moment = -1
        end_doubt_moment = -1
        doubt_mean = 0
        doubt_mean_list = []
        doubt_under_mean_count = 0

        result = np.zeros(moment_count).astype(np.int32)

        for moment in range(0, moment_count):
            if day == 5 and moment == 41:
                cc = 0

            sign = 0
            mean = 0
            mean_list = []
            under_mean_count = 0

            for monitor in range(0, interval):
                monitor_value = MonitorValue(single_detect_monitor_result, single_detect_mean_result,
                                             sequence_detect_day_result, sequence_detect_monitor_result,
                                             moment, day,
                                             monitor)

                sign += monitor_value.single_monitor + monitor_value.sequence_day + monitor_value.sequence_monitor
                mean += abs(monitor_value.mean)
                mean_list.append(monitor_value.mean)
                if monitor_value.mean == -1:
                    under_mean_count += 1

            if sign != 0 and start_doubt_moment == -1:
                start_doubt_moment = moment

            if sign == 0 and mean == 0 and start_doubt_moment != -1 and end_doubt_moment == -1:
                end_doubt_moment = moment - 1

            if start_doubt_moment != -1 and (sign != 0 or mean != 0):
                doubt_mean += mean
                doubt_mean_list.append(mean_list)
                doubt_under_mean_count += under_mean_count

            if start_doubt_moment != -1 and end_doubt_moment != -1:
                if end_doubt_moment - start_doubt_moment + 1 >= list_min_size:
                    sequence_day_0 = []
                    sequence_day_1 = []
                    sequence_day_2 = []

                    sequence_monitor_0 = []
                    sequence_monitor_1 = []
                    sequence_monitor_2 = []

                    for sub_moment in range(start_doubt_moment, end_doubt_moment + 1):
                        monitor_0_value = MonitorValue(single_detect_monitor_result, single_detect_mean_result,
                                                       sequence_detect_day_result, sequence_detect_monitor_result,
                                                       sub_moment, day,
                                                       0)
                        sequence_day_0.append(monitor_0_value.sequence_day)
                        sequence_monitor_0.append(monitor_0_value.sequence_monitor)

                        monitor_1_value = MonitorValue(single_detect_monitor_result, single_detect_mean_result,
                                                       sequence_detect_day_result, sequence_detect_monitor_result,
                                                       sub_moment, day,
                                                       1)
                        sequence_day_1.append(monitor_1_value.sequence_day)
                        sequence_monitor_1.append(monitor_1_value.sequence_monitor)

                        monitor_2_value = MonitorValue(single_detect_monitor_result, single_detect_mean_result,
                                                       sequence_detect_day_result, sequence_detect_monitor_result,
                                                       sub_moment, day,
                                                       2)
                        sequence_day_2.append(monitor_2_value.sequence_day)
                        sequence_monitor_2.append(monitor_2_value.sequence_monitor)

                    sequence_day_result_0 = 4
                    if len([x for x in sequence_day_0 if x == 1]) * 1.0 / len(sequence_day_0) > k1:
                        sequence_day_result_0 = 1
                    if len([x for x in sequence_day_0 if x == 0]) * 1.0 / len(sequence_day_0) > k2:
                        sequence_day_result_0 = 0

                    sequence_day_result_1 = 4
                    if len([x for x in sequence_day_1 if x == 1]) * 1.0 / len(sequence_day_1) > k1:
                        sequence_day_result_1 = 1
                    if len([x for x in sequence_day_1 if x == 0]) * 1.0 / len(sequence_day_1) > k2:
                        sequence_day_result_1 = 0

                    sequence_day_result_2 = 4
                    if len([x for x in sequence_day_2 if x == 1]) * 1.0 / len(sequence_day_2) > k1:
                        sequence_day_result_2 = 1
                    if len([x for x in sequence_day_2 if x == 0]) * 1.0 / len(sequence_day_2) > k2:
                        sequence_day_result_2 = 0

                    sequence_monitor_result = [sequence_monitor_0,
                                               sequence_monitor_1,
                                               sequence_monitor_2]
                    sequence_monitor_result_1 = len(
                        [x for x in sequence_monitor_result if
                         np.sum(np.array(x) == np.array([1 for _ in range(0, len(x))])) / len(x) > k3])
                    sequence_monitor_result_2 = len(
                        [x for x in sequence_monitor_result if
                         np.sum(np.array(x) == np.array([2 for _ in range(0, len(x))])) / len(x) > k3])

                    doubt_under_mean_rate = doubt_under_mean_count / (
                                end_doubt_moment + 1 - start_doubt_moment) / interval

                    if doubt_mean != 0:
                        if sequence_monitor_result_1 == 2 and sequence_monitor_result_2 == 1:
                            end_doubt_moment = end_doubt_moment - rollback

                            if end_doubt_moment >= start_doubt_moment:
                                result[start_doubt_moment: end_doubt_moment + 1] = 1
                        elif sequence_monitor_result_2 == 3 and sequence_day_result_0 + sequence_day_result_1 + sequence_day_result_2 != 3:
                            end_doubt_moment = end_doubt_moment - rollback

                            if end_doubt_moment >= start_doubt_moment:
                                result[start_doubt_moment: end_doubt_moment + 1] = 2
                        elif sequence_day_result_0 + sequence_day_result_1 + sequence_day_result_2 == 1 and \
                                doubt_under_mean_rate <= under_mean_threshold:
                            end_doubt_moment = end_doubt_moment - rollback

                            if end_doubt_moment >= start_doubt_moment:
                                result[start_doubt_moment: end_doubt_moment + 1] = 1
                        elif sequence_day_result_0 + sequence_day_result_1 + sequence_day_result_2 == 2:
                            end_doubt_moment = end_doubt_moment - rollback

                            if end_doubt_moment >= start_doubt_moment:
                                result[start_doubt_moment: end_doubt_moment + 1] = 2
                        elif sequence_day_result_0 + sequence_day_result_1 + sequence_day_result_2 == 3:
                            end_doubt_moment = end_doubt_moment - rollback

                            if end_doubt_moment >= start_doubt_moment:
                                result[start_doubt_moment: end_doubt_moment + 1] = 3
                        elif doubt_under_mean_rate > under_mean_threshold:
                            result[start_doubt_moment: end_doubt_moment + 1] = 4

                        remain_doubt_mean_list = []

                        for sub_doubt_mean in reversed(doubt_mean_list):
                            if np.sum(np.abs(sub_doubt_mean)) != 0:
                                remain_doubt_mean_list.insert(0, sub_doubt_mean)

                        fault_2 = len(remain_doubt_mean_list) > 0 and np.sum(remain_doubt_mean_list[0]) == -2

                        if fault_2:
                            for sub_remain_doubt_mean in remain_doubt_mean_list:
                                if sub_remain_doubt_mean != remain_doubt_mean_list[0]:
                                    fault_2 = False
                                    break

                        if fault_2:
                            result[start_doubt_moment: end_doubt_moment + 1] = 0
                            result[start_doubt_moment: start_doubt_moment + len(remain_doubt_mean_list)] = 2
                    else:
                        if sequence_monitor_result_2 == 3 and sequence_day_result_0 + sequence_day_result_1 + sequence_day_result_2 != 3:
                            end_doubt_moment = end_doubt_moment - rollback

                            if end_doubt_moment >= start_doubt_moment:
                                result[start_doubt_moment: end_doubt_moment + 1] = 2

                start_doubt_moment = -1
                end_doubt_moment = -1
                doubt_mean = 0
                doubt_mean_list = []
                doubt_under_mean_count = 0

        all_result.append(result)

    repair_result = all_result.copy()

    for day in range(0, day_count):
        start_repair_moment = -1
        end_repair_moment = -1
        start_repair_value = -1
        end_repair_value = -1

        for moment in range(0, moment_count):
            current_value = all_result[day][moment]

            if moment != 0 and current_value == 0 and start_repair_moment == -1:
                start_repair_moment = moment
                start_repair_value = all_result[day][moment - 1]

            if start_repair_moment != -1 and current_value != 0 and end_repair_moment == -1:
                end_repair_moment = moment - 1
                end_repair_value = current_value

            if start_repair_moment != -1 and end_repair_moment != -1 and start_repair_value != -1 and end_repair_value != -1:
                if end_repair_moment - start_repair_moment + 1 <= repair_size and start_repair_value == 4 and end_repair_value == 4:
                    repair_result[day][start_repair_moment: end_repair_moment + 1] = start_repair_value

                start_repair_moment = -1
                end_repair_moment = -1
                start_repair_value = -1
                end_repair_value = -1

    return np.array(repair_result).T
