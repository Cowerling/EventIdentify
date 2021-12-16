import os
import numpy as np
import sys

from single_detect import single_detect_event
from sequence_detect import sequence_detect_event
from judge import judge
from monitor import MonitorValue

np.set_printoptions(threshold=sys.maxsize)

root_dir = r'./data'

normal_data_file = os.path.join(root_dir, '正常数据2.csv')
sample_data_file = os.path.join(root_dir, '样本数据1.csv')

estimate_data_file = os.path.join(root_dir, '情况3.csv')

interval = 3
outliers_count = 10
threshold = 0.2
moment_length = 5
label_value = 2
list_min_size = 2
k1 = 0.85
k2 = 0.85
repair_size = 10
rollback = moment_length - 1
under_mean_threshold = 0.8
k3 = 0.85

single_detect_monitor_result, single_detect_mean_result = single_detect_event(sample_data_file,
                                                                              normal_data_file,
                                                                              estimate_data_file,
                                                                              interval, outliers_count, threshold, 3)

sequence_detect_day_result, sequence_detect_monitor_result, labels = sequence_detect_event(normal_data_file,
                                                                                           estimate_data_file,
                                                                                           interval, True, moment_length, 1)

label_count = np.sum(labels)
labels[labels == 1] = label_value

moment_count = single_detect_monitor_result.shape[0]
day_count = single_detect_monitor_result.shape[1]

result = judge(day_count, moment_count, interval,
               single_detect_monitor_result, single_detect_mean_result,
               sequence_detect_day_result, sequence_detect_monitor_result,
               list_min_size, k1, k2, repair_size, rollback, under_mean_threshold, k3)

contrast = result - labels

total_count = contrast.shape[0] * contrast.shape[1]
correct_count = np.sum(contrast == 0)
omit_count = np.sum(contrast == (-labels)) - np.sum(labels == 0)
omit_count = 0 if omit_count < 0 else omit_count
error_count = total_count - correct_count - omit_count

print(result)
print('accuracy: {} {}, error: {} {}, omit: {} {}'
      .format(correct_count, correct_count / total_count,
              error_count, error_count / total_count,
              omit_count, omit_count / label_count))

result_output_file = estimate_data_file.replace('.csv', '_result.txt')
label_output_file = estimate_data_file.replace('.csv', '_label.txt')
output_file = estimate_data_file.replace('.csv', '.txt')

with open(result_output_file, 'w', encoding='utf-8') as file:
    file.write('{}'.format(result))
    file.write('accuracy: {}, error: {}, omit: {}\n'
               .format(correct_count / total_count, error_count / total_count, omit_count / label_count))

with open(label_output_file, 'w', encoding='utf-8') as file:
    file.write('{}'.format(labels))

with open(output_file, 'w', encoding='utf-8') as file:
    for moment in range(0, moment_count):
        print_line = '{'

        for day in range(0, day_count):
            print_line += '['

            for monitor in range(0, interval):
                monitor_value = MonitorValue(single_detect_monitor_result, single_detect_mean_result,
                                             sequence_detect_day_result, sequence_detect_monitor_result,
                                             moment, day,
                                             monitor)

                print_line += '({}, {}, {}, {}), '.format(monitor_value.single_monitor,
                                                          monitor_value.sequence_day,
                                                          monitor_value.sequence_monitor,
                                                          monitor_value.mean)

            print_line = print_line[0:-2] + '], '

        print_line = print_line[0:-2] + '}'
        file.write(print_line + '\n')
