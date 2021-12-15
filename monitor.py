class MonitorValue(object):
    def __init__(self,
                 single_detect_monitor_result, single_detect_mean_result,
                 sequence_detect_day_result, sequence_detect_monitor_result,
                 moment_i, day_j,
                 index):
        monitor_single_monitor = single_detect_monitor_result[moment_i][day_j][index]
        monitor_sequence_day = sequence_detect_day_result[moment_i][day_j][index]
        monitor_mean = single_detect_mean_result[moment_i][day_j][index]

        start = 0
        end = 2

        if index == 0:
            end = 1
        elif index == 2:
            start = 1

        monitor_sequence_monitor = sequence_detect_monitor_result[moment_i][day_j][start] + \
                                   sequence_detect_monitor_result[moment_i][day_j][end]

        self.single_monitor = int(monitor_single_monitor)
        self.sequence_day = int(monitor_sequence_day)
        self.sequence_monitor = int(monitor_sequence_monitor)
        self.mean = int(monitor_mean)
