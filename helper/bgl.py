import numpy as np
from tqdm import tqdm
import random


def get_label(log: str):
    if log.split(' ')[0] == '-':
        label = 0
    else:
        label = 1
    return label


def remove_label_and_time_node(log: str):
    split = log.split(' ')
    # remove node info: [3]
    return split[3] + ' ' + (' ').join(split[6::])


def load_bgl(path: str, window_size, max_samples, n=-1, shuffle: bool = False):
    # window_size in seconds
    print('loading data')
    log_data = open(path, 'r', encoding='utf-8')
    log_data = list(log_data)
    if n is None:
        n = -1
    n = min(n, len(log_data))
    log_data = log_data[0:n]

    def get_timestamp(log):
        return int(log.split()[1])

    timestamps = np.array(list(map(get_timestamp, log_data)))

    log_sessions = []
    log_session = []

    session_start_time = timestamps[0]

    for i in tqdm(range(len(timestamps))):
        current_time = timestamps[i]

        if (current_time - session_start_time > window_size) or (len(log_session) > max_samples):
            if len(log_session) > 0:
                log_sessions.append(log_session)
                log_session = []
            # reset the session start time
            session_start_time = current_time

        log_session.append(log_data[i])

    log_lst = []
    label_lst = []
    for sequence in tqdm(log_sessions):
        sequnce_labels = list(map(get_label, sequence))
        if 1 in sequnce_labels:
            label_lst.append(1)
        else:
            label_lst.append(0)

        sequence = list(map(remove_label_and_time_node, sequence))
        log_lst.append(sequence)

    if shuffle:
        c = list(zip(log_lst, label_lst))
        random.shuffle(c)
        log_lst, label_lst = zip(*c)

    log_lst = np.array(log_lst, dtype=object)
    label_lst = np.array(label_lst)

    return log_lst, label_lst
