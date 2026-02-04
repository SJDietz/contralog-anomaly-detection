
import numpy as np
from tqdm import tqdm
import random


def get_components(log: str):
    log_split = log.split(' ')
    if len(log_split) > 7:
        # label
        if log_split[0] != '-':
            label = 1
        else:
            label = 0
        # tiestamp
        timestamp = int(log_split[1])
        # message
        message = (' ').join(log_split[8::])
        message = message.replace('\n', '')
    else:
        print(f'error when parsing: {log}')
        return 0, 'None', 'None'

    return label, timestamp, message


def load_tbird(path, window_size, max_samples: int = None, n: int = None, shuffle: bool = True):
    print('loading raw data ...', end='\r')
    log_data = open(path, 'r', encoding='cp1252')
    log_data = list(log_data)[:-1]  # cut last message "- 114"
    print(f'extracting sessiond for {n}/{len(log_data)} log messages')

    if n is not None:
        log_data = log_data[0:n]
    if max_samples is None:
        max_samples = np.inf

    log_sessions_lst = []
    log_session = []
    session_labels_lst = []
    session_label = 0

    start_time_session = get_components(log_data[0])[0]

    for log in tqdm(log_data, total=len(log_data)):
        label, timestamp, message = get_components(log)
        if len(message) > 0:
            if (((timestamp - start_time_session)//window_size) > 0 or (len(log_session) >= max_samples)):
                if len(log_session) > 0:
                    # save current session
                    log_sessions_lst.append(log_session)
                    session_labels_lst.append(session_label)
                    # start new session
                    log_session = []
                    session_label = 0
                    start_time_session = timestamp

            # add current message and label to curren session
            log_session.append(message)
            if label == 1:
                session_label = 1

    if shuffle:
        c = list(zip(log_sessions_lst, session_labels_lst))
        random.shuffle(c)
        log_sessions_lst, session_labels_lst = zip(*c)

    log_sessions_lst = np.array(log_sessions_lst, dtype='object')
    session_labels_lst = np.array(session_labels_lst)

    return log_sessions_lst, session_labels_lst
