import pandas as pd
from tqdm import tqdm
import numpy as np


def remove_datatime(log: str) -> str:
    log = log.split(' ')[3::]
    log = ' '.join(log)
    return log


def get_time_int(message):
    m = message.split(' ')[0:3]
    # no trailing zeros in hdfs
    m[2] = m[2].ljust(5, '0')
    return int(''.join(m))


def replace_blk_string(log: str) -> str:
    log_tmp = log.split(' ')
    blk_str = None
    for t in log_tmp:
        if 'blk_' in t:
            blk_str = t
            break
    log = log.replace(blk_str, 'myBlock')
    return log


def get_blk_string(log: str) -> str:
    log_tmp = log.split(' ')
    for t in log_tmp:
        if 'blk_' in t:
            return t


def load_hdfs(label_path: str,
              log_path: str,
              shuffle: bool = True,
              frac: float = 1.0,
              replace_blk: bool = False,
              sort_timestamps: bool = True):
    label_df = pd.read_csv(label_path)
    if shuffle:
        label_df = label_df.sample(frac=frac).reset_index(drop=True)

    all_block_IDs = label_df['BlockId']
    print('n label block IDs:', len(all_block_IDs))
    all_labels = label_df['Label']
    all_labels = [1 if label == 'Anomaly' else 0 for label in all_labels]
    del label_df

    # load all logs, but remove the date

    log_data = list(open(log_path, 'r'))
    if sort_timestamps:
        # Messages are not always sorted by timestamp.
        print('sorting timestamps')
        log_data, _ = zip(
            *sorted(zip(log_data, list(map(get_time_int, log_data)))))
    log_data = list(map(remove_datatime, log_data))
    print('n log lines:', len(log_data))

    # create a dict that contains all cleaned log texts
    log_dict = {}
    print('preprocessing data')
    for log in tqdm(log_data):
        blk_id = get_blk_string(log)
        if replace_blk:
            log = replace_blk_string(log)
        if blk_id in log_dict:
            log_dict[blk_id].append(log)
        else:
            log_dict[blk_id] = [log]
    print('n found block IDs:', len(log_dict.keys()))
    # assign block IDs in list their corresponding log (sequences)
    all_session_logs = []
    for block_ID in all_block_IDs:
        all_session_logs.append(log_dict[block_ID])
    del log_dict
    print(len(all_session_logs), 'sessions')

    # get length of all sessions
    all_session_len = [len(session) for session in all_session_logs]

    n_anomal_sessions = sum(all_labels)
    print(n_anomal_sessions, 'abnormal sessions')
    n_normal_sessions = len(all_labels) - n_anomal_sessions
    print(n_normal_sessions, 'normal sessions')

    all_block_IDs = np.array(all_block_IDs)
    all_session_logs = np.array(all_session_logs, dtype=object)
    all_session_len = np.array(all_session_len)
    all_labels = np.array(all_labels)

    return all_session_logs, all_session_len, all_labels
