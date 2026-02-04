import os
import pickle
import numpy as np
import itertools


def save(data, path, memmap: bool = True):
    if memmap:
        f = np.memmap(path, dtype='object', mode='w+', shape=data.shape)
        f[:] = data[:]
        del f
    else:
        np.save(path, data)


class LogDataUtil():
    def __init__(self, data_path, memmap: bool = True):
        self.data_path = data_path
        self.memmap = memmap

    def get(self, subset: str, ravel: bool, logs: bool, length: bool, labels: bool, n: int = None):
        path = self.data_path
        if subset not in ('train', 'val', 'normal_fit', 'normal_test', 'not_normal_test'):
            raise Exception(f'invalid subset type {subset}')
        path = path + subset + '.npy'

        results = {}

        print('loading from', path)
        if self.memmap:
            # r for just read, c for copy
            log_sequences = np.memmap(path, dtype='object', mode='r')
        else:
            log_sequences = np.load(path, allow_pickle=True)
        if n is not None:
            log_sequences = log_sequences[0:n]

        if length:
            log_lengths = [len(l) for l in log_sequences]
            results['lengths'] = log_lengths
        if labels:
            if subset in ('train', 'val', 'normal_fit', 'normal_test'):
                log_labels = [0] * len(log_sequences)
            else:
                log_labels = [1] * len(log_sequences)
            results['labels'] = log_labels
        if logs:
            if ravel:
                log_sequences = list(itertools.chain(*log_sequences))
            results['logs'] = log_sequences

        return results


def save_log_data(all_session_logs, all_labels, data_path,
                  train_frac: float,
                  val_frac: float,
                  fit_frac: float,
                  test_frac: float,
                  balance: bool = True, memmap: bool = True):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created folder: {data_path}")
    else:
        print(f"{data_path} folder already exists.")

    n_all = len(all_labels)
    n_saved = 0

    # normalize fractions
    fraction_sum = train_frac + val_frac + fit_frac + test_frac
    train_frac /= fraction_sum
    val_frac /= fraction_sum
    fit_frac /= fraction_sum
    test_frac /= fraction_sum

    # fit set is optional
    normal_fit = None

    train_cutoff = int(n_all * train_frac)
    # get normal sequences in train split
    mask = all_labels[:train_cutoff] == 0
    train = all_session_logs[:train_cutoff][mask]
    save(train, data_path + 'train', memmap)
    print(len(train), 'normal train sequences')
    n_saved += len(train)

    val_cutoff = train_cutoff + int(n_all * val_frac)
    # get normal sequences in val split
    mask = all_labels[train_cutoff:val_cutoff] == 0
    val = all_session_logs[train_cutoff:val_cutoff][mask]
    save(val, data_path + 'val', memmap)
    print(len(val), 'normal val sequences')
    n_saved += len(val)

    if fit_frac > 0:
        fit_cutoff = val_cutoff + int(n_all * fit_frac)
        # get normal sequences in fit split
        mask = all_labels[val_cutoff:fit_cutoff] == 0
        normal_fit = all_session_logs[val_cutoff:fit_cutoff][mask]

        save(normal_fit, data_path + 'normal_fit', memmap)
        print(len(normal_fit), 'normal fit sequences')
        n_saved += len(normal_fit)
    else:
        fit_cutoff = val_cutoff

    test_cutoff = val_cutoff + int(n_all * test_frac)
    # get normal and not normal sequences in test split
    mask = all_labels[fit_cutoff:test_cutoff] == 0
    normal_test = all_session_logs[fit_cutoff:test_cutoff][mask]
    not_normal_test = all_session_logs[fit_cutoff:test_cutoff][~mask]
    if balance:
        # shuffle in split
        # this ensures the test split is representative when loaded partially
        np.random.shuffle(normal_test)
        np.random.shuffle(not_normal_test)
        n_fit = min(len(normal_test), len(not_normal_test))
        normal_test = normal_test[:n_fit]
        not_normal_test = not_normal_test[:n_fit]
    save(normal_test, data_path + 'normal_test', memmap)
    save(not_normal_test, data_path + 'not_normal_test', memmap)
    print(len(normal_test), 'normal test sequences, ', len(
        not_normal_test), 'not normal test sequences')
    n_saved += len(normal_test) + len(not_normal_test)

    print(f'Saved {n_saved}/{n_all} sequences to {data_path}')
