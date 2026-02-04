import torch
import itertools


class LogDataset(torch.utils.data.Dataset):
    """A dataset for loading log seqeunces.
    """

    def __init__(self, log_lst, max_sequ_len):
        self.log_lst = log_lst
        self.max_sequ_len = max_sequ_len

    def __getitem__(self, index):
        logs = self.log_lst[index][0:self.max_sequ_len]
        return logs, len(logs)

    def __len__(self):
        return len(self.log_lst)


def LogDataset_collate(data):
    log_sequences, lengths = zip(*data)

    log_sequences = list(itertools.chain(*log_sequences))
    return log_sequences, lengths
