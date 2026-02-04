
import os
import toml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer as toklib
from tokenizers import ByteLevelBPETokenizer, pre_tokenizers

from helper.visualize import plot_loss
from helper.LogDataUtil import LogDataUtil
from contralog.data_loaders import LogDataset, LogDataset_collate


class EarlyStopping():
    """
    Early stopping utility to stop training when validation loss does not improve.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.last_was_best = True

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss:
            self.last_was_best = True
            if val_loss < self.best_loss - self.min_delta:
                self.counter = 0
            self.best_loss = val_loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            self.last_was_best = False


class Trainer():
    """
    Trainer class for ContraLog.

    Args:
        log_data_util: Utility object for accessing log data.
        anomaly_model: The model.
        train_conf_path (str): Path to training configuration tmol file.
        device (str): Device for training.
        n_workers (int): Number of data loader workers.
    """

    def __init__(self, log_data_util, anomaly_model,
                 train_conf_path: str,
                 device: str = 'cuda',
                 n_workers: int = 0):

        self.conf = toml.load(train_conf_path)
        if log_data_util is None:
            log_data_util = LogDataUtil(
                self.conf['Data']['data_path'], memmap=False)
        else:
            self.log_data_util = log_data_util

        self.anomaly_model = anomaly_model

        self.device = device
        self.anomaly_model.message_encoder.to(self.device)
        self.anomaly_model.sequence_encoder.to(self.device)

        self.train_loss_lst = []
        self.eval_loss_lst = []
        self.tmp_loss_lst = []

        self.early_stopping = EarlyStopping(patience=self.conf['EarlyStopping']['patience'],
                                            min_delta=self.conf['EarlyStopping']['min_delta'])

        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.params = list(self.anomaly_model.message_encoder.parameters(
        )) + list(self.anomaly_model.sequence_encoder.parameters())
        self.optimizer = AdamW(
            params=self.params, lr=self.conf['Train']['lr'], weight_decay=0.01)
        if self.conf['Misc']['warm_start']:
            # load exisiting opitimizer state
            self.optimizer.load_state_dict(torch.load(
                self.conf['Misc']['warm_start_model_path'] + 'optimizer.save'))
        max_sequ_len = self.anomaly_model.conf['max_sequ_len']
        self.n_mask = self.conf['Train']['n_mask']

        train_logs = log_data_util.get(
            subset='train', ravel=False, logs=True, length=False, labels=False)['logs']
        train_log_dataset = LogDataset(train_logs, max_sequ_len)

        n_workers = self.conf['Data']['n_workers']
        if n_workers > 0:
            persistent_workers, pin_memory = True, True
        else:
            persistent_workers, pin_memory = False, False
        self.train_data_loader = DataLoader(train_log_dataset, batch_size=self.conf['Train']['batch_size'],
                                            shuffle=True,
                                            num_workers=n_workers,
                                            collate_fn=LogDataset_collate, drop_last=True,
                                            persistent_workers=persistent_workers, pin_memory=pin_memory)

        eval_logs = log_data_util.get(
            subset='val', ravel=False, logs=True, length=False, labels=False)['logs']
        eval_log_dataset = LogDataset(eval_logs, max_sequ_len)
        self.eval_data_loader = DataLoader(eval_log_dataset, batch_size=self.conf['Train']['batch_size'],
                                           shuffle=False,
                                           num_workers=n_workers,
                                           collate_fn=LogDataset_collate, drop_last=True,
                                           persistent_workers=persistent_workers, pin_memory=pin_memory)

    def save(self):
        save_path = self.conf['Misc']['save_path'] + \
            self.conf['Misc']['run_name'] + '/'
        print('Best model so far, saving to ' + save_path)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        self.anomaly_model.save(save_path)
        if self.conf['Misc']['store_loss']:
            [np.array(l).mean()
             for l in self.train_loss_lst]
            train_loss_means = [np.mean(l) for l in self.train_loss_lst]
            eval_loss_means = [np.mean(l) for l in self.eval_loss_lst]
            np.savetxt(save_path + 'train_loss.txt',
                       train_loss_means, fmt='%.6f')
            np.savetxt(save_path + 'eval_loss.txt',
                       eval_loss_means, fmt='%.6f')
        if self.conf['Misc']['plot_loss']:
            plot_loss(self)
            plt.savefig(save_path + 'loss.png', dpi=300)
        torch.save(self.optimizer.state_dict(), save_path + 'optimizer.save')
        plt.close()

    def train(self):
        """
        Runs one training and one evaluation epoch.
        """
        n_max_epochs = self.conf['Train']['n_max_epochs']
        for epoch in range(n_max_epochs):
            print(f'---- Epoch: {epoch+1}/{n_max_epochs} ----')
            self.do_epoch(mode='train')
            with torch.no_grad():
                self.do_epoch(mode='eval')
            if self.early_stopping.early_stop:
                print(
                    f'Early stopping reached at epoch {epoch}/{n_max_epochs}')
                break
            elif self.early_stopping.last_was_best:
                # If the last epoch was the best one, save the model
                if self.conf['Misc']['save']:
                    self.save()

    def forward(self, logs, lengths):
        """
        Forward pass through the model, including random masking.
        This method is only for training, not inference.

        Args:
            logs (List[str]): Log messages in the batch.
            lengths (List[int]): Lengths of log sequences.

        Returns:
            pred (Tensor): Model predictions. SequenceEncoder outputs
            mask_mask (Tensor): Mask indicating the positions of masked tokens.
            targets (Tensor): Target Embedings. MessageEncouder outputs.
            pad_mask (Tensor): Mask for padded positions.
        """
        embs = self.anomaly_model.embed(logs, batch_size=1028)
        sequence_embs = torch.split(embs, lengths)

        # padding for variable length
        padded_embs, pad_mask = pad_embs(sequence_embs, lengths)

        # get mask representation
        mask_representation = self.anomaly_model.sequence_encoder.get_mask_token(
            norm=False)
        mask_mask = sample_mask_location_mat(
            lengths, self.n_mask, device=self.device)

        masked_embs, targets = mask_sequence(
            padded_embs, mask_mask, mask_representation.type(padded_embs.dtype))

        pred = self.anomaly_model.sequence_encoder.forward(
            masked_embs.transpose(0, 1), pad_mask).transpose(0, 1)
        return pred, mask_mask, targets, pad_mask

    def do_epoch(self, mode: str):
        """
        Runs a full training or evaluation epoch.

        Args:
            mode (str): Either 'train' or 'eval'.
        """
        if mode == 'train':
            dataloader = self.train_data_loader
            self.anomaly_model.message_encoder.train()
            self.anomaly_model.sequence_encoder.train()
        else:
            dataloader = self.eval_data_loader
            self.anomaly_model.message_encoder.eval()
            self.anomaly_model.sequence_encoder.eval()

        p_bar = tqdm(dataloader, leave=False, unit='batch')
        for _, (logs, lengths) in enumerate(p_bar):
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                pred, mask_mask, targets, pad_mask = self.forward(
                    logs, lengths)

                mask_mask = mask_mask.flatten().bool()
                pred = pred.flatten(0, 1)[mask_mask]

                # get target embeddings that are not real targets(masked) and not padding,
                # they are used as negative samples
                extra_mask = (~mask_mask.to(self.device) & ~
                              pad_mask.flatten().bool().to(self.device))
                extra_targets = targets.to(
                    self.device).flatten(0, 1)[extra_mask]

                targets = targets.to(self.device).flatten(0, 1)[mask_mask]
                targets = torch.cat([targets, extra_targets], dim=0)

                pred = F.normalize(pred, p=2, dim=1)
                targets = F.normalize(targets, p=2, dim=1)

                scores = torch.mm(pred, targets.transpose(
                    0, 1)) * 4  # tau
                labels = torch.tensor(range(len(scores)),
                                      dtype=torch.long, device=self.device)

                # sym loss with all generated embeddings
                loss = (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(
                    scores[0:len(pred), 0:len(pred)].transpose(0, 1), labels)) / 2
                # alternative loss formulations - remove extra targets first
                # classical loss
                # loss = self.cross_entropy_loss(scores, labels)
                # classical symmetric loss
                # loss = (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(scores.transpose(0, 1), labels)) / 2

                l_item = loss.item()
                p_bar_str = f'{mode} loss: {l_item:.3f}'
                p_bar.set_description(str(p_bar_str))
                self.tmp_loss_lst.append(l_item)

                if mode == 'train':
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.params, self.conf['Train']['max_grad_norm'])
                    self.optimizer.step()

        mean_loss = sum(self.tmp_loss_lst)/len(self.tmp_loss_lst)
        std_loss = torch.std(torch.tensor(self.tmp_loss_lst))
        if mode == 'eval':
            self.eval_loss_lst.append(self.tmp_loss_lst)
            print(f'{mode} loss --------> {mean_loss:.3f}±{std_loss:.3f}')
            self.early_stopping(mean_loss)
        else:
            self.train_loss_lst.append(self.tmp_loss_lst)
            print(f'{mode} loss -> {mean_loss:.3f}±{std_loss:.3f}')
        self.tmp_loss_lst = []

    def set_lr(self, lr):
        """
        Modify the learning rate for the optimizer.

        Args:
            lr (float): New learning rate.
        """
        for g in self.optimizer.param_groups:
            g['lr'] = lr


class Tokenizer:
    """
    Wrapper for a pre-fitted tokenizer.

    Args:
        tokenizer: A tokenizer instance.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str):
        tokenizer = cls(toklib.from_file(path))
        return tokenizer

    def __call__(self, text: list, device: str = 'cpu'):
        """
        Tokenizes input text and returns token IDs and attention masks.

        Args:
            text (list): List of input strings.
            device (str): Device to place the tensors on.

        Returns:
            token_ids (Tensor): Token IDs.
            attention_mask (Tensor): Attention mask indicating padding tokens.
        """
        encodings = self.tokenizer.encode_batch(text)
        token_ids = torch.tensor([enc.ids for enc in encodings], device=device)
        attention_mask = torch.tensor(
            [enc.attention_mask for enc in encodings], device=device)
        return token_ids, attention_mask

    def save_pretrained(self, path: str):
        self.tokenizer.save(path)


def make_new_tokenizer(max_fit_sample, log_data_util, model_conf_path):
    """
    Creates and fits a new ByteLevel BPE tokenizer from training log messages.

    Args:
        max_fit_sample (int): Maximum number of log samples to use for training.
        log_data_util: Utility object to access log datasets.
        model_conf_path (str): Path to the model configuration.
    Returns:
        Tokenizer: A wrapped and configured tokenizer instance.
    """
    def get_training_corpus(log_messages):
        for log_message in log_messages:
            yield log_message

    model_conf = toml.load(model_conf_path)
    n_tokens = model_conf['tokenizer_vocab_len']
    max_log_len = model_conf['max_log_len']
    tokenizer = ByteLevelBPETokenizer()
    tokenizer._tokenizer.model.unk_token = '[UNK]'
    tokenizer._tokenizer.model.fuse_unk = True
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Metaspace(), pre_tokenizers.Digits()])
    # Load all train logs, then randomly sample max_fit_sample
    all_log_messages = log_data_util.get(
        subset='train', ravel=True, logs=True, length=False, labels=False)['logs']
    if len(all_log_messages) > max_fit_sample:
        indices = np.random.choice(
            len(all_log_messages), max_fit_sample, replace=False)
        log_messages = [all_log_messages[i] for i in indices]
    else:
        log_messages = all_log_messages
    training_corpus = get_training_corpus(log_messages)
    tokenizer.train_from_iterator(
        training_corpus, vocab_size=n_tokens, min_frequency=2, special_tokens=['[PAD]', '[UNK]'])

    tokenizer.enable_padding(direction='right', pad_id=0,
                             pad_type_id=0, pad_token='[PAD]', length=None)
    tokenizer.enable_truncation(
        max_length=max_log_len, stride=0, strategy='longest_first')
    return Tokenizer(tokenizer)


def sample_mask_location_mat(lengths, n_mask, device='cpu'):
    """
    Creates a mask matrix where n_mask random positions are masked for each sequence.

    Args:
        lengths (list of int): List of sequence lengths for each batch.
        n_mask (float or int): Number of positions to mask. If float (0 < n_mask <= 1), it is treated as a fraction.
        device (str): Device for the output mask tensor.

    Returns:
        mask_mask (Tensor): Boolean mask of shape (batch_size, max_length) with True for masked positions.
    """
    max_l = max(lengths)
    batch_size = len(lengths)
    mask_mask = torch.zeros([batch_size, max_l],
                            device=device, dtype=torch.bool)
    # Generate random mask locations
    for ind, l in enumerate(lengths):
        if n_mask >= 1:
            sampled_n_mask = int(n_mask)
        else:
            sampled_n_mask = max(1, int(l * n_mask))
        mask_locations = torch.randperm(l, device=device)[:sampled_n_mask]
        mask_mask[ind, mask_locations] = 1
    return mask_mask.bool()


def pad_embs(emb_lst, lengths):
    """
    Pads a list of embedding tensors to the same length and generates a padding mask.

    Args:
        emb_lst (list of Tensors): List of tensors with varying sequence lengths.
        lengths (list of int, optional): List of sequence lengths. If None, lengths are inferred.

    Returns:
        padded_embs (Tensor): Padded tensor of shape (batch_size, max_seq_len, embedding_dim).
        pad_mask (Tensor): Binary mask of shape (batch_size, max_seq_len) with 0 for padding.
    """
    if lengths is None:
        lengths = [len(l) for l in emb_lst]
    padded_embs = pad_sequence(emb_lst, batch_first=True)

    pad_mask = torch.arange(padded_embs.shape[1], device=padded_embs.device).expand(
        len(lengths), -1) >= torch.tensor(lengths, device=padded_embs.device).unsqueeze(1)
    return padded_embs, pad_mask


def mask_sequence(padded_embs, mask_mask, mask_emb):
    """
    Replaces embeddings at masked positions with the mask embedding and stores originals as targets.

    Args:
        padded_embs (Tensor): Input tensor of shape.
        mask_mask (Tensor): Boolean mask indicating which positions to mask.
        mask_emb (Tensor): Embedding vector to insert at masked positions.

    Returns:
        masked_embs (Tensor): Tensor with masked positions replaced by mask_emb.
        targets (Tensor): Original unmasked embeddings.
    """
    masked_embs = padded_embs.clone()
    masked_embs[mask_mask] = mask_emb
    targets = padded_embs
    return masked_embs, targets
