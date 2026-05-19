import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from contralog.data_loaders import LogDataset, LogDataset_collate
from torch.utils.data import DataLoader
import torch.nn.functional as F


def plot_loss(trainer):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].plot(list(itertools.chain(*trainer.train_loss_lst)) +
               trainer.tmp_loss_lst)
    ax[0].set_xlabel('batch')
    ax[0].set_ylabel('loss')
    ax[1].plot([np.array(l).mean()
               for l in trainer.train_loss_lst], label='train')
    ax[1].plot([np.array(l).mean()
               for l in trainer.eval_loss_lst], label='eval')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend()
    plt.grid(True)


def get_sim_mat(trainer, subset: str = 'train', batch_size: int = 64):
    trainer.anomaly_model.message_encoder.eval()
    trainer.anomaly_model.sequence_encoder.eval()
    log_dataset = LogDataset(trainer.log_data_util.get(subset=subset, ravel=False, logs=True, length=False, labels=False, n=batch_size)['logs'],
                             max_sequ_len=trainer.anomaly_model.conf['max_sequ_len'])
    dl = DataLoader(log_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                    collate_fn=LogDataset_collate, drop_last=True)
    with torch.inference_mode():
        for logs, lengths in dl:
            pred, mask_mask, targets, pad_mask = trainer.forward(logs, lengths)
            targets = F.normalize(targets, p=2, dim=-1)

            mask_mask = mask_mask.flatten().bool()
            pred = pred.flatten(0, 1)[mask_mask]
            pred = F.normalize(pred, p=2, dim=1)

            targets = targets.to(trainer.device).flatten(0, 1)[mask_mask]

            scores = torch.mm(pred, targets.transpose(0, 1))
            break

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 5), sharex=True)
    ax[0].imshow(scores.detach().cpu().numpy())
    s = scores.detach().cpu().numpy()
    s = s[np.arange(len(s)), np.arange(len(s))]
    print('similarity score: ', np.mean(s))
    ax[1].plot(s)
    ax[1].set_ylim([-0.1, 1.1])


def tokenizer_plots(trainer, subset: str = 'val', n: int = 2_000, save_path: str = None):
    # Check tokenizer statistics
    print(
        f'vocab size: {len(trainer.anomaly_model.tokenizer.tokenizer.get_vocab())}')
    l_dict = trainer.log_data_util.get(
        subset=subset, ravel=True, logs=True, length=True, labels=False, n=n)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 2))
    ax[0].boxplot(l_dict['lengths'])
    ax[0].plot([0.5, 1.5], [trainer.anomaly_model.conf['max_sequ_len'],
               trainer.anomaly_model.conf['max_sequ_len']])
    ax[0].set_title('Sequence Length')

    tokens, attention_mask = trainer.anomaly_model.tokenizer(l_dict['logs'])
    n_tokens_per_log = attention_mask.sum(1)

    ax[1].boxplot(n_tokens_per_log, showfliers=False)
    ax[1].plot([0.5, 1.5], [trainer.anomaly_model.conf['max_log_len'],
               trainer.anomaly_model.conf['max_log_len']])
    ax[1].set_title('Log Length')
    if save_path is not None:
        plt.savefig(save_path+'_tokenizer_stats1.png',
                    dpi=300, bbox_inches='tight')
    print('average tokens per log:', np.array(n_tokens_per_log).mean(),
          'std:', np.array(n_tokens_per_log).std())
    print('longest tokens in dict:')
    print(sorted(list(trainer.anomaly_model.tokenizer.tokenizer.get_vocab(
    ).keys()), key=len, reverse=True)[0:3])
    # ---
    unique, counts = np.unique(
        np.array(list(itertools.chain(*tokens))), return_counts=True)
    counts, unique = zip(*sorted(zip(counts, unique)))
    print('unique tokens found in data sample:', len(unique))
    lst = [trainer.anomaly_model.tokenizer.tokenizer.decode(
        [unique[-i]]) for i in range(20)]
    print('most common tokens:')
    print(lst)
    plt.figure(figsize=(10, 1))
    plt.plot(sorted(list(
        counts)+(len(trainer.anomaly_model.tokenizer.tokenizer.get_vocab())-len(counts)) * [0]))
    plt.yscale('log')
    plt.xlabel('token ID')
    plt.ylabel('occurrences')
    if save_path is not None:
        print('saving tokenizer statistics figures to: ' + save_path+'...')
        plt.savefig(save_path+'_tokenizer_stats2.png',
                    dpi=300, bbox_inches='tight')


def _plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.detach().cpu().abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def plot_grad_flow(trainer, accumulation_steps: int = 20):
    dataloader = trainer.train_data_loader
    trainer.anomaly_model.message_encoder.train()
    trainer.anomaly_model.sequence_encoder.train()
    trainer.optimizer.zero_grad()
    p_bar = tqdm(dataloader)

    for i, (logs, lengths) in enumerate(p_bar):

        pred, mask_mask, targets, pad_mask = trainer.forward(logs, lengths)

        # ---
        mask_mask = mask_mask.flatten().bool()
        pred = pred.flatten(0, 1)[mask_mask]
        pred = F.normalize(pred, p=2, dim=1)

        # get target embeddings that are not real targets(masked) and not padding, they are used as negative samples
        extra_mask = (~mask_mask.to(trainer.device) & ~
                      pad_mask.flatten().bool().to(trainer.device))
        extra_targets = targets.to(trainer.device).flatten(0, 1)[extra_mask]
        targets = targets.to(trainer.device).flatten(0, 1)[mask_mask]
        targets = torch.cat([targets, extra_targets], dim=0)

        scores = torch.mm(pred, targets.transpose(0, 1))
        labels = torch.tensor(range(len(scores)),
                              dtype=torch.long, device=trainer.device)
        loss = trainer.cross_entropy_loss(scores, labels)
        trainer.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(trainer.params, 1.0)
        _plot_grad_flow(trainer.anomaly_model.named_parameters())
        if i > accumulation_steps:
            break
