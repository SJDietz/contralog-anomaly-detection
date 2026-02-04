from contralog.inference_scripts import get_point_anomaly_scores, get_contextual_anomaly_scores, PointAnomalyDetector
from contralog.trainer import Trainer, make_new_tokenizer
from contralog.log_embedder import LogEmbedder
from contralog.models import AnomalyModel
from helper.LogDataUtil import LogDataUtil, save_log_data
from helper.visualize import tokenizer_plots
from helper.tbird import load_tbird
from helper.hdfs import load_hdfs
from helper.bgl import load_bgl
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import seaborn as sns
from itertools import combinations
import argparse
from datetime import datetime as dt
import numpy as np
import itertools
import random
import torch
import toml
import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = '16'
# ------------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# -------

def load_test_data(log_data_util, train_conf):
    """Load and prepare all test data subsets."""
    n_point_anomaly = train_conf['Test']['max_point_anomaly_ref_samples']
    n_fit = train_conf['Test']['max_threshold_fit_samples']
    n_test = train_conf['Test']['max_test_samples']
    
    normal_fit = log_data_util.get(
        subset='normal_fit', ravel=False, logs=True, length=False, 
        labels=False, n=n_fit)['logs']
    
    # Load training data for point anomaly reference
    normal_train = log_data_util.get(
        subset='train', ravel=False, logs=True, length=False, 
        labels=False, n=-1)['logs']
    
    if len(normal_train) > n_point_anomaly:
        normal_train = np.random.choice(
            normal_train, n_point_anomaly, replace=False)
    
    normal_test = log_data_util.get(
        subset='normal_test', ravel=False, logs=True, length=False, 
        labels=False, n=n_test)['logs']
    not_normal_test = log_data_util.get(
        subset='not_normal_test', ravel=False, logs=True, length=False, 
        labels=False, n=n_test)['logs']
    
    # Balance test set if needed
    if train_conf['Test']['balance_test']:
        cutoff = min(len(normal_test), len(not_normal_test))
        normal_test = normal_test[:cutoff]
        not_normal_test = not_normal_test[:cutoff]
    
    return {
        'normal_fit': normal_fit,
        'normal_train': normal_train,
        'normal_test': normal_test,
        'not_normal_test': not_normal_test
    }


def compute_fit_scores(point_anomaly_detector, log_embedder, normal_fit):
    """Compute anomaly scores for the fit dataset."""
    print('fit data 0/3', end='\r')
    
    print('fit data 1/3', end='\r')
    point_scores = get_point_anomaly_scores(
        point_anomaly_detector=point_anomaly_detector,
        log_embedder=log_embedder,
        log_sequences=normal_fit
    )
    
    print('fit data 2/3', end='\r')
    contextual_scores = get_contextual_anomaly_scores(
        log_embedder=log_embedder,
        log_sequences=normal_fit
    )
    
    print('fit data 3/3')
    return {
        'normal_point_anomaly_scores': point_scores,
        'normal_contextual_anomaly_scores': contextual_scores,
    }


def compute_test_scores(point_anomaly_detector, log_embedder, normal_test, not_normal_test):
    """Compute anomaly scores for the test dataset."""
    print('test data 0/4', end='\r')
    point_scores_normal = get_point_anomaly_scores(
        point_anomaly_detector=point_anomaly_detector,
        log_embedder=log_embedder,
        log_sequences=normal_test
    )
    
    print('test data 1/4', end='\r')
    contextual_scores_normal = get_contextual_anomaly_scores(
        log_embedder=log_embedder,
        log_sequences=normal_test
    )
    
    print('test data 2/4', end='\r')
    point_scores_abnormal = get_point_anomaly_scores(
        point_anomaly_detector=point_anomaly_detector,
        log_embedder=log_embedder,
        log_sequences=not_normal_test
    )
    
    print('test data 3/4', end='\r')
    contextual_scores_abnormal = get_contextual_anomaly_scores(
        log_embedder=log_embedder,
        log_sequences=not_normal_test
    )
    
    print('test data 4/4')
    return {
        'normal_point_anomaly_scores': point_scores_normal,
        'anormal_point_anomaly_scores': point_scores_abnormal,
        'normal_contextual_anomaly_scores': contextual_scores_normal,
        'anormal_contextual_anomaly_scores': contextual_scores_abnormal
    }


def get_features(point_scores, context_scores):
    """Extract feature vector from anomaly scores."""
    X = []
    for point, context in zip(point_scores, context_scores):
        X.append([
            np.mean(point), 
            point.max(),
            np.mean(context), 
            context.max(), #can add more features here
        ])
    return np.array(X)


def get_features_and_labels(precomputed_scores, only_normal=False, feature_func=None):
    """Extract features and labels from precomputed anomaly scores."""
    if feature_func is None:
        feature_func = get_features
        
    X = feature_func(
        precomputed_scores['normal_point_anomaly_scores'],
        precomputed_scores['normal_contextual_anomaly_scores']
    )
    y = np.zeros(X.shape[0])
    
    if only_normal:
        return X, None
    
    X_abnormal = feature_func(
        precomputed_scores['anormal_point_anomaly_scores'],
        precomputed_scores['anormal_contextual_anomaly_scores']
    )
    X = np.concatenate((X, X_abnormal), axis=0)
    y = np.concatenate((y, np.ones(X_abnormal.shape[0])), axis=0)
    
    return X, y


def compute_robust_z_scores(X, med, mad):
    """Compute robust z-scores using median and MAD."""
    mad_safe = mad.copy()
    mad_safe[mad_safe == 0] = 1e-9
    return np.abs((X - med) / mad_safe)


def evaluate_feature_combinations(X_fit, X_test, y_test, feature_names, percentile_threshold):
    """Evaluate all combinations of features and print results."""
    for num_features in range(1, len(feature_names) + 1):
        for feature_indices in combinations(range(len(feature_names)), num_features):
            selected_features = [feature_names[i] for i in feature_indices]
            print(', '.join(selected_features), end=', ')
            
            X_fit_selected = X_fit[:, feature_indices]
            med = np.median(X_fit_selected, axis=0)
            mad = np.median(np.abs(X_fit_selected - med), axis=0)
            
            rz = compute_robust_z_scores(X_fit_selected, med, mad)
            score = np.linalg.norm(rz, axis=1, ord=2)
            thr = np.percentile(score, percentile_threshold)
            
            X_test_selected = X_test[:, feature_indices]
            rz = compute_robust_z_scores(X_test_selected, med, mad)
            score_test = np.linalg.norm(rz, axis=1, ord=2)
            y_pred = (score_test > thr).astype(int)
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            f1 = report['1.0']['f1-score'] * 100
            precision = report['1.0']['precision'] * 100
            recall = report['1.0']['recall'] * 100
            print(f"---- F1: {f1:.3f}% ----- Precision: {precision:.3f}%, Recall: {recall:.3f}%")
    
    return y_pred, rz


def plot_f1_by_sequence_length(y_test, y_pred, precomputed_scores_test, dataset, model_path):
    """Plot F1 score vs sequence length."""
    if dataset == 'HDFS':
        bin_edges = [0, 10, 20, 30, 256]
        bin_labels = ['1-10', '11-20', '21-30', '>30']
    else:
        bin_edges = [0, 64, 128, 192, 256]
        bin_labels = ['1-64', '65-128', '129-192', '>192']
    
    # Get sequence lengths
    length = np.array([
        len(s) for s in 
        precomputed_scores_test['normal_point_anomaly_scores'] +
        precomputed_scores_test['anormal_point_anomaly_scores']
    ])
    
    f1_scores = []
    sample_counts = []
    for i in range(len(bin_edges) - 1):
        mask = (length >= bin_edges[i]) & (length < bin_edges[i+1])
        if mask.sum() > 0:
            f1 = f1_score(
                y_test[mask], y_pred[mask],
                zero_division=0, average='binary', pos_label=1
            )
            f1_scores.append(f1)
            sample_counts.append(mask.sum())
        else:
            f1_scores.append(0)
            sample_counts.append(0)
    
    plt.figure(figsize=(3, 3))
    plt.bar(bin_labels, f1_scores, alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('F1 Score')
    plt.title(dataset)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_path, f'{dataset}_sequence_length.pdf'),
        dpi=300, bbox_inches='tight'
    )


def plot_feature_contributions(rz, y_test, feature_names, dataset, model_path):
    """Plot feature contribution analysis."""
    #boxplot version
    plt.figure(figsize=(3, 3))
    tmp = rz[y_test == 1]
    tmp = tmp / tmp.sum(1)[:, None]
    plt.boxplot(
        tmp, tick_labels=feature_names,
        flierprops=dict(marker='o', color='red', markersize=4, alpha=0.8, rasterized=True)
    )
    plt.xticks(rotation=45)
    plt.ylabel('Score Fraction')
    plt.title(dataset)
    plt.ylim(-0.03, 1.03)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_path, f'{dataset}_score_fractions.pdf'),
        dpi=300, bbox_inches='tight'
    )
    
    #violin plot version
    plt.figure(figsize=(3, 3))
    tmp_df = pd.DataFrame(tmp, columns=feature_names)
    sns.violinplot(
        data=tmp_df, density_norm='width', inner='quartile',
        linewidth=0.8, cut=0, fill=False, color="black",
        inner_kws={'color': 'C1'}
    )
    plt.xticks(rotation=45)
    plt.ylabel('Score Fraction')
    plt.title(dataset)
    plt.ylim(-0.03, 1.03)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(
        os.path.join(model_path, f'{dataset}_score_fractions_violin.pdf'),
        dpi=300, bbox_inches='tight'
    )


def plot_threshold_sensitivity(X_fit, X_test, y_test, dataset, model_path):
    """Plot F1 score vs threshold percentile."""
    mask = [True, True, True, True]  #Use all features
    score_lst = []
    space = np.linspace(80, 100, 1000)
    
    for p in space:
        med = np.median(X_fit, axis=0)
        mad = np.median(np.abs(X_fit - med), axis=0)
        
        rz = compute_robust_z_scores(X_fit, med, mad)
        score = np.linalg.norm(rz[:, mask], axis=1, ord=2)
        thr = np.percentile(score, p)
        
        rz_test = compute_robust_z_scores(X_test, med, mad)
        score_test = np.linalg.norm(rz_test[:, mask], axis=1, ord=2)
        y_pred = (score_test > thr).astype(int)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1.0']['f1-score'] * 100
        score_lst.append(f1)
    
    plt.figure(figsize=(3, 3))
    plt.grid(True)
    plt.plot(space, score_lst)
    plt.xlabel('Percentile Threshold')
    plt.ylabel('F1 Score')
    if dataset is not None:
        plt.title(dataset)
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, f'{dataset}_f1_vs_threshold.pdf'))
    
    # Print best threshold
    max_score_idx = np.argmax(score_lst)
    highest_score = score_lst[max_score_idx]
    threshold_with_highest_score = space[max_score_idx]
    print(f"best threshold: {threshold_with_highest_score:.2f}")
    print(f"best score: {highest_score:.2f}")

def test(main_conf, dataset=None):
    """
    Main test function for evaluating anomaly detection model.
    
    Args:
        main_conf: Configuration dictionary from main.toml
        dataset: Name of the dataset being evaluated (e.g., 'HDFS', 'BGL')
    """
    # Load configuration and setup device
    train_conf = toml.load(main_conf['train_conf'])
    device = train_conf['Misc']['device'] if train_conf['Misc']['device'] != '' else \
             ('cuda' if torch.cuda.device_count() else 'cpu')
    print('device:', device)

    # Model and data
    data_path = train_conf['Data']['data_path']
    log_data_util = LogDataUtil(data_path, memmap=False)
    
    model_path = train_conf['Test']['model_path']
    print('loading pretrained model:', model_path)
    anomaly_model = AnomalyModel.from_pretrained(model_path, device=device)

    n_params_msg = sum(p.numel() for p in anomaly_model.message_encoder.parameters())
    n_params_seq = sum(p.numel() for p in anomaly_model.sequence_encoder.parameters())
    print('message_encoder parameter count:', n_params_msg)
    print('sequence_encoder parameter count:', n_params_seq)
    print('current configuration:')
    print(anomaly_model.conf, train_conf, end='\n\n')

    data_subsets = load_test_data(log_data_util, train_conf)

    # Compute anomaly scores
    anomaly_model.message_encoder.eval()
    anomaly_model.sequence_encoder.eval()
    
    with torch.no_grad():
        log_embedder = LogEmbedder(anomaly_model=anomaly_model)
        
        print('calculating scores')
        print('fit data 0/3', end='\r')
        normal_embs_fit = log_embedder.embed(
            list(set(itertools.chain(*data_subsets['normal_train'])))
        )
        point_anomaly_detector = PointAnomalyDetector(normal_embs_fit)
        
        precomputed_scores_fit = compute_fit_scores(
            point_anomaly_detector, log_embedder, data_subsets['normal_fit']
        )
        precomputed_scores_test = compute_test_scores(
            point_anomaly_detector, log_embedder,
            data_subsets['normal_test'], data_subsets['not_normal_test']
        )

    X_fit, _ = get_features_and_labels(precomputed_scores_fit, only_normal=True)
    X_test, y_test = get_features_and_labels(precomputed_scores_test)
    print('fit shape:', X_fit.shape, 'test shape:', X_test.shape)

    # evaluate
    feature_names = np.array(['Point Mean', 'Point Max', 'Context Mean', 'Context Max'])
    y_pred, rz = evaluate_feature_combinations(
        X_fit, X_test, y_test, feature_names, 
        train_conf['Test']['percentile_threshold']
    )
    
    print('-' * 10)
    print(classification_report(y_test, y_pred, output_dict=False))

    # Create plots
    plot_f1_by_sequence_length(y_test, y_pred, precomputed_scores_test, dataset, model_path)
    plot_feature_contributions(rz, y_test, feature_names, dataset, model_path)
    plot_threshold_sensitivity(X_fit, X_test, y_test, dataset, model_path)

    plt.show(block=True)


def make_data(main_conf, dataset):
    """
    Load and preprocess raw log data for a specific dataset.
    
    Args:
        main_conf: Configuration from main.toml
        dataset: Name of the dataset ('HDFS', 'BGL', 'TBird', or 'ToyDataset')
    """
    train_conf = toml.load(main_conf['train_conf'])

    raw_path = train_conf['Data']['raw_path']
    if 'label_path' in train_conf['Data'].keys():
        label_path = train_conf['Data']['label_path']

    if dataset == "HDFS":
        all_session_logs, all_session_len, all_labels = load_hdfs(label_path=label_path,
                                                                  log_path=raw_path, replace_blk=True,
                                                                  shuffle=False)
    elif dataset == "BGL":
        all_session_logs, all_labels = load_bgl(path=raw_path,
                                                window_size=train_conf['Data']['window_size'],
                                                max_samples=train_conf['Data']['max_samples'],
                                                shuffle=False)
    elif dataset == "TBird":
        # If you want to work with a subset of the Thunderbird dataset, set 'n' to the
        # desired number of sequences. Processing the entire dataset might take a while.
        all_session_logs, all_labels = load_tbird(path=raw_path,
                                                  window_size=train_conf['Data']['window_size'],
                                                  max_samples=train_conf['Data']['max_samples'])
    elif dataset == "ToyDataset":
        all_session_logs, all_labels = load_bgl(path=raw_path,
                                        window_size=train_conf['Data']['window_size'],
                                        max_samples=train_conf['Data']['max_samples'],
                                        shuffle=False)
    else:
        raise NotImplementedError(
            f'Dataset "{dataset}" not supported. Please choose from [HDFS, BGL, TBird, ToyDataset] or add your own data parser. '
            f'Implement a function that takes the path to your raw log file and returns individual log sequences and labels.')
             

    # Save loaded log sequences for future use
    save_log_data(all_session_logs=all_session_logs, all_labels=all_labels,
                  data_path=train_conf['Data']['data_path'],
                  train_frac=train_conf['Data']['train_frac'],
                  val_frac=train_conf['Data']['val_frac'],
                  fit_frac=train_conf['Data']['fit_frac'],
                  test_frac=train_conf['Data']['test_frac'],
                  balance=train_conf['Data']['balance_test'],
                  memmap=False)


def train(main_conf):
    """
    Train the ContraLog anomaly detection model.
    
    Args:
        main_conf: Configuration from main.toml
    """
    train_conf = toml.load(main_conf['train_conf'])

    if train_conf['Misc']['device'] != '':
        device = train_conf['Misc']['device']
    else:
        device = 'cuda' if torch.cuda.device_count() else 'cpu'
    print('device:', device)

    # prepare saved data
    data_path = train_conf['Data']['data_path']
    log_data_util = LogDataUtil(data_path, memmap=False)

    # Create new model with model_conf.toml
    if train_conf['Misc']['warm_start']:
        # load a pretrained model
        model_path = train_conf['Misc']['warm_start_model_path']
        print('Warm start selected! Loading pretrained model:', model_path)
        anomaly_model = AnomalyModel.from_pretrained(model_path, device=device)
    else:
        os.makedirs('models/'+train_conf['Misc']['run_name'], exist_ok=True)
        # Fit new tokenizer, use max_fit_sample for fitting
        print('making new tokenizer')
        tokenizer = make_new_tokenizer(
            max_fit_sample=train_conf['Data']['tokenizer_fit_samples'], log_data_util=log_data_util, model_conf_path=main_conf['model_conf'])
        print('making new model')
        anomaly_model = AnomalyModel(
            model_config_path=main_conf['model_conf'], tokenizer=tokenizer, device=device)

    n_params = sum(p.numel()
                   for p in anomaly_model.message_encoder.parameters())
    print('message_encoder parameter count:', n_params)
    n_params = sum(p.numel()
                   for p in anomaly_model.sequence_encoder.parameters())
    print('sequence_encoder parameter count:', n_params)

    trainer = Trainer(log_data_util, anomaly_model,
                      main_conf['train_conf'], device=device, n_workers=0)
    # Set learning rate from config
    trainer.set_lr(train_conf['Train']['lr'])
    if train_conf['Misc']['calc_tokenizer_stats']:
        print('calculating tokenizer statistics')
        img_path = train_conf['Misc']['save_path'] + \
            train_conf['Misc']['run_name']+'/'
        tokenizer_plots(trainer, subset='val', n=train_conf['Misc']['n_sequences_tokenizer_stats'],
                        save_path=img_path)
    print('current configuration:')
    print(anomaly_model.conf, trainer.conf, end='\n\n')

    # Train loop
    trainer.train()


if __name__ == '__main__':
    print('executable:', sys.executable)
    print('run time:', dt.now().strftime('%Y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser(description='ContraLog main script')
    parser.add_argument('--dataset', action="store",
                        dest='dataset', default='toy')
    parser.add_argument('--script', action="store",
                        dest='script', default='predict')
    args = parser.parse_args()
    dataset = args.dataset
    script = args.script

    if script not in ['make_data', 'train', 'test']:
        raise ValueError(
            "script must be one of ['make_data', 'train', 'test']")
    if dataset not in ['HDFS', 'BGL', 'TBird', 'ToyDataset']:
       print("WARNING - dataset must be one of ['HDFS', 'BGL', 'TBird', 'ToyDataset'] - WARNING")

    print(f"run script: '{script}' for dataset: '{dataset}'")
    main_conf = toml.load('contralog/config/main.toml')[dataset]

    if script == 'make_data':
        make_data(main_conf, dataset)
    elif script == 'train':
        train(main_conf)
    elif script == 'test':
        test(main_conf, dataset)
