from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
from torch import nn
import itertools


class PointAnomalyDetector:
    """
    A simple unsupervised anomaly detector that scores log embeddings based on
    their cosine distance to the closest embedding from a set of known normal logs.

    Attributes:
        normal_embs (np.ndarray): Embeddings of known normal log messages.
        nena (NearestNeighbors): Nearest neighbor model fitted on normal embeddings.
    """

    def __init__(self, normal_embs):
        norms = np.linalg.norm(normal_embs, ord=2, axis=1, keepdims=True)
        normal_embs = normal_embs / norms
        # store normal embs for accurate cosine sim calculation later
        self.normal_embs = normal_embs
        # 'euclidean' is faster than 'cosine'
        self.nena = NearestNeighbors(
            n_neighbors=1, algorithm='auto', metric='euclidean', n_jobs=16).fit(normal_embs)

    def get_score(self, embs):
        """
        Computes anomaly scores for given embeddings based on their distance
        to the closest normal log embedding.

        Args:
            embs (list): Embeddings of logs to score.

        Returns:
            np.ndarray: Anomaly scores (cosine distance to nearest normal embedding).
        """
        norms = np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
        embs = embs / norms
        # Order is preserved, but distances are not cosine
        distances, indices = self.nena.kneighbors(embs)
        a = embs.squeeze()
        b = self.normal_embs[indices].squeeze()
        scores = 1 - np.sum(a * b, axis=1)
        # cos_dist
        return scores


def get_point_anomaly_scores(point_anomaly_detector, log_embedder, log_sequences):
    """
    Computes point-wise anomaly scores for log messages in sequences using a 
    nearest-neighbor-based detector.

    Each log message is embedded and scored based on its cosine distance to the
    nearest known normal log embedding.

    Args:
        point_anomaly_detector (PointAnomalyDetector): 
            Anomaly detector model.
        log_embedder (LogEmbedder): 
            Object used to convert raw log messages into embeddings.
        log_sequences (List[List[str]]): 
            A list of sequences, where each sequence is a list of log messages.

    Returns:
        List[np.ndarray]: 
            A list of arrays of anomaly scores, one per input sequence. Each score
            corresponds to a log in the original sequence and indicates how anomalous 
            the log is (higher -> no similar known logs -> more anomalous).
    """
    lengths = [len(l) for l in log_sequences]
    logs = list(itertools.chain(*log_sequences))
    embs = log_embedder.direct_embed(logs)
    embs = np.array(embs)
    scores = point_anomaly_detector.get_score(embs)

    c_sum = np.cumsum([0] + lengths)
    scores_tmp = []
    for i in range(len(c_sum)-1):
        scores_tmp.append(scores[c_sum[i]: c_sum[i+1]])
    scores = scores_tmp
    return scores


def get_contextual_anomaly_score_single(log_embedder, single_log_sequence):
    """
    Computes contextual anomaly scores for a single log sequence by
    masking each log one at a time and measuring reconstruction error.

    Args:
        log_embedder (LogEmbedder): 
            Embedding wrapper with access to the anomaly model.
        single_log_sequence (List[str]): 
            A single sequence of raw log messages.

    Returns:
        np.ndarray: 
            A 1D array of contextual anomaly scores for each log in the sequence.
    """
    with torch.no_grad():
        max_len = log_embedder.anomaly_model.conf['max_sequ_len']
        embs = log_embedder.direct_embed(logs=single_log_sequence[0:max_len])
        embs = torch.tensor(np.array(embs)).to(
            log_embedder.anomaly_model.device)
        original_embs = embs.clone()

        embs = embs[np.newaxis, :, :]
        embs = embs.repeat(embs.shape[1], 1, 1)  # shape [batch, len, feat]

        ind = torch.arange(embs.shape[1])

        embs[ind, ind, :] = log_embedder.anomaly_model.sequence_encoder.get_mask_token(
            norm=False)

        pred = log_embedder.anomaly_model.sequence_encoder.forward(
            embs.transpose(0, 1))
        pred = pred[np.arange(pred.shape[1]), np.arange(pred.shape[1]), :]

        criterion = nn.CosineEmbeddingLoss(reduction='none')
        loss = criterion(pred, original_embs, torch.tensor(
            [1]).to(log_embedder.anomaly_model.device))

        l_item = loss.detach().cpu().numpy()
        return l_item


def get_contextual_anomaly_scores(log_embedder, log_sequences):
    """
    Computes contextual anomaly scores for a list of log sequences.

    Each log in a sequence is masked in turn, and the model attempts to 
    reconstruct it using the surrounding context. A higher reconstruction 
    error indicates a higher likelihood of being anomalous.

    Args:
        log_embedder (LogEmbedder): 
            The embedding and anomaly model wrapper.
        log_sequences (List[List[str]]): 
            A list of log message sequences.

    Returns:
        List[np.ndarray]: 
            A list of 1D arrays containing contextual anomaly scores per log.
            Each array corresponds to one input sequence.
    """
    anomaly_scores = []
    for single_log_sequence in log_sequences:
        try:
            anomaly_scores.append(get_contextual_anomaly_score_single(
                log_embedder, single_log_sequence))
        except Exception as e:
            print(f"Error processing sequence: {e}")
    return anomaly_scores
