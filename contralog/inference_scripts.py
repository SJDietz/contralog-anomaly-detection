from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
from torch import nn
import itertools
import os
# Optional replace log_embedder.embed() with log_embedder.direct_embed() to disable caching
# if you run into memory issues.

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

    def save(self, model_path: str):
        """
        Saves the normalized reference embeddings to disk.

        Args:
            model_path (str): Directory to save the detector file.
        """
        os.makedirs(model_path, exist_ok=True)
        np.savez(
            os.path.join(model_path, 'point_anomaly_detector.npz'),
            normal_embs=self.normal_embs
        )

    @classmethod
    def load_from_pretrained(cls, model_path: str):
        """
        Restores a PointAnomalyDetector from a previously saved file.

        Args:
            model_path (str): Directory containing 'point_anomaly_detector.npz'.

        Returns:
            PointAnomalyDetector: Restored detector, ready to score.
        """
        path = os.path.join(model_path, 'point_anomaly_detector.npz')
        data = np.load(path)
        instance = cls.__new__(cls)
        # normal_embs are already L2-normalised, restore without re-normalising
        instance.normal_embs = data['normal_embs']
        instance.nena = NearestNeighbors(
            n_neighbors=1, algorithm='auto', metric='euclidean', n_jobs=16
        ).fit(instance.normal_embs)
        return instance


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


def get_contextual_anomaly_score_single(log_embedder, single_log_sequence, truncate:bool=True):
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
    with torch.inference_mode():
        if truncate:
            max_len = log_embedder.anomaly_model.conf['max_sequ_len']
            embs = log_embedder.direct_embed(logs=single_log_sequence[0:max_len])
        else:
            embs = log_embedder.direct_embed(logs=single_log_sequence)
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


def get_contextual_anomaly_scores(log_embedder, log_sequences, truncate:bool=True):
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
                log_embedder, single_log_sequence, truncate=truncate))
        except Exception as e:
            print(f"Error processing sequence: {e}")
    return anomaly_scores


# ---------------------------------------------------------------------------

def _robust_z_scores(X: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    """Compute robust z-scores from feature values."""
    mad_safe = mad.copy()
    mad_safe[mad_safe == 0] = 1e-9
    return np.abs((X - med) / mad_safe)


def _extract_sequence_features(
    point_scores: list, contextual_scores: list
) -> np.ndarray:
    """
    Aggregate scores into one feature vector per sequence.
    """
    rows = []
    for point, context in zip(point_scores, contextual_scores):
        rows.append([
            float(np.mean(point)),
            float(point.max()),
            float(np.mean(context)),
            float(context.max()),
        ])
    return np.array(rows)


class InferenceManager:
    """
    Wrapper for model loading, calibration, and inference.

    This class bundles the pretrained anomaly model, the point anomaly
    detector, and calibration parameters for sequence-level scoring.
    """

    _CALIBRATION_FILE = 'calibration_params.npz'

    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Avoid a circular dependency
        from contralog.models import AnomalyModel
        from contralog.log_embedder import LogEmbedder

        self.anomaly_model = AnomalyModel.from_pretrained(model_path, device=device)
        self.anomaly_model.eval()
        self.log_embedder = LogEmbedder(anomaly_model=self.anomaly_model)

        # Populated by calibrate() or from_pretrained() !
        self.point_anomaly_detector: PointAnomalyDetector = None
        self._cal_med: np.ndarray = None
        self._cal_mad: np.ndarray = None
        self._cal_rz_max: np.ndarray = None
        self._cal_threshold: float = None


    def _save_calibration(self):
        np.savez(
            os.path.join(self.model_path, self._CALIBRATION_FILE),
            med=self._cal_med,
            mad=self._cal_mad,
            rz_max=self._cal_rz_max,
            threshold=self._cal_threshold,
        )

    def _load_calibration(self):
        cal_path = os.path.join(self.model_path, self._CALIBRATION_FILE)
        if not os.path.exists(cal_path):
            raise FileNotFoundError(
                f"Calibration file not found at {cal_path}. "
                "Run calibrate() first."
            )
        data = np.load(cal_path)
        self._cal_med = data['med']
        self._cal_mad = data['mad']
        self._cal_rz_max = data['rz_max']
        self._cal_threshold = float(data['threshold'])

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = None):
        """
        Load model weights and saved calibration from disk.

        Args:
            model_path: Directory with model and calibration files.
            device: Torch device. Auto-detected if None.

        Returns:
            InferenceManager: Ready-to-use inference manager.
        """
        manager = cls(model_path=model_path, device=device)
        manager.point_anomaly_detector = PointAnomalyDetector.load_from_pretrained(
            model_path
        )
        manager._load_calibration()
        return manager

    def calibrate(
        self,
        point_anomaly_references: list,
        calibration_sequences: list,
        percentile_threshold: float = 95.0,
        truncate:bool=True
    ):
        """
        Fit point-anomaly references and sequence-level calibration stats.

        Args:
            point_anomaly_references:
                Normal sequences used as reference logs for nearest-neighbor
                point anomaly scoring.
            calibration_sequences:
                Normal sequences used to compute median/MAD and threshold.
            percentile_threshold:
                Percentile used as decision threshold.
            truncate:
                If True, limit contextual scoring to max sequence length.
        """
        with torch.inference_mode():
            flat_logs = list(set(itertools.chain(*point_anomaly_references)))
            print('Embedding point anomaly references...')
            normal_embs = self.log_embedder.direct_embed(flat_logs)
            self.point_anomaly_detector = PointAnomalyDetector(
                np.array(normal_embs)
            )
            self.point_anomaly_detector.save(self.model_path)

            print('Calculating point anomaly scores...')
            cal_point = get_point_anomaly_scores(
                self.point_anomaly_detector,
                self.log_embedder,
                calibration_sequences,
            )
            print('Calculating contextual anomaly scores...')
            cal_ctx = get_contextual_anomaly_scores(
                self.log_embedder, calibration_sequences, truncate
            )

        X_cal = _extract_sequence_features(cal_point, cal_ctx)

        med = np.median(X_cal, axis=0)
        mad = np.median(np.abs(X_cal - med), axis=0)
        rz = _robust_z_scores(X_cal, med, mad)
        rz_max = rz.max(axis=0)
        rz_max[rz_max == 0] = 1.0
        scores = np.linalg.norm(rz / rz_max, axis=1, ord=2)

        self._cal_med = med
        self._cal_mad = mad
        self._cal_rz_max = rz_max
        self._cal_threshold = float(np.percentile(scores, percentile_threshold))

        self._save_calibration()
        print(
            f"Calibration complete. Threshold={self._cal_threshold:.4f} "
            f"(p{percentile_threshold}). Saved to '{self.model_path}'."
        )

    def score(self, log_sequences: list, truncate: bool = True) -> dict:
        """
            Score one or more log sequences.
        Args:
            log_sequences: Log sequences to evaluate.
            truncate: If True, truncate for contextual scoring.
        Returns:
            dict: Point/context scores, sequence features, sequence anomaly
                scores, and binary anomaly predictions.
        """
        if self.point_anomaly_detector is None or self._cal_threshold is None:
            raise RuntimeError(
                "Model is not calibrated. Call calibrate() or load with "
                "InferenceManager.from_pretrained()."
            )

        with torch.inference_mode():
            point_scores = get_point_anomaly_scores(
                self.point_anomaly_detector, self.log_embedder, log_sequences
            )
            contextual_scores = get_contextual_anomaly_scores(
                self.log_embedder, log_sequences, truncate
            )

        features = _extract_sequence_features(point_scores, contextual_scores)
        rz = _robust_z_scores(features, self._cal_med, self._cal_mad)
        seq_scores = np.linalg.norm(rz / self._cal_rz_max, axis=1, ord=2)

        return {
            'point_scores': point_scores,
            'contextual_scores': contextual_scores,
            'sequence_features': features,
            'sequence_anomaly_scores': seq_scores,
            'is_anomaly': seq_scores > self._cal_threshold,
        }

