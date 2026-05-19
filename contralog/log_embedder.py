import torch
import numpy as np


class LogEmbedder():
    """
    Caches and returns embeddings for input logs using the MessageEncoder.
    Duplicate messages are only encoded once and stored for repeat use.

    Args:
        logs (list of str): Log messages to embed.
        batch_size (int): Number of log messages to process per batch.

    Returns:
        list of np.ndarray: Embeddings corresponding to each input log messages.
    """

    def __init__(self, anomaly_model):
        self.emb_dict = {}
        self.anomaly_model = anomaly_model

    def embed(self, logs, batch_size: int = 256):
        """Cach log embeddings"""
        self.anomaly_model.message_encoder.eval()
        # Preserve first-occurrence order when deduping
        unique_logs = list(dict.fromkeys(logs))
        # Find which unique logs are not yet cached
        to_embed = [log for log in unique_logs if hash(log) not in self.emb_dict]
        if len(to_embed) > 0:
            with torch.inference_mode():
                embs_new = self.anomaly_model.embed(logs=to_embed, batch_size=batch_size).cpu().detach().numpy()
            for log, emb in zip(to_embed, embs_new):
                self.emb_dict[hash(log)] = emb

        # Return embeddings in the same order as input `logs` as a numpy array
        embs = np.array([self.emb_dict[hash(log)] for log in logs])
        return embs

    def direct_embed(self, logs, batch_size: int = 256):
        """Embed without caching"""
        self.anomaly_model.message_encoder.eval()
        with torch.inference_mode():
            embs = self.anomaly_model.embed(
                logs=logs, batch_size=batch_size).cpu().detach().numpy()
        return embs
