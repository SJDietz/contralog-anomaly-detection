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
        unique_logs = np.array(list(set(logs)))
        mask = [False if hash(
            log) in self.emb_dict else True for log in unique_logs]
        unique_logs = list(unique_logs[mask])
        # Only embed unique logs that are not in the cach (emb_dict)
        if len(unique_logs) > 0:
            # print(len(unique_logs), 'new logs found')
            with torch.no_grad():
                embs = self.anomaly_model.embed(
                    logs=unique_logs, batch_size=batch_size).cpu().detach().numpy()

            log_emb_dict = dict(zip([hash(log) for log in unique_logs], embs))
            self.emb_dict = self.emb_dict | log_emb_dict

        embs = [self.emb_dict[hash(log)] for log in logs]
        return embs

    def direct_embed(self, logs, batch_size: int = 256):
        """Embed without caching"""
        self.anomaly_model.message_encoder.eval()
        with torch.no_grad():
            embs = self.anomaly_model.embed(
                logs=logs, batch_size=batch_size).cpu().detach().numpy()
        return embs
