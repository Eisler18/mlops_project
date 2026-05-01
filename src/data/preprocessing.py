import numpy as np


class Preprocessor:
    """
    Inference-time preprocessing wrapper.

    IMPORTANT:
    - The scaler is loaded from W&B elsewhere
    - This class does NOT interact with W&B
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, raw_features):
        """
        Args:
            raw_features: list[float] or np.ndarray (19 features)

        Returns:
            np.ndarray: shape (1, n_features)
        """
        X = np.array(raw_features, dtype=float).reshape(1, -1)

        if self.scaler is not None:
            return self.scaler.transform(X)

        return X