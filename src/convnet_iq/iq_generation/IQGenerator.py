import numpy as np



class IQGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)