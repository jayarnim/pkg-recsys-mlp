import random


class NegativeSampler(object):
    def __init__(
        self,
        unobs: dict[int, list], 
        ratio: int,
        seed: int,
    ):
        super().__init__()
        self.unobs = unobs
        self.ratio = ratio
        self.rng = random.Random(seed)

    def __call__(
        self, 
        user_idx: int,
    ) -> int:
        return self.rng.choice(self.unobs[user_idx])