from dataclasses import dataclass

@dataclass(frozen=True)
class MPNeuron:
    features: tuple[str, ...]
    cutoffs: dict[str, float]
    threshold: int
    accuracy: float = 0.9510

    def predict(self, sample: dict[str, float]) -> tuple[bool, int]:
        # 1 si el valor está por debajo del corte, igual que en el notebook
        score = sum(int(sample[f] <= self.cutoffs[f]) for f in self.features)
        return score >= self.threshold, score   # True = benigno
