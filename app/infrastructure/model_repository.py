import json
from pathlib import Path
from app.domain.mp_neuron import MPNeuron

def load_model(path: Path) -> MPNeuron:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MPNeuron(
        features=tuple(data["features"]),
        cutoffs=data["cutoffs"],
        threshold=data["threshold"],
        accuracy=data.get("accuracy", 0.9510),
    )
