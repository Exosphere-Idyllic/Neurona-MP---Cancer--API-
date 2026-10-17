import json
from pathlib import Path
from app.domain.mp_neuron import MPNeuron

def load_model(path: Path) -> MPNeuron:
    data = json.loads(path.read_text())
    return MPNeuron(tuple(data["features"]), data["cutoffs"], data["threshold"])