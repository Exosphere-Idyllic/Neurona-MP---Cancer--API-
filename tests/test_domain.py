from pathlib import Path
from app.domain.mp_neuron import MPNeuron
from app.infrastructure.model_repository import load_model

def test_mp_neuron_logic():
    features = ("worst perimeter", "worst radius")
    cutoffs = {"worst perimeter": 100.0, "worst radius": 15.0}
    threshold = 2
    neuron = MPNeuron(features=features, cutoffs=cutoffs, threshold=threshold)

    # Caso benigno: ambos valores por debajo del corte -> score = 2 >= 2 -> True
    is_benign, score = neuron.predict({"worst perimeter": 80.0, "worst radius": 12.0})
    assert is_benign is True
    assert score == 2

    # Caso maligno: solo uno por debajo del corte -> score = 1 < 2 -> False
    is_benign, score = neuron.predict({"worst perimeter": 110.0, "worst radius": 12.0})
    assert is_benign is False
    assert score == 1

def test_load_model(tmp_path: Path):
    dummy_file = tmp_path / "model.json"
    dummy_file.write_text(
        '{"features": ["a"], "cutoffs": {"a": 5.0}, "threshold": 1}',
        encoding="utf-8"
    )
    neuron = load_model(dummy_file)
    assert neuron.features == ("a",)
    assert neuron.cutoffs == {"a": 5.0}
    assert neuron.threshold == 1
