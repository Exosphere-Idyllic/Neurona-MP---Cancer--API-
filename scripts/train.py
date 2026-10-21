import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "worst perimeter",
    "worst radius",
    "worst concave points",
    "mean concave points",
]

class MpNeuronTrainer:
    def __init__(self):
        self.threshold = None

    def model(self, x):
        return sum(x) >= self.threshold

    def predict(self, X):
        return np.array([self.model(x) for x in X])

    def fit(self, X, Y):
        accuracy = {}
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y, Y_pred)
        self.threshold = max(accuracy, key=accuracy.get)
        return accuracy[self.threshold]

def optimize_cutoffs(X_train: pd.DataFrame, y_train: np.ndarray, features: list[str]) -> dict[str, float]:
    """
    Encuentra el punto de corte óptimo para cada variable buscando
    el umbral que maximiza la exactitud individual en train.
    """
    cutoffs = {}
    for col in features:
        best_cutoff = None
        best_acc = 0.0
        # Evaluamos 100 percentiles para buscar el umbral que mejor separa las clases
        threshold_candidates = np.percentile(X_train[col], np.linspace(5, 95, 100))
        for cand in threshold_candidates:
            # Valores menores al corte indican benigno (1)
            pred = (X_train[col].to_numpy() <= cand).astype(int)
            acc = accuracy_score(y_train, pred)
            if acc > best_acc:
                best_acc = acc
                best_cutoff = float(cand)
        cutoffs[col] = best_cutoff
    return cutoffs

def train_and_export():
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)[FEATURES]
    y = dataset.target  # 1 = benigno, 0 = maligno

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, stratify=y, test_size=0.25, random_state=42
    )

    # 1. Optimización de cutoffs individuales en train (evitando el sesgo de pd.cut)
    cutoffs = optimize_cutoffs(X_train, y_train, FEATURES)

    # 2. Binarización usando los cortes óptimos
    X_train_bin = np.zeros(X_train.shape, dtype=int)
    X_test_bin = np.zeros(X_test.shape, dtype=int)

    for idx, col in enumerate(FEATURES):
        X_train_bin[:, idx] = (X_train[col].to_numpy() <= cutoffs[col]).astype(int)
        X_test_bin[:, idx] = (X_test[col].to_numpy() <= cutoffs[col]).astype(int)

    # 3. Entrenar la neurona MP para encontrar el umbral óptimo de votos
    neuron = MpNeuronTrainer()
    train_acc = neuron.fit(X_train_bin, y_train)
    test_preds = neuron.predict(X_test_bin)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"Entrenamiento optimizado completado:")
    print(f" - Features seleccionadas: {FEATURES}")
    print(f" - Cutoffs óptimos calculados (train): {cutoffs}")
    print(f" - Threshold óptimo: {neuron.threshold}")
    print(f" - Train Accuracy: {train_acc:.4f}")
    print(f" - Test Accuracy: {test_acc:.4f}")

    # 4. Guardar artefacto
    output_path = Path("artifacts/model.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "features": FEATURES,
        "cutoffs": cutoffs,
        "threshold": int(neuron.threshold),
        "accuracy": round(float(test_acc), 4),
    }

    output_path.write_text(json.dumps(model_data, indent=2), encoding="utf-8")
    print(f"Modelo guardado exitosamente en: {output_path.resolve()}")

if __name__ == "__main__":
    train_and_export()
