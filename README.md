# API de Clasificación de Cáncer de Mama (Neurona de McCulloch-Pitts)

API REST moderna construida con **FastAPI** para la detección y clasificación de muestras de cáncer de mama en tumores **benignos** o **malignos**, implementando un modelo lineal binarizado de **Neurona McCulloch-Pitts (MP Neuron)** entrenado sobre el conjunto de datos *Breast Cancer Wisconsin* de Scikit-Learn.

> [!WARNING]
> **Aviso Médico / Disclaimer**: Este proyecto tiene fines estrictamente educativos y de investigación demostrativa. No debe utilizarse como herramienta de diagnóstico médico ni para la toma de decisiones clínicas.

---

## 📌 Tabla de Contenidos
- [Características y Rendimiento](#-características-y-rendimiento)
- [Fundamento Teórico del Modelo](#-fundamento-teórico-del-modelo)
- [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Ejecución del Servidor](#-ejecución-del-servidor)
- [Documentación de los Endpoints](#-documentación-de-los-endpoints)
  - [1. Health Check](#1-health-check)
  - [2. Inferencia / Predicción](#2-inferencia--predicción)
- [Ejecución de Pruebas](#-ejecución-de-pruebas)
- [Reentrenamiento del Modelo](#-reentrenamiento-del-modelo)

---

## ⚡ Características y Rendimiento

- **Framework**: FastAPI (asíncrono, validación con Pydantic V2 y documentación automática OpenAPI).
- **Exactitud (Accuracy)**: **95.10%** en conjunto de prueba (`accuracy = 0.951`).
- **Inferencia Ultrarrápida**: Operaciones binarias directas en memoria sin sobrecarga de cómputo.
- **Diseño Limpio**: Arquitectura modular desacoplada por capas (Dominio, Servicios, Infraestructura, API).

---

## 🧠 Fundamento Teórico del Modelo

La **Neurona McCulloch-Pitts** es una de las primeras representaciones matemáticas de una neurona biológica. En este proyecto se optimizó para clasificación binaria mediante:

1. **Variables Seleccionadas (Features)**:
   - `worst perimeter`: Peor perímetro medido del núcleo celular.
   - `worst radius`: Peor radio celular.
   - `worst concave points`: Mayor número/proporción de porciones cóncavas del contorno.
   - `mean concave points`: Promedio de puntos cóncavos en las células.

2. **Binarización por Umbrales de Corte (Cutoffs)**:
   En el tejido mamario, valores más bajos en estas métricas suelen asociarse a células sanas o tejido benigno:
   $$\text{entrada\_binaria}_i = 1 \iff x_i \le \text{cutoff}_i$$

3. **Función de Activación por Umbral (Threshold)**:
   $$\text{score} = \sum_{i=1}^{n} \text{entrada\_binaria}_i$$
   $$\text{diagnóstico} = \begin{cases} \text{"benigno"} & \text{si } \text{score} \ge 3 \\ \text{"maligno"} & \text{si } \text{score} < 3 \end{cases}$$

---

## 🏗 Arquitectura del Proyecto

El código está estructurado siguiendo principios de separación de responsabilidades:

```text
├── app/
│   ├── api/
│   │   ├── deps.py               # Inyección de dependencias (FastAPI Depends)
│   │   └── routes/
│   │       ├── health.py         # Endpoint de verificación de estado
│   │       └── predict.py        # Endpoint de predicción
│   ├── core/
│   │   └── config.py             # Configuración centralizada (Pydantic Settings)
│   ├── domain/
│   │   └── mp_neuron.py          # Entidad de dominio pura con la lógica MP Neuron
│   ├── infrastructure/
│   │   └── model_repository.py   # Carga y deserialización del modelo desde disco
│   ├── schemas/
│   │   └── prediction.py         # Esquemas de entrada y salida (Pydantic)
│   ├── services/
│   │   └── prediction_service.py # Servicio que orquesta la inferencia
│   └── main.py                   # Fábrica de la aplicación y ciclo de vida (Lifespan)
├── artifacts/
│   └── model.json                # Parámetros entrenados (features, cutoffs, threshold, accuracy)
├── notebooks/
│   └── Machine Learning II - Neurona M-P cancer.ipynb # Análisis exploratorio y notebook original
├── scripts/
│   └── train.py                  # Script para entrenar y re-exportar el modelo
├── tests/
│   ├── test_api.py               # Tests de integración de la API HTTP
│   └── test_domain.py            # Tests unitarios de la lógica del dominio
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Documentación general
```

---

## 📋 Requisitos Previos

- **Python**: Versión 3.10 o superior (recomendado Python 3.11).
- **Git** instalado.

---

## 🚀 Instalación y Configuración

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/Exosphere-Idyllic/Neurona-MP---Cancer--API-.git
   cd "Neurona MP - Cancer (API)"
   ```

2. **Crear y activar el entorno virtual**:
   - En Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - En Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥 Ejecución del Servidor

Inicia el servidor en modo desarrollo con recarga automática:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Una vez levantado, puedes explorar la documentación interactiva provista automáticamente por FastAPI:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 Documentación de los Endpoints

### 1. Health Check

Verifica que el servicio esté activo y respondiendo correctamente.

- **Método**: `GET`
- **Ruta**: `/health`
- **Respuesta Exitosa (`200 OK`)**:
  ```json
  {
    "status": "ok"
  }
  ```

---

### 2. Inferencia / Predicción

Evalúa los 4 biomarcadores celulares e infiere el diagnóstico estimado.

- **Método**: `POST`
- **Ruta**: `/api/v1/predict`
- **Headers**: `Content-Type: application/json`

#### Parámetros del Body (JSON)

| Campo | Tipo | Validación | Descripción |
|---|---|---|---|
| `worst_perimeter` | `float` | `> 0` | Peor valor de perímetro celular observado. |
| `worst_radius` | `float` | `> 0` | Peor valor de radio celular observado. |
| `worst_concave_points` | `float` | `ge 0` | Peor proporción de puntos cóncavos en el contorno. |
| `mean_concave_points` | `float` | `ge 0` | Promedio de puntos cóncavos observados. |

#### Ejemplo de Solicitud (Caso Benigno)
```json
{
  "worst_perimeter": 85.0,
  "worst_radius": 13.0,
  "worst_concave_points": 0.08,
  "mean_concave_points": 0.03
}
```

#### Respuesta Exitosa (`200 OK`)
```json
{
  "diagnosis": "benigno",
  "score": 4,
  "threshold": 3,
  "accuracy": 0.951,
  "disclaimer": "Modelo educativo, no es una herramienta de diagnóstico."
}
```

#### Ejemplo de Solicitud (Caso Maligno)
```json
{
  "worst_perimeter": 170.0,
  "worst_radius": 25.0,
  "worst_concave_points": 0.25,
  "mean_concave_points": 0.15
}
```

#### Respuesta Exitosa (`200 OK`)
```json
{
  "diagnosis": "maligno",
  "score": 0,
  "threshold": 3,
  "accuracy": 0.951,
  "disclaimer": "Modelo educativo, no es una herramienta de diagnóstico."
}
```

#### Ejemplo con cURL
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "worst_perimeter": 85.0,
       "worst_radius": 13.0,
       "worst_concave_points": 0.08,
       "mean_concave_points": 0.03
     }'
```

#### Ejemplo en Python (`requests`)
```python
import requests

url = "http://localhost:8000/api/v1/predict"
payload = {
    "worst_perimeter": 85.0,
    "worst_radius": 13.0,
    "worst_concave_points": 0.08,
    "mean_concave_points": 0.03
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Diagnóstico: {data['diagnosis']}")
print(f"Puntaje obtenido: {data['score']}/{data['threshold']}")
print(f"Exactitud del modelo: {data['accuracy'] * 100:.2f}%")
```

---

## 🧪 Ejecución de Pruebas

El proyecto cuenta con cobertura de pruebas unitarias y de integración utilizando `pytest`:

```bash
pytest
```

Salida esperada:
```text
tests\test_api.py ..                                    [ 50%]
tests\test_domain.py ..                                 [100%]
========================= 4 passed in 0.30s =========================
```

---

## 🔄 Reentrenamiento del Modelo

Para reentrenar la neurona, reoptimizar los puntos de corte por percentiles y regenerar el artefacto [`artifacts/model.json`](file:///c:/Users/USER/Codespaces/Neurona%20MP%20-%20Cancer%20(API)/artifacts/model.json):

```bash
python scripts/train.py
```

El script imprimirá en consola las métricas obtenidas tanto en el conjunto de entrenamiento como en el de prueba, y actualizará automáticamente el archivo de configuración consumido por la API al iniciar.
