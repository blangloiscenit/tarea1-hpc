"""
Módulo para generar datos sintéticos para regresión lineal con bootstrap.
Item (a) de la Tarea 1 - IIC3533 Computación de Alto Rendimiento
"""

import numpy as np

RANDOM_SEED = 42
NUM_OBSERVATIONS = 100_000
NUM_FEATURES = 300
NUM_RESAMPLES = 48


def generate_synthetic_data(
    num_observations: int = NUM_OBSERVATIONS,
    num_features: int = NUM_FEATURES,
    seed: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos para regresión lineal.

    Pasos:
    (i)   Muestrea k+1 coeficientes verdaderos β* desde N(0,1)
    (ii)  Genera matriz X de N×(k+1) con columna de unos al inicio
    (iii) Calcula y = Xβ* + ruido N(0,1)

    Args:
        num_observations: Número de observaciones N
        num_features: Número de variables de entrada k
        seed: Semilla aleatoria para reproducibilidad

    Returns:
        X: Matriz de diseño (N, k+1) con columna de unos
        y: Vector de salidas (N,)
        beta_true: Coeficientes verdaderos (k+1,)
    """
    rng = np.random.default_rng(seed)

    num_coefficients = num_features + 1

    beta_true = rng.standard_normal(num_coefficients)

    feature_matrix = rng.standard_normal((num_observations, num_features))
    ones_column = np.ones((num_observations, 1))
    X = np.hstack([ones_column, feature_matrix])

    noise = rng.standard_normal(num_observations)
    y = X @ beta_true + noise

    return X, y, beta_true


def compute_confidence_intervals(
    bootstrap_coefficients: np.ndarray,
    confidence_level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula intervalos de confianza bootstrap para cada coeficiente.

    Args:
        bootstrap_coefficients: Matriz (B, k+1) con coeficientes de cada resample
        confidence_level: Nivel de confianza (default 0.95 para 95%)

    Returns:
        lower_bounds: Límites inferiores del intervalo
        upper_bounds: Límites superiores del intervalo
    """
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bounds = np.percentile(bootstrap_coefficients, lower_percentile, axis=0)
    upper_bounds = np.percentile(bootstrap_coefficients, upper_percentile, axis=0)

    return lower_bounds, upper_bounds


def save_data(X: np.ndarray, y: np.ndarray, beta_true: np.ndarray, output_dir: str = "data"):
    """
    Guarda los datos generados en archivos .npy

    Args:
        X: Matriz de diseño
        y: Vector de salidas
        beta_true: Coeficientes verdaderos
        output_dir: Directorio de salida
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    np.save(os.path.join(output_dir, "beta_true.npy"), beta_true)

    print(f"Datos guardados en {output_dir}/")


def load_data(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga los datos desde archivos .npy

    Args:
        data_dir: Directorio donde están los datos

    Returns:
        X, y, beta_true
    """
    import os
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    beta_true = np.load(os.path.join(data_dir, "beta_true.npy"))

    return X, y, beta_true


if __name__ == "__main__":
    print("Generando datos sintéticos...")
    X, y, beta_true = generate_synthetic_data()

    print(f"Dimensiones de X: {X.shape}")
    print(f"Dimensiones de y: {y.shape}")
    print(f"Número de coeficientes: {len(beta_true)}")
    print(f"Primeros 5 coeficientes verdaderos: {beta_true[:5]}")

    save_data(X, y, beta_true)
    print("\nDatos guardados exitosamente.")
