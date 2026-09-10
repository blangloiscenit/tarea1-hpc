"""
Bootstrap con joblib.Parallel y LinearRegression de sklearn.
El paralelismo se implementa manualmente con joblib.

Item (b) - Versión 2: bs_sklearn.py
"""

import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from data_generator import (
    generate_synthetic_data,
    compute_confidence_intervals,
    NUM_RESAMPLES,
    RANDOM_SEED
)


def fit_single_resample(
    X: np.ndarray,
    y: np.ndarray,
    resample_index: int,
    seed: int
) -> np.ndarray:
    """
    Ajusta un modelo de regresión lineal sobre un resample bootstrap.

    Args:
        X: Matriz de diseño original (N, k+1)
        y: Vector de salidas original (N,)
        resample_index: Índice del resample (para seed)
        seed: Semilla base

    Returns:
        Coeficientes del modelo ajustado
    """
    rng = np.random.default_rng(seed + resample_index)
    num_observations = X.shape[0]

    indices = rng.integers(0, num_observations, size=num_observations)
    X_resample = X[indices]
    y_resample = y[indices]

    model = LinearRegression(fit_intercept=False)
    model.fit(X_resample, y_resample)

    return model.coef_


def bootstrap_with_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    num_resamples: int = NUM_RESAMPLES,
    num_jobs: int = 1,
    seed: int = RANDOM_SEED
) -> np.ndarray:
    """
    Ejecuta bootstrap usando joblib.Parallel y LinearRegression.

    Args:
        X: Matriz de diseño (N, k+1)
        y: Vector de salidas (N,)
        num_resamples: Número de resamples B
        num_jobs: Número de procesos paralelos
        seed: Semilla aleatoria

    Returns:
        bootstrap_coefficients: Matriz (B, k+1) con coeficientes de cada resample
    """
    coefficients_list = Parallel(n_jobs=num_jobs)(
        delayed(fit_single_resample)(X, y, i, seed)
        for i in range(num_resamples)
    )

    return np.array(coefficients_list)


def run_bootstrap(num_jobs: int = 1) -> dict:
    """
    Ejecuta el experimento completo de bootstrap.

    Args:
        num_jobs: Número de procesos paralelos

    Returns:
        Diccionario con resultados y métricas
    """
    X, y, beta_true = generate_synthetic_data()

    start_time = time.perf_counter()
    bootstrap_coefficients = bootstrap_with_sklearn(X, y, num_jobs=num_jobs)
    elapsed_time = time.perf_counter() - start_time

    lower_bounds, upper_bounds = compute_confidence_intervals(bootstrap_coefficients)

    coefficients_in_interval = np.sum(
        (beta_true >= lower_bounds) & (beta_true <= upper_bounds)
    )
    coverage_rate = coefficients_in_interval / len(beta_true)

    return {
        "method": "bs_sklearn (joblib + LinearRegression)",
        "num_jobs": num_jobs,
        "elapsed_time": elapsed_time,
        "bootstrap_coefficients": bootstrap_coefficients,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "beta_true": beta_true,
        "coverage_rate": coverage_rate
    }


if __name__ == "__main__":
    import sys

    num_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    print(f"Ejecutando bs_sklearn.py con {num_jobs} proceso(s)...")
    results = run_bootstrap(num_jobs=num_jobs)

    print(f"Tiempo de ejecución: {results['elapsed_time']:.4f} segundos")
    print(f"Cobertura del intervalo de confianza: {results['coverage_rate']:.2%}")
    print(f"Primeros 5 coeficientes verdaderos: {results['beta_true'][:5]}")
    print(f"Primeros 5 límites inferiores: {results['lower_bounds'][:5]}")
    print(f"Primeros 5 límites superiores: {results['upper_bounds'][:5]}")
