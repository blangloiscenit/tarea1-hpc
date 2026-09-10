"""
Bootstrap con joblib.Parallel y NumPy puro.
Resuelve (X'X)β = X'y usando solo NumPy.

Item (b) - Versión 3: bs_numpy.py
"""

import time
import numpy as np
from joblib import Parallel, delayed

from data_generator import (
    generate_synthetic_data,
    compute_confidence_intervals,
    NUM_RESAMPLES,
    RANDOM_SEED
)


def fit_single_resample_numpy(
    X: np.ndarray,
    y: np.ndarray,
    resample_index: int,
    seed: int
) -> np.ndarray:
    """
    Ajusta regresión lineal usando mínimos cuadrados con NumPy.
    Resuelve β = (X'X)^(-1) X'y

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

    XtX = X_resample.T @ X_resample
    Xty = X_resample.T @ y_resample
    beta = np.linalg.solve(XtX, Xty)

    return beta


def bootstrap_with_numpy(
    X: np.ndarray,
    y: np.ndarray,
    num_resamples: int = NUM_RESAMPLES,
    num_jobs: int = 1,
    seed: int = RANDOM_SEED
) -> np.ndarray:
    """
    Ejecuta bootstrap usando joblib.Parallel y NumPy puro.

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
        delayed(fit_single_resample_numpy)(X, y, i, seed)
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
    bootstrap_coefficients = bootstrap_with_numpy(X, y, num_jobs=num_jobs)
    elapsed_time = time.perf_counter() - start_time

    lower_bounds, upper_bounds = compute_confidence_intervals(bootstrap_coefficients)

    coefficients_in_interval = np.sum(
        (beta_true >= lower_bounds) & (beta_true <= upper_bounds)
    )
    coverage_rate = coefficients_in_interval / len(beta_true)

    return {
        "method": "bs_numpy (joblib + NumPy)",
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
    from threadpoolctl import threadpool_info

    num_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    print("Información de threadpool:")
    for info in threadpool_info():
        print(f"  {info['user_api']}: {info['num_threads']} threads ({info['prefix']})")

    print(f"\nEjecutando bs_numpy.py con {num_jobs} proceso(s)...")
    results = run_bootstrap(num_jobs=num_jobs)

    print(f"Tiempo de ejecución: {results['elapsed_time']:.4f} segundos")
    print(f"Cobertura del intervalo de confianza: {results['coverage_rate']:.2%}")
    print(f"Primeros 5 coeficientes verdaderos: {results['beta_true'][:5]}")
    print(f"Primeros 5 límites inferiores: {results['lower_bounds'][:5]}")
    print(f"Primeros 5 límites superiores: {results['upper_bounds'][:5]}")
