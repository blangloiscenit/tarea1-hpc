"""
Bootstrap con BaggingRegressor de sklearn.
El bootstrapping y el paralelismo son completamente internos.

Item (b) - Versión 1: bs_auto.py
"""

import time
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

from data_generator import (
    generate_synthetic_data,
    compute_confidence_intervals,
    NUM_RESAMPLES,
    RANDOM_SEED
)


def bootstrap_with_bagging(
    X: np.ndarray,
    y: np.ndarray,
    num_resamples: int = NUM_RESAMPLES,
    num_jobs: int = 1,
    seed: int = RANDOM_SEED
) -> np.ndarray:
    """
    Ejecuta bootstrap usando BaggingRegressor de sklearn.

    Args:
        X: Matriz de diseño (N, k+1)
        y: Vector de salidas (N,)
        num_resamples: Número de resamples B
        num_jobs: Número de procesos paralelos
        seed: Semilla aleatoria

    Returns:
        bootstrap_coefficients: Matriz (B, k+1) con coeficientes de cada resample
    """
    base_estimator = LinearRegression(fit_intercept=False)

    bagging = BaggingRegressor(
        estimator=base_estimator,
        n_estimators=num_resamples,
        max_samples=1.0,
        bootstrap=True,
        n_jobs=num_jobs,
        random_state=seed
    )

    bagging.fit(X, y)

    bootstrap_coefficients = np.array([
        estimator.coef_ for estimator in bagging.estimators_
    ])

    return bootstrap_coefficients


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
    bootstrap_coefficients = bootstrap_with_bagging(X, y, num_jobs=num_jobs)
    elapsed_time = time.perf_counter() - start_time

    lower_bounds, upper_bounds = compute_confidence_intervals(bootstrap_coefficients)

    coefficients_in_interval = np.sum(
        (beta_true >= lower_bounds) & (beta_true <= upper_bounds)
    )
    coverage_rate = coefficients_in_interval / len(beta_true)

    return {
        "method": "bs_auto (BaggingRegressor)",
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

    print(f"Ejecutando bs_auto.py con {num_jobs} proceso(s)...")
    results = run_bootstrap(num_jobs=num_jobs)

    print(f"Tiempo de ejecución: {results['elapsed_time']:.4f} segundos")
    print(f"Cobertura del intervalo de confianza: {results['coverage_rate']:.2%}")
    print(f"Primeros 5 coeficientes verdaderos: {results['beta_true'][:5]}")
    print(f"Primeros 5 límites inferiores: {results['lower_bounds'][:5]}")
    print(f"Primeros 5 límites superiores: {results['upper_bounds'][:5]}")
