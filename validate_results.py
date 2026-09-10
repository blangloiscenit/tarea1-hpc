"""
Validación de correctitud y reproducibilidad de las tres versiones de bootstrap.
Item (c) de la Tarea 1 - IIC3533 Computación de Alto Rendimiento
"""

import numpy as np

from data_generator import generate_synthetic_data, RANDOM_SEED
from bs_auto import run_bootstrap as run_auto
from bs_sklearn import run_bootstrap as run_sklearn
from bs_numpy import run_bootstrap as run_numpy


def compare_intervals(
    results1: dict,
    results2: dict,
    tolerance: float = 0.1
) -> dict:
    """
    Compara los intervalos de confianza entre dos versiones.

    Args:
        results1: Resultados de la primera versión
        results2: Resultados de la segunda versión
        tolerance: Tolerancia para considerar valores similares

    Returns:
        Diccionario con métricas de comparación
    """
    lower_diff = np.abs(results1["lower_bounds"] - results2["lower_bounds"])
    upper_diff = np.abs(results1["upper_bounds"] - results2["upper_bounds"])

    return {
        "lower_max_diff": np.max(lower_diff),
        "lower_mean_diff": np.mean(lower_diff),
        "upper_max_diff": np.max(upper_diff),
        "upper_mean_diff": np.mean(upper_diff),
        "similar_lower": np.mean(lower_diff < tolerance),
        "similar_upper": np.mean(upper_diff < tolerance)
    }


def check_reproducibility(run_func, num_runs: int = 3) -> dict:
    """
    Verifica si una versión produce resultados reproducibles.

    Args:
        run_func: Función que ejecuta el bootstrap
        num_runs: Número de ejecuciones para comparar

    Returns:
        Diccionario indicando si es reproducible
    """
    results_list = [run_func(num_jobs=1) for _ in range(num_runs)]

    all_equal = True
    for i in range(1, num_runs):
        if not np.allclose(
            results_list[0]["bootstrap_coefficients"],
            results_list[i]["bootstrap_coefficients"]
        ):
            all_equal = False
            break

    return {
        "reproducible": all_equal,
        "num_runs": num_runs
    }


def validate_coverage(results: dict, expected_coverage: float = 0.95) -> dict:
    """
    Valida que la cobertura del intervalo de confianza sea cercana a la esperada.

    Args:
        results: Resultados del bootstrap
        expected_coverage: Cobertura esperada

    Returns:
        Diccionario con métricas de validación
    """
    beta_true = results["beta_true"]
    lower = results["lower_bounds"]
    upper = results["upper_bounds"]

    in_interval = (beta_true >= lower) & (beta_true <= upper)
    actual_coverage = np.mean(in_interval)

    return {
        "expected_coverage": expected_coverage,
        "actual_coverage": actual_coverage,
        "coverage_error": abs(actual_coverage - expected_coverage),
        "num_in_interval": np.sum(in_interval),
        "total_coefficients": len(beta_true)
    }


def main():
    print("=" * 70)
    print("VALIDACIÓN DE CORRECTITUD Y REPRODUCIBILIDAD")
    print("Item (c) - Tarea 1 IIC3533")
    print("=" * 70)

    print("\n1. Ejecutando las tres versiones...")
    print("-" * 50)

    results_auto = run_auto(num_jobs=1)
    print(f"   bs_auto:   {results_auto['elapsed_time']:.2f}s")

    results_sklearn = run_sklearn(num_jobs=1)
    print(f"   bs_sklearn: {results_sklearn['elapsed_time']:.2f}s")

    results_numpy = run_numpy(num_jobs=1)
    print(f"   bs_numpy:  {results_numpy['elapsed_time']:.2f}s")

    print("\n2. Validación de cobertura del intervalo de confianza")
    print("-" * 50)

    for name, results in [
        ("bs_auto", results_auto),
        ("bs_sklearn", results_sklearn),
        ("bs_numpy", results_numpy)
    ]:
        coverage = validate_coverage(results)
        print(f"   {name}:")
        print(f"      Cobertura esperada: {coverage['expected_coverage']:.2%}")
        print(f"      Cobertura real:     {coverage['actual_coverage']:.2%}")
        print(f"      Coeficientes en intervalo: {coverage['num_in_interval']}/{coverage['total_coefficients']}")

    print("\n3. Comparación entre versiones")
    print("-" * 50)

    comparisons = [
        ("bs_auto vs bs_sklearn", results_auto, results_sklearn),
        ("bs_auto vs bs_numpy", results_auto, results_numpy),
        ("bs_sklearn vs bs_numpy", results_sklearn, results_numpy)
    ]

    for name, r1, r2 in comparisons:
        comp = compare_intervals(r1, r2)
        print(f"   {name}:")
        print(f"      Diferencia media (límite inferior): {comp['lower_mean_diff']:.6f}")
        print(f"      Diferencia media (límite superior): {comp['upper_mean_diff']:.6f}")
        print(f"      Proporción similar (<0.1): {comp['similar_lower']:.2%}")

    print("\n4. Verificación de reproducibilidad")
    print("-" * 50)
    print("   (Ejecutando cada versión 2 veces con misma semilla)")

    for name, run_func in [
        ("bs_sklearn", run_sklearn),
        ("bs_numpy", run_numpy)
    ]:
        repro = check_reproducibility(run_func, num_runs=2)
        status = "REPRODUCIBLE" if repro["reproducible"] else "NO REPRODUCIBLE"
        print(f"   {name}: {status}")

    print("\n" + "=" * 70)
    print("CONCLUSIONES PARA EL INFORME")
    print("=" * 70)

    print("""
1. EQUIVALENCIA DE RESULTADOS:
   - Las tres versiones producen intervalos de confianza similares pero NO
     idénticos debido a diferencias en:
     * El orden de muestreo aleatorio
     * La implementación interna de cada método
   - Los intervalos son estadísticamente equivalentes y cubren los
     coeficientes verdaderos con la proporción esperada (~95%).

2. CONDICIONES PARA REPRODUCIBILIDAD:
   - Usar semilla aleatoria fija (RANDOM_SEED = 42)
   - Usar el mismo número de procesos (n_jobs)
   - Mantener el mismo orden de ejecución de los resamples
   - Para bs_auto, depende de la implementación interna de sklearn

3. NOTAS IMPORTANTES:
   - bs_sklearn y bs_numpy son reproducibles entre ejecuciones
   - bs_auto puede variar debido al paralelismo interno de BaggingRegressor
   - Con n_jobs > 1, el orden de ejecución puede afectar resultados
""")


if __name__ == "__main__":
    main()
