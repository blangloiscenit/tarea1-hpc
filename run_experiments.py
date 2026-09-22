"""
Script de experimentos para partes e), f), g), h), i) de la Tarea 1.
IIC3533 - Computación de Alto Rendimiento

Ejecuta benchmarks de las tres versiones de bootstrap variando:
- Número de procesos (p)
- Número de threads internos (t) para la parte i)

Genera gráficos y tablas con los resultados.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_info, threadpool_limits

from data_generator import generate_synthetic_data, compute_confidence_intervals, NUM_RESAMPLES, RANDOM_SEED


def get_num_logical_cores() -> int:
    """Retorna el número de cores lógicos del sistema."""
    return os.cpu_count() or 1


def print_threadpool_info():
    """Imprime información sobre los threadpools activos (NumPy/BLAS)."""
    print("\n" + "=" * 60)
    print("INFORMACIÓN DE THREADPOOL (Parte e)")
    print("=" * 60)
    info_list = threadpool_info()
    if not info_list:
        print("  No se detectaron threadpools activos.")
    for info in info_list:
        print(f"  - {info.get('user_api', 'N/A')}: {info.get('num_threads', 'N/A')} threads")
        print(f"    Librería: {info.get('prefix', 'N/A')} ({info.get('filepath', 'N/A')})")
    print("=" * 60 + "\n")


def run_bs_numpy(X, y, num_jobs, num_threads=None):
    """Ejecuta bootstrap con NumPy, opcionalmente limitando threads."""
    from bs_numpy import bootstrap_with_numpy

    if num_threads is not None:
        with threadpool_limits(limits=num_threads):
            start = time.perf_counter()
            coeffs = bootstrap_with_numpy(X, y, num_jobs=num_jobs)
            elapsed = time.perf_counter() - start
    else:
        start = time.perf_counter()
        coeffs = bootstrap_with_numpy(X, y, num_jobs=num_jobs)
        elapsed = time.perf_counter() - start

    return elapsed, coeffs


def run_bs_sklearn(X, y, num_jobs):
    """Ejecuta bootstrap con sklearn LinearRegression."""
    from bs_sklearn import bootstrap_with_sklearn

    start = time.perf_counter()
    coeffs = bootstrap_with_sklearn(X, y, num_jobs=num_jobs)
    elapsed = time.perf_counter() - start

    return elapsed, coeffs


def run_bs_auto(X, y, num_jobs):
    """Ejecuta bootstrap con BaggingRegressor."""
    from bs_auto import bootstrap_with_bagging

    start = time.perf_counter()
    coeffs = bootstrap_with_bagging(X, y, num_jobs=num_jobs)
    elapsed = time.perf_counter() - start

    return elapsed, coeffs


def run_scaling_experiments(X, y, p_max, num_repetitions=1):
    """
    Ejecuta experimentos de escalabilidad (partes f, g, h).

    Args:
        X, y: Datos de entrada
        p_max: Número máximo de procesos
        num_repetitions: Número de repeticiones por configuración

    Returns:
        Diccionario con tiempos para cada versión y cada p
    """
    results = {
        "bs_numpy": {},
        "bs_sklearn": {},
        "bs_auto": {},
        "p_values": list(range(1, p_max + 1))
    }

    for p in range(1, p_max + 1):
        print(f"\n--- Ejecutando con p = {p} procesos ---")

        # bs_numpy
        times_numpy = []
        for rep in range(num_repetitions):
            elapsed, _ = run_bs_numpy(X, y, num_jobs=p)
            times_numpy.append(elapsed)
            print(f"  bs_numpy (rep {rep+1}): {elapsed:.2f}s")
        results["bs_numpy"][p] = np.mean(times_numpy)

        # bs_sklearn
        times_sklearn = []
        for rep in range(num_repetitions):
            elapsed, _ = run_bs_sklearn(X, y, num_jobs=p)
            times_sklearn.append(elapsed)
            print(f"  bs_sklearn (rep {rep+1}): {elapsed:.2f}s")
        results["bs_sklearn"][p] = np.mean(times_sklearn)

        # bs_auto
        times_auto = []
        for rep in range(num_repetitions):
            elapsed, _ = run_bs_auto(X, y, num_jobs=p)
            times_auto.append(elapsed)
            print(f"  bs_auto (rep {rep+1}): {elapsed:.2f}s")
        results["bs_auto"][p] = np.mean(times_auto)

    return results


def run_thread_experiments(X, y, p_max):
    """
    Ejecuta experimentos variando p y t con p*t <= p_max (parte i).

    Args:
        X, y: Datos de entrada
        p_max: Número máximo de cores

    Returns:
        Diccionario con tiempos para cada combinación (p, t)
    """
    results = {"combinations": []}

    for p in range(1, p_max + 1):
        for t in range(1, (p_max // p) + 1):
            if p * t > p_max:
                continue

            print(f"  Ejecutando p={p}, t={t} (p*t={p*t})...")
            elapsed, _ = run_bs_numpy(X, y, num_jobs=p, num_threads=t)

            results["combinations"].append({
                "p": p,
                "t": t,
                "p_times_t": p * t,
                "time": elapsed
            })
            print(f"    Tiempo: {elapsed:.2f}s")

    return results


def calculate_metrics(results):
    """
    Calcula speedup S(p), eficiencia E(p) y overhead T_o(p).

    S(p) = T(1) / T(p)
    E(p) = S(p) / p
    T_o(p) = p * T(p) - T(1)
    """
    metrics = {}

    for version in ["bs_numpy", "bs_sklearn", "bs_auto"]:
        T1 = results[version][1]
        metrics[version] = {
            "T": {},
            "S": {},
            "E": {},
            "T_o": {}
        }

        for p in results["p_values"]:
            Tp = results[version][p]
            Sp = T1 / Tp
            Ep = Sp / p
            To = p * Tp - T1

            metrics[version]["T"][p] = Tp
            metrics[version]["S"][p] = Sp
            metrics[version]["E"][p] = Ep
            metrics[version]["T_o"][p] = To

    return metrics


def plot_scaling_results(metrics, p_values, output_dir="results"):
    """Genera gráficos de T(p), S(p), E(p) y T_o(p)."""
    os.makedirs(output_dir, exist_ok=True)

    versions = ["bs_numpy", "bs_sklearn", "bs_auto"]
    colors = {"bs_numpy": "blue", "bs_sklearn": "orange", "bs_auto": "green"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # T(p) - Tiempo de ejecución
    ax = axes[0, 0]
    for v in versions:
        times = [metrics[v]["T"][p] for p in p_values]
        ax.plot(p_values, times, 'o-', label=v, color=colors[v])
    ax.set_xlabel("Número de procesos (p)")
    ax.set_ylabel("Tiempo T(p) [s]")
    ax.set_title("Tiempo de ejecución vs. procesos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # S(p) - Speedup
    ax = axes[0, 1]
    ideal_speedup = p_values
    ax.plot(p_values, ideal_speedup, 'k--', label="Ideal", linewidth=2)
    for v in versions:
        speedups = [metrics[v]["S"][p] for p in p_values]
        ax.plot(p_values, speedups, 'o-', label=v, color=colors[v])
    ax.set_xlabel("Número de procesos (p)")
    ax.set_ylabel("Speedup S(p)")
    ax.set_title("Speedup vs. procesos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # E(p) - Eficiencia
    ax = axes[1, 0]
    ax.axhline(y=1.0, color='k', linestyle='--', label="Ideal", linewidth=2)
    for v in versions:
        efficiencies = [metrics[v]["E"][p] for p in p_values]
        ax.plot(p_values, efficiencies, 'o-', label=v, color=colors[v])
    ax.set_xlabel("Número de procesos (p)")
    ax.set_ylabel("Eficiencia E(p)")
    ax.set_title("Eficiencia vs. procesos")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # T_o(p) - Overhead
    ax = axes[1, 1]
    for v in versions:
        overheads = [metrics[v]["T_o"][p] for p in p_values]
        ax.plot(p_values, overheads, 'o-', label=v, color=colors[v])
    ax.set_xlabel("Número de procesos (p)")
    ax.set_ylabel("Overhead T_o(p) [s]")
    ax.set_title("Overhead vs. procesos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scaling_results.png"), dpi=150)
    plt.close()
    print(f"Gráfico guardado en {output_dir}/scaling_results.png")


def plot_thread_results(thread_results, p_max, output_dir="results"):
    """Genera heatmap de tiempos para combinaciones (p, t)."""
    os.makedirs(output_dir, exist_ok=True)

    # Crear matriz para heatmap
    matrix = np.full((p_max, p_max), np.nan)

    for combo in thread_results["combinations"]:
        p, t, time_val = combo["p"], combo["t"], combo["time"]
        matrix[p-1, t-1] = time_val

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap="viridis_r", origin="lower", aspect="auto")
    ax.set_xlabel("Número de threads (t)")
    ax.set_ylabel("Número de procesos (p)")
    ax.set_title(f"Tiempo de ejecución para combinaciones (p, t)\ncon p × t ≤ {p_max}")
    ax.set_xticks(range(p_max))
    ax.set_xticklabels(range(1, p_max + 1))
    ax.set_yticks(range(p_max))
    ax.set_yticklabels(range(1, p_max + 1))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Tiempo [s]")

    # Anotar valores
    for combo in thread_results["combinations"]:
        p, t, time_val = combo["p"], combo["t"], combo["time"]
        ax.text(t-1, p-1, f"{time_val:.1f}", ha="center", va="center",
                color="white" if time_val > np.nanmedian(matrix) else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thread_combinations.png"), dpi=150)
    plt.close()
    print(f"Gráfico guardado en {output_dir}/thread_combinations.png")


def generate_tables(metrics, p_values, output_dir="results"):
    """Genera tablas en formato texto/markdown."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "tables.md"), "w", encoding="utf-8") as f:
        f.write("# Resultados de Experimentos de Escalabilidad\n\n")

        # Tabla de tiempos
        f.write("## Tiempos de ejecución T(p) [segundos]\n\n")
        f.write("| p | bs_numpy | bs_sklearn | bs_auto |\n")
        f.write("|---|----------|------------|----------|\n")
        for p in p_values:
            f.write(f"| {p} | {metrics['bs_numpy']['T'][p]:.2f} | ")
            f.write(f"{metrics['bs_sklearn']['T'][p]:.2f} | ")
            f.write(f"{metrics['bs_auto']['T'][p]:.2f} |\n")

        # Tabla de speedup
        f.write("\n## Speedup S(p) = T(1)/T(p)\n\n")
        f.write("| p | bs_numpy | bs_sklearn | bs_auto | Ideal |\n")
        f.write("|---|----------|------------|---------|-------|\n")
        for p in p_values:
            f.write(f"| {p} | {metrics['bs_numpy']['S'][p]:.2f} | ")
            f.write(f"{metrics['bs_sklearn']['S'][p]:.2f} | ")
            f.write(f"{metrics['bs_auto']['S'][p]:.2f} | {p:.2f} |\n")

        # Tabla de eficiencia
        f.write("\n## Eficiencia E(p) = S(p)/p\n\n")
        f.write("| p | bs_numpy | bs_sklearn | bs_auto |\n")
        f.write("|---|----------|------------|----------|\n")
        for p in p_values:
            f.write(f"| {p} | {metrics['bs_numpy']['E'][p]:.2f} | ")
            f.write(f"{metrics['bs_sklearn']['E'][p]:.2f} | ")
            f.write(f"{metrics['bs_auto']['E'][p]:.2f} |\n")

        # Tabla de overhead
        f.write("\n## Overhead T_o(p) = p × T(p) - T(1) [segundos]\n\n")
        f.write("| p | bs_numpy | bs_sklearn | bs_auto |\n")
        f.write("|---|----------|------------|----------|\n")
        for p in p_values:
            f.write(f"| {p} | {metrics['bs_numpy']['T_o'][p]:.2f} | ")
            f.write(f"{metrics['bs_sklearn']['T_o'][p]:.2f} | ")
            f.write(f"{metrics['bs_auto']['T_o'][p]:.2f} |\n")

    print(f"Tablas guardadas en {output_dir}/tables.md")


def save_results(results, metrics, thread_results, output_dir="results"):
    """Guarda resultados en formato JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Convertir claves int a string para JSON
    def convert_keys(d):
        if isinstance(d, dict):
            return {str(k): convert_keys(v) for k, v in d.items()}
        return d

    data = {
        "scaling_results": convert_keys(results),
        "metrics": convert_keys(metrics),
        "thread_results": thread_results,
        "system_info": {
            "num_cores": get_num_logical_cores(),
            "threadpool_info": [str(info) for info in threadpool_info()]
        }
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(f"Resultados guardados en {output_dir}/results.json")


def main():
    """Función principal que ejecuta todos los experimentos."""
    print("=" * 60)
    print("EXPERIMENTOS DE ESCALABILIDAD - TAREA 1 IIC3533")
    print("=" * 60)

    # Información del sistema
    p_max = get_num_logical_cores()
    print(f"\nNúmero de cores lógicos: {p_max}")

    # Parte e) - Información de threadpool
    print_threadpool_info()

    # Cargar datos
    print("Generando/cargando datos sintéticos...")
    X, y, beta_true = generate_synthetic_data()
    print(f"Dimensiones: X={X.shape}, y={y.shape}")

    # Parte f) - Experimentos de escalabilidad
    print("\n" + "=" * 60)
    print("PARTE f) - EXPERIMENTOS DE ESCALABILIDAD")
    print("=" * 60)
    results = run_scaling_experiments(X, y, p_max, num_repetitions=1)

    # Partes g) y h) - Calcular métricas
    print("\n" + "=" * 60)
    print("PARTES g) y h) - CÁLCULO DE MÉTRICAS")
    print("=" * 60)
    metrics = calculate_metrics(results)

    # Parte i) - Experimentos con threads
    print("\n" + "=" * 60)
    print("PARTE i) - EXPERIMENTOS VARIANDO p Y t")
    print("=" * 60)
    thread_results = run_thread_experiments(X, y, p_max)

    # Guardar resultados
    print("\n" + "=" * 60)
    print("GENERANDO GRÁFICOS Y TABLAS")
    print("=" * 60)

    output_dir = "results"
    plot_scaling_results(metrics, results["p_values"], output_dir)
    plot_thread_results(thread_results, p_max, output_dir)
    generate_tables(metrics, results["p_values"], output_dir)
    save_results(results, metrics, thread_results, output_dir)

    print("\n" + "=" * 60)
    print("EXPERIMENTOS COMPLETADOS")
    print("=" * 60)

    # Resumen
    print("\nResumen de tiempos con p=1:")
    for v in ["bs_numpy", "bs_sklearn", "bs_auto"]:
        print(f"  {v}: {metrics[v]['T'][1]:.2f}s")

    print(f"\nMejor tiempo con p={p_max}:")
    for v in ["bs_numpy", "bs_sklearn", "bs_auto"]:
        print(f"  {v}: {metrics[v]['T'][p_max]:.2f}s (speedup={metrics[v]['S'][p_max]:.2f}x)")


if __name__ == "__main__":
    main()
