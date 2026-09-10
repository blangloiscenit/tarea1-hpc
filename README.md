# Tarea 1 - IIC3533 Computación de Alto Rendimiento

Implementación de Bootstrap para Regresión Lineal con paralelismo de tareas usando `joblib`.

## Estructura del Proyecto

```
tarea1-hpc/
├── data/                    # Datos generados localmente (no incluidos en repo)
├── data_generator.py        # Generación de datos sintéticos
├── bs_auto.py               # Bootstrap con BaggingRegressor
├── bs_sklearn.py            # Bootstrap con joblib + LinearRegression
├── bs_numpy.py              # Bootstrap con joblib + NumPy puro
├── validate_results.py      # Validación de correctitud (item c)
├── requirements.txt         # Dependencias
└── README.md
```

> **Nota:** Los datos (`data/`) no se incluyen en el repositorio debido a su tamaño (~230 MB). Se generan localmente con semilla fija para garantizar reproducibilidad.

## Instalación

```bash
# Opción 1: Con pip
pip install -r requirements.txt

# Opción 2: Con conda (recomendado)
conda create -n tarea1-hpc python=3.13 -y
conda activate tarea1-hpc
conda install numpy matplotlib joblib threadpoolctl scikit-learn -y
```

## Uso

### Paso 1: Generar datos sintéticos (obligatorio)

**Ejecutar primero antes de cualquier otro script:**

```bash
python data_generator.py
```

Genera y guarda en `data/`:
- `X.npy`: Matriz de diseño N×(k+1) con N=100,000 y k=300 (~230 MB)
- `y.npy`: Vector de salidas con ruido gaussiano
- `beta_true.npy`: Coeficientes verdaderos β*

Los datos se generan con `RANDOM_SEED = 42` para reproducibilidad.

### Item (a): Datos sintéticos

El script `data_generator.py` implementa:
1. Muestreo de k+1 coeficientes β* desde N(0,1)
2. Generación de matriz X con columna de unos
3. Cálculo de y = Xβ* + ruido N(0,1)

### Item (b): Ejecutar las tres versiones de bootstrap

Cada versión acepta el número de procesos como argumento:

```bash
# bs_auto: usa BaggingRegressor de sklearn
python bs_auto.py 4      # 4 procesos

# bs_sklearn: usa joblib.Parallel + LinearRegression
python bs_sklearn.py 4   # 4 procesos

# bs_numpy: usa joblib.Parallel + NumPy puro (más eficiente)
python bs_numpy.py 4     # 4 procesos
```

### Item (c): Validar correctitud y reproducibilidad

```bash
python validate_results.py
```

Ejecuta las tres versiones y compara:
- Cobertura del intervalo de confianza (esperado ~95%)
- Diferencias entre versiones
- Reproducibilidad entre ejecuciones

## Parámetros del Experimento

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| N | 100,000 | Número de observaciones |
| k | 300 | Variables de entrada |
| B | 48 | Número de resamples bootstrap |
| SEED | 42 | Semilla aleatoria |

## Resultados de Validación (Item c)

### Tiempos de ejecución (1 proceso)

| Versión | Tiempo |
|---------|--------|
| bs_auto | ~65s |
| bs_sklearn | ~63s |
| bs_numpy | ~16s |

### Cobertura del intervalo de confianza

Las tres versiones producen intervalos que cubren ~93-95% de los coeficientes verdaderos, consistente con el nivel de confianza del 95%.

### Equivalencia entre versiones

- `bs_sklearn` y `bs_numpy` producen **resultados idénticos**
- `bs_auto` produce resultados similares (diferencia media < 0.002)

### Reproducibilidad

- ✅ `bs_sklearn`: Reproducible con misma semilla
- ✅ `bs_numpy`: Reproducible con misma semilla
- ⚠️ `bs_auto`: Puede variar con n_jobs > 1

## Condiciones para Reproducibilidad

1. Usar semilla aleatoria fija (`RANDOM_SEED = 42`)
2. Usar el mismo número de procesos (`n_jobs`)
3. Ejecutar en el mismo orden los resamples
4. Misma versión de NumPy/sklearn

## Autores

- Grupo Tarea 1 - IIC3533 2026-2
