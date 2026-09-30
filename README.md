# Ecommerce Customer — Predicción de Abandono y Saldo

Proyecto de la asignatura **MLY1101 Machine Learning (Duoc UC)** — *Laboratorio de Modelamiento
Supervisado: Regresión y Clasificación*.

## ¿De qué trata?

Analiza el [Ecommerce Customer Dataset](data/ecommerce_customer_data.csv) (10.000 clientes con
datos demográficos, saldo, productos contratados y actividad) y entrena modelos supervisados
para dos problemas:

- **Clasificación:** predecir si un cliente abandonará la empresa (`Exited`).
- **Regresión:** predecir el saldo de un cliente (`Balance`).

No es una aplicación: el resultado es el análisis en sí, expresado en notebooks de Jupyter con el
razonamiento detrás de cada decisión.

## Estructura del proyecto

```
Ecommerce-Customer-Dataset/
├── README.md                          — este archivo
├── requirements.txt                   — dependencias de Python del proyecto
├── data/
│   └── ecommerce_customer_data.csv    — dataset crudo (10.000 filas)
├── docs/
│   ├── 2.2.2_Laboratorio_Regresion_y_Clasificacion.pdf  — guía de este laboratorio
│   └── 2.3.2_Guia_Clustering.pdf                        — guía de la actividad de clustering
├── notebooks/
│   ├── analisis_exploratorio.ipynb    — Fase 2 CRISP-DM: comprensión de los datos (EDA)
│   ├── preprocesamiento.ipynb         — Fase 3 CRISP-DM: preparación de datos
│   ├── modelamiento.ipynb             — Fases 4 y 5 CRISP-DM: modelado y evaluación
│   └── clustering.ipynb               — Actividad 2.3.2: segmentación con K-Means y PCA
└── images/                            — figuras exportadas por los notebooks
```

## Cómo usarlo

**Requisitos:** Python 3.12 y las librerías listadas en [`requirements.txt`](requirements.txt)
(`pandas`, `numpy`, `matplotlib`, `scikit-learn` y `jupyter`). Para verificar la versión
instalada de una librería puntual, ejecutar `import <lib>; print(<lib>.__version__)` como se
indica en la primera celda de cada notebook.

```bash
pip install -r requirements.txt
```

**Ejecución:** abrir y correr cada notebook de punta a punta desde la carpeta `notebooks/`
(es la carpeta de trabajo que esperan las rutas relativas como `../data/...`):

```bash
cd notebooks
jupyter notebook
```

- [`analisis_exploratorio.ipynb`](notebooks/analisis_exploratorio.ipynb) — audita la calidad de
  los datos (incluida la detección del probable sesgo de muestreo de Alemania) y presenta 10
  hallazgos sobre el abandono y el saldo, cada uno con su gráfico.
- [`preprocesamiento.ipynb`](notebooks/preprocesamiento.ipynb) — agrega ingeniería de
  características y arma los pipelines de `scikit-learn` que dejan los datos listos para
  modelar.
- [`modelamiento.ipynb`](notebooks/modelamiento.ipynb) — pasos 2 y 3 del laboratorio: entrena y
  evalúa una regresión logística y un Random Forest para clasificar el abandono, y una regresión
  lineal y un Random Forest para estimar el saldo.
- [`clustering.ipynb`](notebooks/clustering.ipynb) — actividad 2.3.2: agrupa a los clientes con
  K-Means (K elegido con el método del codo y el coeficiente de Silhouette), visualiza los grupos
  con PCA e interpreta el perfil de negocio de cada segmento.

Los tres notebooks son autocontenidos: cada uno reconstruye lo que necesita desde el CSV crudo.

## Resultados

| Problema | Modelo final | Resultado en prueba | Línea base |
|---|---|---|---|
| Clasificación (`Exited`) | Random Forest | accuracy 0,835 · recall 0,64 · F1 0,61 | recall 0 (predice siempre "se mantiene") |
| Regresión (`Balance`) | Random Forest | MAE 41.576 · R² 0,31 | MAE 57.091 (predice siempre el promedio) |

El R² de regresión está inflado por un probable sesgo de muestreo (ningún cliente de Alemania
tiene saldo 0): sin la variable de país, el R² del Random Forest cae de 0,31 a 0,11, y la estimación
honesta es que el modelo explica entre un 11% y un 19% del saldo. El
detalle e interpretación de cada resultado está en [`modelamiento.ipynb`](notebooks/modelamiento.ipynb).
