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

**Requisitos:** Python 3.12 y las versiones exactas fijadas en [`requirements.txt`](requirements.txt)
(pandas 2.3.3, numpy 2.0.2, scipy 1.15.3, matplotlib 3.9.4, scikit-learn 1.6.1 y jupyter 1.1.1).
Se recomienda instalarlas en un entorno virtual, para no mezclarlas con otras versiones del
sistema:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Aun con las mismas versiones, el Random Forest puede variar en el tercer decimal entre sistemas
operativos (macOS y Linux usan librerías de cálculo numérico distintas), así que al volver a
ejecutar un notebook en otro equipo pueden aparecer diferencias mínimas respecto de las cifras
del texto.

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
- [`modelamiento.ipynb`](notebooks/modelamiento.ipynb) — pasos 2 y 3 del laboratorio: ajusta los
  hiperparámetros con validación cruzada (`GridSearchCV` y `RandomizedSearchCV`), y entrena y
  evalúa una regresión logística y un Random Forest para clasificar el abandono, y una regresión
  lineal y un Random Forest para estimar el saldo.
- [`clustering.ipynb`](notebooks/clustering.ipynb) — actividad 2.3.2: agrupa a los clientes con
  K-Means (K elegido con el método del codo y el coeficiente de Silhouette, con 300 inicializaciones
  para que el resultado no dependa de la semilla), visualiza los grupos con PCA e interpreta el
  perfil de negocio de cada segmento.

Los tres notebooks son autocontenidos: cada uno reconstruye lo que necesita desde el CSV crudo.

## Resultados

| Problema | Modelo final | Resultado en prueba | Línea base |
|---|---|---|---|
| Clasificación (`Exited`) | Random Forest (umbral 0,35) | ROC-AUC 0,862 · PR-AUC 0,703 · recall 0,60 · precisión 0,65 · F1 0,62 | recall 0 (predice siempre "se mantiene") |
| Regresión (`Balance`) | Random Forest | MAE 41.230 · R² 0,33 | MAE 54.786 (predice siempre la mediana) |

En los dos problemas, los hiperparámetros, el modelo y (en clasificación) el umbral se eligieron
con validación cruzada de 5 partes sobre el conjunto de entrenamiento; la prueba se usó una sola
vez, al final. En clasificación, el Random Forest supera a la regresión logística en las 5
particiones (ROC-AUC 0,859 contra 0,848; prueba t pareada corregida, p = 0,02), pero por poco: lo
que realmente define cuántos abandonos se detectan es el umbral. Los modelos se entrenan sin
`class_weight='balanced'` para que sus probabilidades queden calibradas.

El R² de regresión está inflado por un probable sesgo de muestreo (ningún cliente de Alemania
tiene saldo 0): sin la variable de país, el R² del Random Forest cae de 0,32 a 0,13 en validación
cruzada, y la estimación honesta es que el modelo explica entre un 13% y un 21% del saldo. El
detalle e interpretación de cada resultado está en [`modelamiento.ipynb`](notebooks/modelamiento.ipynb).
