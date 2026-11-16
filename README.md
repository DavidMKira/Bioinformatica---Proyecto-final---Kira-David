# Clasificación Multiclase y Triaje Clínico de Radiografías de Tórax mediante Aprendizaje Profundo

**Autor:** David Kira  
**Asignatura:** Bioinformática (4.° Año — UNSTA)  
**Entorno de ejecución:** Google Colaboratory (GPU NVIDIA T4) con PyTorch y Scikit-learn  

---

## Descripción del Proyecto

Este repositorio contiene el desarrollo completo del proyecto final de asignatura, centrado en la detección y clasificación automatizada de patologías torácicas sobre radiografías frontales provenientes del catálogo público **ChestX-ray8 / ChestX-ray14** del *National Institutes of Health (NIH) Clinical Center* (2017; Wang et al., 2017).

El sistema implementa una red convolucional **DenseNet-121** (Huang et al., 2017) adaptada mediante ajuste fino (*fine-tuning*) sobre tres categorías del catálogo:
- **`0` — Sin hallazgos (`No Finding`):** radiografías en las que no se identificaron las patologías contempladas en *ChestX-ray14*.
- **`1` — Cardiomegalia (`Cardiomegaly`)**
- **`2` — Derrame Pleural (`Effusion`)**

Además de la clasificación multiclase directa, la evaluación desglosa el comportamiento del modelo en dos etapas:
1. **Etapa 1 (Triaje Binario — Sin hallazgos vs. Patología):** detección de estudios con hallazgos patológicos a partir de probabilidades calibradas por temperatura ($T = 1.121$), comparando el corte estándar (`0.50`), el índice J de Youden (`0.58`) y un umbral de tamizaje definido en validación (`0.35`, $\text{Val} \ge 90\%$).
2. **Etapa 2 Condicional (Diagnóstico Diferencial — Cardiomegalia vs. Derrame Pleural):** discriminación específica entre ambas afecciones evaluada de forma condicional sobre los `140` casos realmente patológicos del conjunto de prueba.

---

## Estructura del Repositorio

```text
├── Bioinformatica - Proyecto final - Kira David.ipynb   # Cuaderno principal (informe, código y gráficos)
├── data/
│   ├── Data_Entry_2017_v2020.csv                        # Catálogo original del NIH (112.120 registros)
│   ├── multiclase_labels.csv                            # Muestreo inicial de 2.400 radiografías (800 por clase)
│   └── multiclase_labels_limpio.csv                     # Catálogo depurado final (1.739 placas, 1.518 pacientes)
├── checkpoints/
│   └── best_multiclase_model.pth                        # Pesos entrenados de DenseNet-121 (mejor F1-Macro en validación)
├── figuras_presentacion/                                # Gráficos exportados en alta resolución (300 DPI)
│   ├── 01_comparativa_evolucion_modelo.png
│   ├── 02_analisis_multiclase_completo.png
│   ├── 03_triaje_etapa1_detallado.png
│   ├── 04_etapa2_diagnostico_diferencial.png
│   └── 05_galeria_predicciones_clinicas.png
├── papers/                                              # Artículos científicos de referencia utilizados en el informe
└── README.md
```

---

## Metodología

1. **Curación del catálogo y exclusión mutua:** A partir del registro original del NIH (`Data_Entry_2017_v2020.csv`), se eliminaron las radiografías donde coexistían ambos hallazgos (`Cardiomegaly|Effusion`) y los pacientes con diagnósticos cambiantes entre estudios sucesivos, conformando un conjunto limpio de **1.739 radiografías** (`759` sin hallazgos, `407` cardiomegalias y `573` derrames pleurales) pertenecientes a **1.518 pacientes únicos**.
2. **Preprocesamiento preservando la anatomía torácica:** Reescalado a $224 \times 224$ píxeles con interpolación de Lanczos sin filtrado de mediana previo (para conservar los bordes costales y la línea pleural) y aumentación de datos mediante rotaciones y traslaciones suaves ($\pm 7^\circ$), evitando la inversión horizontal (`RandomHorizontalFlip`) para no alterar la orientación izquierda de la silueta cardíaca.
3. **Particionado estratificado por paciente (`StratifiedGroupKFold`):** División en entrenamiento ($71.4\%$, $n = 1.241$), validación ($14.3\%$, $n = 249$) y prueba ($14.3\%$, $n = 249$) agrupando por `Patient_ID`, con intersección nula de pacientes entre subconjuntos ($0\%$ de fuga de datos).
4. **Ajuste fino y regularización a 40 épocas:** Descongelamiento selectivo de `denseblock3`, `transition3`, `denseblock4` y `norm5` en `DenseNet-121` con tasas de aprendizaje diferenciadas en `AdamW` (`1e-4` para los bloques convolucionales y `1e-3` para el cabezal clasificador), regularización mediante `Dropout(p=0.35)`, decaimiento de pesos $L_2$ (`1e-2`), *label smoothing* (`0.05`) y entropía cruzada ponderada por clase.

---

## Resumen de Resultados (Conjunto de Prueba, $n = 249$)

| Etapa / Dimensión | Métrica | Modelo Inicial (Red Congelada / CSV sin depurar) | Modelo Final (Fine-Tuning / CSV depurado) | Diferencia |
| :--- | :--- | :---: | :---: | :---: |
| **Multiclase (3 vías, $n=249$)** | Accuracy Global | `52.17%` | **`69.08%`** | `+16.91 pts` |
| **Multiclase (3 vías, $n=249$)** | F1-Score Macro | `0.5100` | **`0.6800`** | `+0.1700` |
| **Multiclase (3 vías, $n=249$)** | ROC-AUC Macro (OvR) | — | **`0.8654`** | — |
| **Por categoría ($n=249$)** | F1 `Sin hallazgos (0)` | `0.47` (Recall `0.36`) | **`0.70` (Recall `0.72`)** | `+0.23` |
| **Por categoría ($n=249$)** | F1 `Cardiomegalia (1)` | `0.47` (Precisión `0.54`) | **`0.62` (Precisión `0.81`)** | `+0.15` |
| **Por categoría ($n=249$)** | F1 `Derrame Pleural (2)` | `0.59` (Precisión `0.46`) | **`0.71` (Precisión `0.65`)** | `+0.12` |
| **Etapa 1: Triaje ($n=249$)** | ROC-AUC Binario | `0.7254` | **`0.8231`** | `+0.0977` |
| **Etapa 1: Triaje ($n=249$)** | Corte Youden (`0.58`) | Esp. `26.40%` (corte `0.57`) | **Esp. `72.48%`** (Sens. `75.00%`, Acc `73.90%`) | `+46.08 pts Esp.` |
| **Etapa 1: Triaje ($n=249$)** | Corte Estándar (`0.50`) | Esp. `16.00%` (corte `0.50`) | **Esp. `70.64%`** (Sens. `77.14%`, Acc `74.30%`) | `+54.64 pts Esp.` |
| **Etapa 1: Triaje ($n=249$)** | Corte Tamizaje (`0.35`, Val $\ge 90\%$) | Esp. `16.00%` (corte `0.50`) | **Esp. `64.22%`** (Sens. Test `83.57%`, Acc `75.10%`) | `+48.22 pts Esp.` |
| **Etapa 2 Condicional ($n=140$)** | Cardio vs. Derrame (solo patológicos) | — | **ROC-AUC `0.9521` (Acc. cond. `88.00%`, F1 `0.87`)** | — |

---

## Requisitos e Instrucciones de Ejecución

### Dependencias principales
El proyecto está preparado para ejecutarse en **Google Colaboratory** (Python 3.10+ con soporte de GPU CUDA):

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow
```

### Cómo reproducir el proyecto en Google Colab
1. Abrir el archivo `Bioinformatica - Proyecto final - Kira David.ipynb` en Google Colaboratory.
2. Colocar en Google Drive (o ajustar las constantes `Path` al inicio de cada bloque) los archivos:
   - `multiclase_labels.csv` y `multiclase_labels_limpio.csv`
   - `multiclase_dataset.zip` (radiografías preprocesadas a $224 \times 224$ px)
   - `best_multiclase_model.pth` (opcional, si se desea ejecutar directamente la evaluación de la **Parte 05** sin reentrenar las 40 épocas de la **Parte 04**).
3. Ejecutar secuencialmente las celdas desde la **Parte 02** hasta la sección de **Visualización de Resultados**.

---

## Referencias Bibliográficas

- Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. En *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 4700–4708). IEEE. https://doi.org/10.1109/CVPR.2017.243
- Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., van der Laak, J. A. W. M., van Ginneken, B., & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, *42*, 60–88. https://doi.org/10.1016/j.media.2017.07.005
- National Institutes of Health Clinical Center. (2017). *NIH Chest X-rays (ChestX-ray14) dataset* [Conjunto de datos]. National Institutes of Health. https://nihcc.app.box.com/v/ChestXray-NIHCC
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. En H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, & R. Garnett (Eds.), *Advances in Neural Information Processing Systems* (Vol. 32, pp. 8024–8035). Curran Associates, Inc. https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*(85), 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html
- Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. En *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2097–2106). IEEE. https://doi.org/10.1109/CVPR.2017.369
