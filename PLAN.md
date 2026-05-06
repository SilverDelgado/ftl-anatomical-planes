# Proyecto 6 — FTL de comparación por plano anatómico

**Equipo:** María Vázquez López, Pablo Jesús Delgado Muñoz, Paola Gambero Díaz  
**Asignatura:** Aprendizaje Federado — Grado en Ciberseguridad e IA, UMA

---

## 1. Descripción del problema

**Tarea clínica:** clasificación binaria EM vs control usando imágenes MRI FLAIR 2D.

**Tipo de AF:** Federated Transfer Learning (FTL). La tarea y la modalidad (FLAIR) son
idénticas en todos los hospitales, pero el **plano anatómico** cambia entre clientes.
Esto introduce un cambio de dominio visual severo: un corte axial y uno sagital del mismo
cerebro tienen apariencias completamente distintas para una CNN.

**Por qué es FTL y no HFL:** en HFL clásico los clientes comparten el mismo espacio de
características P(X). Aquí P(X) cambia entre clientes (imágenes axiales vs coronales vs
sagitales), aunque el espacio de etiquetas Y sea el mismo. Según la definición del
Tema 4, esto es FTL con cambio de dominio visual.

---

## 2. Clientes y datos

| Cliente | Hospital | Plano | Train slices | Test slices | Resolución original |
|---------|----------|-------|-------------|-------------|---------------------|
| 1       | H1       | Axial    | 3 458 | ~866  | 128×218 px |
| 2       | H2       | Coronal  | 4 142 | ~1036 | 182×182 px |
| 3       | H3       | Sagital  | 3 458 | ~866  | 218×182 px |
| 4       | H4       | Mixto    | 3 583 | ~897  | variada    |

Todas las imágenes se redimensionan a **224×224 px** antes de entrar al modelo,
eliminando las diferencias de resolución como fuente de variación artificial.

**Distribución de clases:** cada hospital tiene ~23 pacientes MS y ~15 control.
La heterogeneidad está en P(X), no en P(Y|X).

---

## 3. Modelo base

**ResNet-18 preentrenado en ImageNet** (torchvision).  
Modificación: capa FC final → `Linear(512, 1)` + `BCEWithLogitsLoss`.  
Las imágenes en escala de grises se replican a 3 canales para compatibilidad.

Elección justificada por:
- Ligera (~11M parámetros), adecuada para el tamaño del dataset.
- Transfer learning de ImageNet útil cuando hay pocos datos por cliente.
- Estándar en clasificación de imágenes médicas 2D (Informe §7).

---

## 4. Algoritmos a comparar

### FedAvg (McMahan et al., 2017)
Baseline federado estándar. En cada ronda:
1. El servidor distribuye `w_global` a todos los clientes.
2. Cada cliente entrena localmente E épocas minimizando su pérdida local.
3. El servidor agrega con media ponderada por tamaño de dataset:

$$w^{(r+1)} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot w_k^{(r)}$$

### FedProx (Li et al., 2020)
Variante que añade un **término proximal** al objetivo local de cada cliente:

$$\min_w \; \mathcal{L}_k(w) + \frac{\mu}{2} \|w - w^{\text{global}}\|^2$$

El término `(μ/2)||w - w_global||²` ancla los pesos locales al modelo global,
reduciendo el **client drift** causado por la heterogeneidad entre dominios.
Con μ=0, FedProx se reduce a FedAvg.

**Hipótesis:** el cambio de plano anatómico produce client drift severo.
FedProx debería mitigarlo y producir mayor equidad entre hospitales.

---

## 5. Baselines de referencia

| Baseline       | Descripción | Rol |
|----------------|-------------|-----|
| **Local-only** | Cada hospital entrena su propio modelo sin colaborar | Cota inferior |
| **FedAvg**     | Federación estándar | Referencia federada |
| **FedProx**    | Federación con término proximal (μ=0.1) | Variante robusta |
| **Centralizado** | Todos los datos en un servidor único | Cota superior |

---

## 6. Métricas

| Métrica | Nivel | Justificación |
|---------|-------|---------------|
| **AUC-ROC** | Global | Robusta al desbalance; no depende de umbral fijo |
| **F1-score** | Por cliente | Equilibra precisión y recall |
| **Sensibilidad** (recall MS) | Por cliente | FN tiene mayor coste clínico que FP |
| **σ(AUC)** entre clientes | Sistema | Equidad: detecta si un plano queda rezagado |
| **Bytes/ronda** | Sistema | Coste de comunicación = 2 × nparams × 4 bytes |

---

## 7. Presupuesto experimental

```
K = 4 clientes    C = 1.0 (todos participan en cada ronda)
R = 50 rondas     E = 5 épocas locales
lr = 1e-4         batch = 32
μ_FedProx = 0.1   seeds = [42, 123, 456]
```

Se repiten FedAvg y FedProx con 3 semillas y se reporta media ± std.
Para que la comparación sea justa, ambos algoritmos usan exactamente el mismo presupuesto.

---

## 8. Estructura del proyecto

```
p6/
├── pyproject.toml          # dependencias (uv)
├── PLAN.md                 # este fichero
├── run_all.py              # punto de entrada principal
├── src/
│   ├── config.py           # rutas e hiperparámetros
│   ├── dataset.py          # MSDataset desde CSV
│   ├── model.py            # ResNet-18 modificado
│   ├── train_utils.py      # local_train + evaluate
│   ├── federated.py        # FedAvg y FedProx
│   └── baselines.py        # local-only y centralizado
├── notebooks/
│   └── analysis.ipynb      # visualización de resultados
└── results/                # JSONs generados por run_all.py
```

---

## 9. Cómo ejecutar

```bash
# Instalar dependencias
uv sync

# Verificar que todo funciona (2 rondas, 1 semilla)
uv run python run_all.py --dry-run

# Experimento completo
uv run python run_all.py

# Solo un experimento concreto
uv run python run_all.py --only fedavg
uv run python run_all.py --only fedprox
```

---

## 10. Riesgo principal

El cambio de dominio visual entre planos puede ser tan severo que promediar pesos
no tenga sentido semántico: features aprendidas en slices axiales son ortogonales
a las aprendidas en sagitales. Si esto ocurre, el modelo federado podría no superar
al local-only. FedProx mitiga este riesgo anclando cada cliente al global, pero
no lo elimina.

---

## 11. Decisiones de evaluación

- **No se reserva un hospital para test**: con solo 4 hospitales y planos tan distintos,
  excluir uno entero privaría al modelo de un dominio completo durante el entrenamiento
  (Informe §6).
- **Test local**: cada hospital usa su propia partición interna de test (20% pacientes).
- **Test global**: `global_test.csv` mezcla slices de los 4 planos, simula un servidor
  central con acceso a todos los dominios.
- **Agregación simple y ponderada**: se reportan ambas. La ponderada es más justa con el
  centralizado; la simple evita que H2 (coronal, mayor dataset) domine la media.
