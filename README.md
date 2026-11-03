# Aprendizaje Federado por Transferencia entre Planos Anatómicos

**[Leer el paper (PDF)](paper.pdf)**

Proyecto 6 de la asignatura de Aprendizaje Federado (Grado en Ciberseguridad e
Inteligencia Artificial, Universidad de Málaga).

El proyecto simula cuatro hospitales que clasifican imágenes de resonancia
magnética FLAIR como Esclerosis Múltiple o control. Todos resuelven la misma
tarea, pero cada hospital adquiere las imágenes en un plano de corte distinto
(axial, coronal, sagital y uno mixto), así que las distribuciones de entrada no
coinciden. Ese desajuste de dominio es lo que convierte el problema en
aprendizaje federado por transferencia y no en federado horizontal clásico.

Sobre ese escenario comparamos cuatro formas de entrenar: cada hospital por su
cuenta (local-only), un modelo centralizado con todos los datos juntos, FedAvg y
FedProx con varios valores del término proximal. La arquitectura es una
ResNet-18 sin preentrenar. El detalle del análisis y los resultados está en el
informe.

## Contenido del repositorio

```
p6/
├── run_all.py            Lanzamiento de los experimentos
├── src/                  Codigo del proyecto
│   ├── config.py         Rutas y params
│   ├── dataset.py        Carga de imágenes y CSV
│   ├── model.py          Resnet18
│   ├── train_utils.py    Entrenamiento local y cálculo de métricas
│   ├── baselines.py      Local-only y centralizado
│   ├── federated.py      Simulación federada con Flower
│   ├── fl_client.py      Cliente Flower
│   └── fl_server.py      Servidor y estrategias FedAvg / FedProx
├── results/              Métricas, modelos
├── pyproject.toml        Dependencias
```

## Datos

Las imágenes no están incluidas en el repositorio por su tamaño. Hay que
colocarlas en la carpeta data/, que config toma como root.

El reparto de imágenes entre hospitales viene dado por los CSV 06_hfl_cls_plane_comparison_flair, se puede seleccionar su ruta en el archivo config.

## Requisitos y ejecución

Las dependencias están en pyproject.toml y el proyecto se gestiona con uv:

uv run python run_all.py

run_all.py entrena centralizado, local-only, FedAvg y FedProx, y guarda los
resultados en /results. Se puede lanzar un solo expermiento con:

uv run python run_all.py --only fedavg

Se admiten otros params como --rounds, --lr, --local-epochs desde run_all o config.py

Autores: Pablo Jesús Delgado Muñoz, María Vázquez López y Paola Gambero Díaz.
