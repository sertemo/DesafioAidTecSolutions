# DesafioAidTecSolutions v0.1.0
## Predicción de la calidad del vino a través de muestras de laboratorio

![Tests](https://github.com/sertemo/DesafioAidTecSolutions/actions/workflows/tests.yml/badge.svg)
![Dependabot](https://img.shields.io/badge/dependabot-enabled-blue.svg?logo=dependabot)
![GitHub](https://img.shields.io/github/license/sertemo/DesafioAidTecSolutions)


## Consideraciones del desafío
1. Cada una de las muestras (registros de las muestras) ha sido tomada de una única botella en las mismas condiciones y momento del proceso (tras el embotellado). Por tanto, todas las muestras están sujetas al mismo proceso, ya que se trata de vinos de año.
2. Es importante analizar en profundidad los datos en busca de posibles errores, que puedan afectar al modelo, y gestionarlos de forma correcta para que no afecten. Se recomienda realizar un EDA previo, que se incluirá en el informe.
3. Es importante realizar un análisis de cuáles son las variables más representativas en la decisión de la asignación de la calidad.
4. Es importante establecer cuáles son los valores máximos y mínimos admisibles para cada parámetro.
5. Aplica transformaciones de variables si fuera necesario.
6. Normaliza la información antes de realizar el modelo y toma decisiones que documentarás en el informe final.

## Objetivo del desafío
1. Desarrollar un modelo de clasificación capaz de predecir con los datos de las muestras del 2024 cuál será la calidad de dichas muestras, tomando como referencia datos entre 2019 y 2023.
2. Desarrollar un informe explicativo donde se desarrollen las conclusiones sobre variables que más influyen en los resultados y el por qué, así como el análisis previo de los datos y sus descriptivos.

## Notas
Proyecto con la intención de practicar la organización y estructura de proyectos de **data science** siguiendo la arquitectura de [cookiecutter](https://drivendata.github.io/cookiecutter-data-science/#starting-a-new-project) pero sin usar dicha herramienta.

- La creación de las diferentes carpetas se realiza a mano y se empaqueta usando **setuptools**
- **pyproject.toml** para configuración del la wheel y el backend
- **setup.cfg** para la metadata del proyecto
- **setup.py** para realizar el setup

Con estos archivos configurados ya se puede hacer el paquete **aidtecsolutions** instalable con 
```sh
$ pip install -e .
```

Esto crea enlaces al paquete en el entorno virtual


## Documentación
La documentación ha sido generada con **Sphinx** y está disponible desde **docs/_build/html/index.html**

Para generar de nuevo la documentación. Desde el directorio **docs**:
```sh
$ sphinx-build -b html . _build/html
```

Esto generará la documentación en html y lo guardará en **docs/_build/html**.

## Uso
### 1. Make Dataset
Para descargar los datasets de la web de [Kopuru](https://kopuru.com/challenge/modelo-de-prediccion-de-calidad-en-el-vino-para-aidtec-solutions/?tab=tab-link_datos) usaremos el comando `./make_dataset` desde la raiz del proyecto. Este argumento necesita 1 argumento que corresponde con el tipo de dataset a descargar: **train** o **test**.

Para ver la ayuda:
```sh
$ ./make_dataset.sh -h
```
Esto imprimirá lo siguiente:
```sh
usage: make_dataset.py [-h] [--train] [--test]

Descarga los datasets train y test de la web de Kopuru

options:
  -h, --help  show this help message and exit
  --train     Descarga el dataset de train de la web de kopuru
  --test      Descarga el dataset de test la web de kopuru
```

Ejemplo de uso:
```sh
$ ./make_dataset.sh --train 
```

Esto descargará el dataset de train de la web de kopuru y lo almacenará con el nombre de **train.csv** en **data/processed**. El nombre se puede editar desde **settings.py**.

### 2. Make Features
Para la creación del dataset definitivo de cara al entrenamiento usaremos el comando `./make_features.sh` desde la raiz del proyecto. Este comando admite varias flags que aplicarán una serie de transformaciones al dataset original.

Para ver todas las transformaciones disponibles:
```sh
$ ./make_features.sh -h
```

Esto printeará en la consola la siguiente lista:
```sh
options:
  -h, --help            show this help message and exit
  --con CON             Especificar el archivo a abrir para las transformaciones
  --alcohol             Corrige los valores de la variable alcohol
  --densidad            Corrige los valores de la variable densidad
  --shuffle             Baraja el dataset
  --color               Crea todas las interacciones con la variable color
  --densidad_alcohol    Crea interacci▒n entre alcohol y densidad
  --estandarizar        Estandariza todas las variables tipo float
  --ratiodiox           Crea un ratio con la variable dioxido de azufre
  --rbfdiox             Crea las similitudes con los modos de dioxido de
                        azufre
  --outliers            Elimina outliers del datset usando IsolationForest
                        como m▒todo
  --drop DROP [DROP ...]
                        Dropea las columnas pasadas
  --log LOG [LOG ...]   Crea transformaciones logar▒tmicas a las varibales
                        pasadas
  --save                Guarda el dataset en formato csv en data\processed

```

La flag `--save` guarda el dataset transformado con una estructura de nombre que sigue el siguiente formato:
```
<nombre dataset original>-<transformacion_1>-<transfomacion_2>-drop=columa1-columna2.csv
```

**Importante**: Los nombres de las columnas que contengan espacios deberán pasarse sustituyendo dichos espacios por guiones bajos. Por ejemplo:

```sh
$ ./make_features.sh --con train.csv --alcohol --densidad --shuffle --densidad_alcohol --ratiodiox --rbfdiox --drop color densidad alcohol dioxido_de_azufre_libre --save
```

### 3. Train Model
Para entrenar un modelo usaremos el comando `./train_model.sh` junto con varios argumentos.

La estructura a ejecutar es la siguiente:

```sh
$ ./train_model.sh --data <nombre dataset> [--save] <modelo: xgb o randomforest> [parámetros del modelo]
```

Para ver los comandos principales disponibles podemos recurrir a la ayuda:

```sh
$ ./train_model.sh --help
```

Esto imprimirá lo siguiente

```sh
options:
  -h, --help          show this help message and exit
  --data DATA         Nombre del archivo del dataset sobre el que entrenar el modelo. Debe estar en                   la carpeta data/processed
  --save              Guarda el modelo serializado con joblib en /models

models:
  Modelos disponibles para entrenamiento

  {xgb,randomforest}
    xgb               Entrena un modelo xgb con sus parámetros
    randomforest      Entrena un modelo RF con sus parámetros
```

El primer argumento a pasar representa el dataset con el que queremos entrenar el modelo. El argumento es `--data` seguido del nombre de archivo. El archivo debe estar guardado dentro de la carpeta **data/processed**.

El siguiente argumento opcional a pasar es `--save`. Si este argumento es pasado, el modelo entrenado resultante se guardará serializado con **joblib** en la carpeta **models/**. El nombre del archivo se generará automáticamente cogiendo el nombre del dataset de entrenamiento y los parámetros del modelo.

El siguiente argumento obligatorio representa el nombre del modelo que queremos entrenar. De momento disponibles están **xgb** y **randomforest**.

Para ver los argumentos opcionales de cada modelo se puede ejecutar la ayuda:

Los del **xgb**:
```sh
$ ./train_model xgb --help
```

Esto imprimirá lo siguiente:
```sh
usage: train_model.py xgb [-h] [--gamma GAMMA] [--max_depth MAX_DEPTH] [--alpha ALPHA] [--learning_rate LEARNING_RATE]
                          [--n_estimators N_ESTIMATORS]

options:
  -h, --help            show this help message and exit
  --gamma GAMMA         Minimum loss reduction required to make a further partition on a leaf node of the tree.         The larger gamma    
                        is, the more conservative the algorithm will be.
  --max_depth MAX_DEPTH
                        Maximum depth of a tree. Increasing this value will make the model more complex and more likely to
                        overfit. 0 indicates no limit on depth.
  --alpha ALPHA         L1 regularization term on weights. Increasing this value will make model more conservative.
  --learning_rate LEARNING_RATE
                        Learning rate
  --n_estimators N_ESTIMATORS
                        Number of trees
```

Los del **random forest**:
```sh
$ ./train_model randomforest --help
```
Esto imprimirá lo siguiente:
```sh
usage: train_model.py randomforest [-h] [--n_estimators N_ESTIMATORS] [--criterion {gini,entropy,log_loss}]
                                   [--max_depth MAX_DEPTH] [--class_weight {balanced,balanced_subsample,None}]

options:
  -h, --help            show this help message and exit
  --n_estimators N_ESTIMATORS
                        Number of trees
  --criterion {gini,entropy,log_loss}
                        The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and     
                        “log_loss” and “entropy” both for the Shannon information gain, see Mathematical formulation. Note: This    
                        parameter is tree-specific.
  --max_depth MAX_DEPTH
                        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all      
                        leaves contain less than min_samples_split samples.
  --class_weight {balanced,balanced_subsample,None}
                        Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed   
                        to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the     
                        columns of y
```

Un ejemplo de sentencia completa podría ser:
```sh
$  ./train_model.sh --save --data  train.csv-corregir_alcohol-corregir_densidad-drop=alcohol-densidad-year-color.csv randomforest --n_estimators 150 --criterion log_loss
```

Al entrenar el modelo se evaluará primero con **cross validation** y 5 splits en el dataset y se imprimirá la accuracy media de todos los splits y un informe con otras métricas.

### 4. Make Prediction
Para realizar predicciones necesitaremos:
1. Un dataset **X_test** transformado sin la columna **calidad**.
2. Un modelo entrenado sobre un dataset con las mismas transformacioens que **X_test**

Para hacer predicciones utilizaremos el comando: `./make_prediction.sh` junto dos argumentos.

El dataset a utilizar deberá estar en la carpeta **data/processed**.

El modelo a utilizar deberá estar en la carpeta **/models**.

El argumento opcional `--merge` guarda las predicciones junto con el archivo `test.csv` especificado en **settings.py** como `TEST_FILE` y lo guarda en la carpeta **data/processed** con el nombre especificado.

Para ver la ayuda:
```sh
$ ./make_prediction.sh --help
```

Esto imprimirá:
```sh
Utilizando modelos entrenados y un dataset previamente transformado, lanza predicciones

options:
  -h, --help     show this help message and exit
  --data DATA    El archivo dataset en data/processed sobre el que hacer predicciones
  --model MODEL  El archivo del modelo de /models con el que hacer las predicciones
  --merge MERGE  Mergea las predicciones junto con el dataset `test.csv` que debe estar en data/raw. Guarda en data/processed el nuevo dataframe con el nombre especificado.
```

Ejemplo de uso:
```sh
$ ./make_prediction.sh --data test.csv-corregir_alcohol-corregir_densidad-densidad_alcohol_interaction-ratio_diox-rbf_diox-shuffle-drop=color-densidad-alcohol-year-dioxido_de_azufre_libre-calidad.csv --model model_data=train.csv-corregir_alcohol-corregir_densidad-densidad_alcohol_interaction-ratio_diox-rbf_diox-shuffle-drop=color-densidad-alcohol-year-dioxido_de_azufre_libre.csv_model=randomforest_n_estimators=900_criterion=gini.joblib --merge y_preds.csv
```

La línea anterior lanza predicciones sobre el dataset especificado con el modelo especificado, une las predicciones al dataset **test.csv** que debe localizarse en **data/raw** y las guarda en el archivo **y_preds.csv** en **data/processed**.


## Licencia
Copyright 2024 Sergio Tejedor Moreno

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.