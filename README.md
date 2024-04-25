# Desafío Kopuru
## AidTec Solutions
## Predicción de la calidad del vino a través de muestras de laboratorio

![Tests](https://github.com/sertemo/DesafioAidTecSolutions/actions/workflows/tests.yml/badge.svg)

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

## Uso
# 1. Build Features
En primer lugar para la creación del dataset definitivo de cara al entrenamiento usaremos el comando **./make_features.sh** desde la raiz del proyecto. Este comando admite varias flags que aplicarán una serie de transformaciones al dataset original.

Para ver todas las transformaciones disponibles:
```sh
$ ./make_features.sh -h
```

Esto printeará en la consola la siguiente lista:
```
options:
  -h, --help            show this help message and exit
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

La flag **--save** guarda el dataset transformado con una estructura de nombre que sigue el siguiente formato:
```
<nombre dataset original>-<transformacion_1>-<transfomacion_2>-drop=['columa1', 'columna2'].csv
```


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