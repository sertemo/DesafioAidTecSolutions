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
- Proyecto con la intención de practicar la organización y estructura de proyectos de **data science** siguiendo la arquitectura de [cookiecutter](https://drivendata.github.io/cookiecutter-data-science/#starting-a-new-project) pero sin usar dicha herramienta.

- La creación de las diferentes carpetas se realiza a mano y se empaqueta usando **setuptools**
- **pyproject.toml** para configuración del la wheel y el backend
- **setup.cfg** para la metadata del proyecto
- **setup.py** para realizar el setup

Con estos archivos configurados ya se puede hacer el paquete **aidtecsolutions** instalable con 
```sh
pip install -e .
```

Esto crea enlaces al paquete en el entorno virtual

## Uso

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