build_features.py
=================

Descripción
-----------
``build_features.py`` es usado para realizar la transformación de datos necesaria para preparar el entrenamiento de modelos. Este script soporta varios parámetros para ajustar el proceso de transformación.
Este script utiliza la clase ``WineDatasetTransformer`` que hereda de ``BaseEstimator`` y ``TransformerMixin``.

La clase ``WineDatasetTransformer`` se utiliza para transformar los datasets de vino.
A continuación se detallan sus métodos y parámetros.

.. autoclass:: aidtecsolutions.features.custom_transformers.WineDatasetTransformer
    :members:
    :undoc-members:
    :show-inheritance:

Para utilizar este script se usa la terminal mediante el comando:

.. code-block:: bash


    $ ./make_features.sh --flag

Uso
---
Para acceder a la ayuda de los comandos disponibles. Desde la raiz del proyecto:

.. code-block:: bash

    $ ./make_features.sh -h

Parámetros
----------
- ``--con archivo.csv``: Nombre del dataset de data/raw para hacer transformaciones.
- ``--alcohol``: Corrige los valores de la variable alcohol.
- ``--densidad``: Corrige los valores de la variable densidad.
- ``--shuffle``: Baraja el dataset.
- ``--color``: Crea interacciones entre la variable color y las variables correlacionadas.
- ``--densidad_alcohol``: Crea la interacción entre la variable densidad y alcohol.
- ``--estandarizar``: estandariza las variables tipo float
- ``--ratiodiox``: Crea el ratio entre el dioxido de azufre libre y el dioxido de azufre total
- ``--rbfdiox``: Crea similitudes Gausianas mediante método rbf entre los 2 modos del dioxido de azufre total.
- ``--outliers``: Elimina outliers del dataset usando el método IsolationForest.
- ``--drop columna1 columna2``: Elimina las columnas pasadas como argumento.
- ``--log columna1 columna2``: Aplica una transformación logarítmica a las columnas pasadas
- ``--save``: Guarda el dataset en ``data/processed``.

Ejemplos
--------
Aquí se muestra cómo puedes correr ``build_features.py`` con diferentes configuraciones:

.. code-block:: bash

    $ ./make_features.sh --con train.csv --alcohol --densidad --shuffle --drop year calidad color
    $ ./make_features.sh --con train.csv --alcohol --densidad --shuffle --estandarizar --save

El primer comando abrirá el dataset **train.csv**, corregirá los valores de la variable alcohol y densidad, barajará el dataset y dropeará las columnas year, calidad y color

El segundo comando además de corregir y barajar, estandarizará el dataset y lo guardará con la carpeta **data/processed**. El nombre del archivo creado será:
**train.csv-corregir_alcohol-corregir_densidad-shuffle-estandarizar.csv

