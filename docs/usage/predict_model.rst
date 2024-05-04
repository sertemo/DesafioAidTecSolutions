predict_model.py
=================

Descripción
-----------
``predict_model.py`` es usado para hacer predicciones utilizando un determinado modelo y un determinado dataset.

Para utilizar este script se usa la terminal mediante el comando:

.. code-block:: bash


    $ ./make_prediction.sh --data test.csv --model modelo.joblib --merge X_test_with_preds.csv

Uso
---
Para acceder a la ayuda de los comandos disponibles. Desde la raiz del proyecto:

.. code-block:: bash

    $ ./make_dataset.sh -h

Parámetros
----------
- ``--data DATA``: El dataset usado para las predicciones. Debe estar en **data/processed**.
- ``--model MODEL``: El modelo usado para las predicciones. Debe estar en **models/**.
- ``--merge MERGE`` : Argumento opcional. A pasar con el nombre del dataset para guardar las predicciones. Mergea el dataset **test.csv** situado en **data/raw** con las predicciones.