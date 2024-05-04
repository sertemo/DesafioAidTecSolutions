train_model.py
=================

Descripción
-----------
El módulo ``train_model`` se usa para entrenar dos tipos de modelos:
- XgBoost
- RandomForest

Clases
-------

.. autoclass:: aidtecsolutions.wrappers.SerializableClassifier
    :members:
    :undoc-members:
    :show-inheritance:

Esta clase se usa para envolver los clasificadores de sklearn y dotarlos de la posibilidad de serialización y deserialización

Para utilizar este script se usa la terminal mediante el comando:

.. code-block:: bash

    $ ./train_model.sh --data <nombre del dataset> [--save] <nombre modelo: xgb o randomforest> [parámetros del modelo]

Uso
---
Para acceder a la ayuda de los comandos disponibles. Desde la raiz del proyecto:

.. code-block:: bash

    $ ./train_model.sh -h

Parámetros
----------
- ``--data DATA``: Nombre del dataset de data/processed sobre el que entrenar el modelo.
- ``--save``: Guarda el modelo serializado con joblib en ``models``.
- ``models``:
    {xgb, randomforest}
    - ``xgb`` Entrena un modelo xgboost con sus parámetros.
    - ``randomforest``  Entrena un modelo random forest con sus parámetros.

Para acceder a los parámetros de cada modelo:

.. code-block:: bash

    $ ./train_model.sh xgb -h

- ``--gamma GAMMA``     Minimum loss reduction required
                        to make a further partition on a
                        leaf node of the tree. The
                        larger gamma is, the more
                        conservative the algorithm will
                        be.
- ``--max_depth MAX_DEPTH``
                        Maximum depth of a tree.
                        Increasing this value will make
                        the model more complex and more
                        likely to overfit. 0 indicates
                        no limit on depth.
- ``--alpha ALPHA``     L1 regularization term on
                        weights. Increasing this value
                        will make model more
                        conservative.
- ``--learning_rate LEARNING_RATE``
                        Learning rate
- ``--n_estimators N_ESTIMATORS``
                        Number of trees

.. code-block:: bash

    $ ./train_model.sh randomforest -h

- ``--n_estimators N_ESTIMATORS``
                        Number of trees
- ``--criterion {gini,entropy,log_loss}``
                        The function to measure the quality of a split. Supported criteria
                        are “gini” for the Gini impurity and “log_loss” and “entropy” both
                        for the Shannon information gain, see Mathematical formulation.   
                        Note: This parameter is tree-specific.
- ``--max_depth MAX_DEPTH``
                        The maximum depth of the tree. If None, then nodes are expanded     
                        until all leaves are pure or until all leaves contain less than     
                        min_samples_split samples.
- ``--class_weight {balanced,balanced_subsample,None}``
                        Weights associated with classes in the form {class_label: weight}.  
                        If not given, all classes are supposed to have weight one. For      
                        multi-output problems, a list of dicts can be provided in the same  
                        order as the columns of y


Ejemplos
--------
Aquí se muestra cómo puedes correr ``train_models.py`` con diferentes configuraciones:

.. code-block:: bash

    $ ./train_model.sh --data train.csv-corregir_alcohol-corregir_densidad-shuffle.csv randomforest --n_estimators 800
    $ ./train_model.sh --data train.csv-corregir_alcohol-corregir_densidad.csv randomforest
    $ ./train_model.sh --data train.csv-corregir_alcohol-corregir_densidad-shuffle-drop=year-color.csv --save xgb --learning_rate 0.1

Estos comandos realizarán una evaluación en **cross validation** del modelo en cuestión con **5 splits** y printearán la media de las accuracies.

Si el argumento ``--save`` ha sido pasado, el modelo será guardado en la carpeta **models** con un nombre acorde al dataset, el modelo y sus argumentos.

Por ejemplo para el tercer caso del ejemplo anterior, el nombre del modelo serializado será:

**model_data=train.csv-corregir_alcohol-corregir_densidad-shuffle-drop=year-color.csv_save=True_model=xgb_learning_rate_0.1.joblib**

