make_dataset.py
=================

Descripción
-----------
``make_dataset.py`` es usado para la descarga de los datasets **train** y **test** de la web de Kopuru.

Para utilizar este script se usa la terminal mediante el comando:

.. code-block:: bash


    $ ./make_dataset.sh --flag

Uso
---
Para acceder a la ayuda de los comandos disponibles. Desde la raiz del proyecto:

.. code-block:: bash

    $ ./make_dataset.sh -h

Parámetros
----------
- ``--train``: Descarga el dataset train. Lo guarda en **data/raw** con el nombre de **train.csv**
- ``--test``: Descarga el dataset test. Lo guarda en **data/raw** con el nombre de **test.csv**.

Ejemplos
--------
Aquí se muestra cómo puedes correr ``make_dataset.py`` con diferentes configuraciones:

.. code-block:: bash

    $ ./make_dataset.sh --train
    $ ./make_dataset.sh --test