#!/bin/bash

# Este script facilita la ejecución de build_features.py con diferentes configuraciones.

function make_dataset() {
    # Construir el comando con los argumentos pasados a este script de Bash
    CMD="python src/aidtecsolutions/data/make_dataset.py"  # Asegúrate de ajustar la ruta.

    # Añadir flags y argumentos basado en lo que se pase a este script de Bash
    for arg in "$@"; do
        CMD="$CMD $arg"
    done

        # Ejecutar el comando
        echo "Ejecutando: $CMD"
        $CMD
    }

# Llamar a la función run_build_features con todos los argumentos pasados a este script
make_dataset "$@"