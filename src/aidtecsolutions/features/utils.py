# Copyright 2024 Sergio Tejedor Moreno

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script con funciones auxiliares para crear las features"""


import argparse


def generate_dataset_name(args: argparse.Namespace) -> str:
    """Devuelve el nombre de archivo para
    guardar el dataset como csv en función
    de las transformaciones realizadas

    Parameters
    ----------
    settings : types.ModuleType
        _description_
    args : argparse.Namespace
        _description_

    Returns
    -------
    str
        Devuelve el nombre del dataset para guardar
    """
    # Base del nombre del archivo, con el nombre del dataset
    name_parts = [args.con]

    # Añadir partes del nombre basado en argumentos True
    if args.alcohol:
        name_parts.append("corregir_alcohol")
    if args.densidad:
        name_parts.append("corregir_densidad")
    if args.color:
        name_parts.append("color_interactions")
    if args.densidad_alcohol:
        name_parts.append("densidad_alcohol_interaction")
    if args.ratiodiox:
        name_parts.append("ratio_diox")
    if args.rbfdiox:
        name_parts.append("rbf_diox")
    if args.outliers:
        name_parts.append("remove_outliers")
    if args.estandarizar:
        name_parts.append("estandarizar")
    if args.shuffle:
        name_parts.append("shuffle")

    # Para listas, solo añadir si no están vacías
    if args.drop and len(args.drop) > 0:
        # drop_columns = "_".join(args.drop)
        name_parts.append(f"drop={'-'.join(args.drop)}")

    # Para transformaciones logarítmicas, verificar que no sea None
    if args.log:
        name_parts.append(f"log_transformation={'-'.join(args.log)}")

    # Unir todas las partes con guiones bajos
    return "-".join(name_parts) + ".csv"
