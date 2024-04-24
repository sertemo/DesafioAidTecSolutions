#!/usr/bin/env python


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

"""Scripts to turn raw data into features for modeling"""

import argparse

import pandas as pd

import settings
from aidtecsolutions.features.custom_transformers import WineDatasetTransformer


def main() -> None:
    parser = argparse.ArgumentParser()

    # Añadimos argumentos
    parser.add_argument(
        "--alcohol",
        help="Corrige los valores de la variable alcohol",
        action="store_true",
    )
    parser.add_argument(
        "--densidad",
        help="Corrige los valores de la variable densidad",
        action="store_true",
    )
    parser.add_argument("--shuffle", help="Baraja el dataset", action="store_true")
    parser.add_argument(
        "--color",
        help="Crea todas las interacciones con la variable color",
        action="store_true",
    )
    parser.add_argument(
        "--densidad_alcohol",
        help="Crea interacción entre alcohol y densidad",
        action="store_true",
    )
    parser.add_argument(
        "--estandarizar",
        help="Etandariza todas las variables tipo float",
        action="store_true",
    )
    parser.add_argument(
        "--ratiodiox",
        help="Crea un ratio con la variable dioxido de azufre",
        action="store_true",
    )
    parser.add_argument(
        "--rbfdiox",
        help="Crea las similitudes con los modos de dioxido de azufre",
        action="store_true",
    )
    parser.add_argument(
        "--drop",
        nargs="+",
        help="Dropea las columnas pasadas",
    )
    parser.add_argument(
        "--log",
        nargs="+",
        help="Crea transformaciones logarítmicas a las varibales pasadas",
    )
    parser.add_argument(
        "--save", help="Guarda el dataset en formato csv", action="store_true"
    )

    # Parseamos los argumentos
    args = parser.parse_args()

    wt = WineDatasetTransformer(
        corregir_alcohol=args.alcohol,
        corregir_densidad=args.densidad,
        shuffle=args.shuffle,
        color_interactions=args.color,
        densidad_alcohol_interaction=args.densidad_alcohol,
        standardize=args.estandarizar,
        ratio_diox=args.ratiodiox,
        rbf_diox=args.rbfdiox,
        drop_columns=args.drop,
        log_transformation=args.log,
    )

    X = pd.read_csv(settings.RUTA_TRAIN_DATASET_RAW, index_col=0)

    X_transformed: pd.DataFrame = wt.fit_transform(X)
    print(X_transformed.columns)
    print(X_transformed.head())

    if args.save:
        dataset_name = f"""{settings.TRAIN_DATASET}_\
        {'alcohol' if args.alcohol else ''}-\
        {'corregir_densidad' if args.densidad else ''}-\
        {'shuffle' if args.shuffle else ''}-\
        {'color_interactions' if args.color else ''}-\
        {'densidad_alcohol_interaction' if args.densidad_alcohol else ''}-\
        {'standardize' if args.estandarizar else ''}-\
        {'ratio_diox' if args.ratiodiox else ''}-\
        {'rbf_diox' if args.rbfdiox else ''}-\
        drop_columns={str(args.drop)}-\
        log_transformation={args.log}"""
        dataset_name = "".join(dataset_name.split())

        ruta_dataset = settings.FOLDER_DATA_PROCESSED / (dataset_name + ".csv")
        X_transformed.to_csv(ruta_dataset)

        print("Guardado dataset:")
        print(ruta_dataset)


if __name__ == "__main__":
    main()
