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
from aidtecsolutions.custom_exceptions import NonValidDataset
from aidtecsolutions.features.custom_transformers import WineDatasetTransformer
from aidtecsolutions.features.utils import generate_dataset_name
from aidtecsolutions.utils import is_valid_dataset, is_valid_dataframe


def setup_parser() -> argparse.ArgumentParser:
    """Crea el parser con los argumentos

    Returns
    -------
    argparse.ArgumentParser
        _description_
    """
    parser = argparse.ArgumentParser()

    # Añadimos argumentos
    parser.add_argument(
        "--con",
        help="Nombre del archivo a aplicar las transformaciones",
        required=True,
    )
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
        help="Estandariza todas las variables tipo float",
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
        "--outliers",
        help="Elimina outliers del datset usando IsolationForest como método",
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
        "--save",
        help=f"Guarda el dataset en formato csv en {settings.FOLDER_DATA_PROCESSED}",
        action="store_true",
    )

    return parser


def main() -> None:

    # Parseamos los argumentos
    parser = setup_parser()
    args = parser.parse_args()

    # Verificar que el archivo esté en data/raw
    dataset = args.con
    if not is_valid_dataset(dataset, settings.FOLDER_DATA_RAW):
        print(
            f"""
        El archivo {dataset} no se encuentra en {settings.FOLDER_DATA_RAW},
        quizás se te haya olvidado descargarlo.
        Haz: ./make_dataset.sh --train para descargar el dataset train.csv o
        ./make_dataset.sh --test para descargar el dataset test.csv.
        """
        )
        return

    # Verificar que se trate de un archivo válido, si lo es carga el dataset
    try:
        df_train = is_valid_dataframe(settings.FOLDER_DATA_RAW, dataset)
    except NonValidDataset as exc:
        print(f"Dataset erróneo. Error: {exc}")
        return

    wt = WineDatasetTransformer(
        corregir_alcohol=args.alcohol,
        corregir_densidad=args.densidad,
        color_interactions=args.color,
        densidad_alcohol_interaction=args.densidad_alcohol,
        ratio_diox=args.ratiodiox,
        rbf_diox=args.rbfdiox,
        remove_outliers=args.outliers,
        standardize=args.estandarizar,
        log_transformation=args.log,
        drop_columns=args.drop,
        shuffle=args.shuffle,
    )

    # Aplicamos las transformaciones pasadas por consola
    df_train_transformed: pd.DataFrame = wt.fit_transform(df_train)
    print(df_train_transformed.columns)
    print(df_train_transformed.head())

    if args.save:
        nombre_dataset: str = generate_dataset_name(args)
        ruta_completa = settings.FOLDER_DATA_PROCESSED / nombre_dataset
        df_train_transformed.to_csv(ruta_completa)

        print("Guardado dataset correctamente en:")
        print(ruta_completa)


if __name__ == "__main__":
    main()
