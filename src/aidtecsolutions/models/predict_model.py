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

"""Script para usar los modelos y lanzar predicciones"""

import argparse

import pandas as pd

from aidtecsolutions.custom_exceptions import NonValidDataset
from aidtecsolutions.wrappers import SerializableClassifier, SerializableTransformer
from aidtecsolutions.utils import is_valid_dataset, is_valid_dataframe
import settings


def setup_parser() -> argparse.ArgumentParser:
    """Configura el parser de predict model

    Returns
    -------
    argparse.ArgumentParser
        _description_
    """
    parser = argparse.ArgumentParser(
        description="Utilizando modelos entrenados y un dataset \
            previamente transformado, lanza predicciones"
    )
    parser.add_argument(
        "--data",
        help="El archivo dataset en data/processed sobre el que hacer predicciones",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        help="El archivo del modelo de /models con el que hacer las predicciones",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--merge",
        help="Mergea las predicciones junto con el dataset `test.csv`\
            que debe estar en data/raw. \
                Guarda en data/processed el nuevo dataframe con el nombre\
                    especificado.",
        type=str,
    )
    return parser


def main() -> None:
    # Parseamos los argumentos
    parser = setup_parser()
    args = parser.parse_args()
    print(args)

    # Comprobamos que existe el archivo pasado por el usuario
    if not is_valid_dataset(args.data, settings.FOLDER_DATA_PROCESSED):
        print(
            f"No se encuentra el archivo {args.data} en {settings.FOLDER_DATA_PROCESSED}"
        )
        return

    # Comprobamos que esté bien el archivo y sea válido
    try:
        df_test = is_valid_dataframe(settings.FOLDER_DATA_PROCESSED, args.data)
    except NonValidDataset as exc:
        print(f"Dataset erróneo. Error: {exc}")
        return

    # Comprobamos que el archivo del modelo exista
    model_filename: str = args.model
    if not (settings.FOLDER_MODELS_SERIALISED / model_filename).exists():
        print(
            f"No se encuentra el modelo {args.modelo} en {settings.FOLDER_MODELS_SERIALISED}"
        )
        return

    # Cargamos modelo y label encoders
    model: SerializableClassifier = SerializableClassifier.load(
        settings.FOLDER_MODELS_SERIALISED / model_filename
    )
    label_encoders: SerializableTransformer = SerializableTransformer.load(
        settings.FOLDER_MODELS_SERIALISED / (settings.LABEL_ENCODER_NAME + ".joblib")
    )
    try:
        preds = model.predict(df_test)
        preds = label_encoders.inverse_transform(preds)
    except Exception as exc:
        print("Se ha producido el siguiente error al predecir:", exc)
        return

    df_preds = pd.DataFrame(preds, columns=[settings.TARGET_FEATURE])
    print(df_preds)

    if args.merge is not None:
        y_test_w_preds_filename: str = args.merge
        if ".csv" not in y_test_w_preds_filename:
            print("El nombre de archivo no es válido. Debes pasar un .csv")
            return
        if not (settings.FOLDER_DATA_RAW / settings.TEST_FILE).exists():
            print(f"No se encuentra {settings.TEST_FILE} en data/raw")
            return

        y_test = pd.read_csv(
            (settings.FOLDER_DATA_RAW / settings.TEST_FILE),
            index_col=0,
        )
        # Adecuamos los índices
        y_test_ = y_test.copy()
        df_preds.index = y_test_.index
        # Añadimos a y_test la variable target
        y_test_[settings.TARGET_FEATURE] = df_preds
        print(y_test_)
        # Guardamos
        y_test_.to_csv(settings.FOLDER_DATA_PROCESSED / y_test_w_preds_filename)
        print("Guardadas correctamente en data/processed las predicciones.")


if __name__ == "__main__":
    main()
