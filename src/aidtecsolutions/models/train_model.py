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

"""Script para entrenar modelos"""

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from aidtecsolutions.custom_exceptions import NonValidDataset
from aidtecsolutions.models.custom_models import SerializableClassifier
from aidtecsolutions.models.utils import generate_model_name
from aidtecsolutions.utils import is_valid_dataset, is_valid_dataframe
import settings


def setup_parser() -> argparse.ArgumentParser:
    """Configura el parser

    Returns
    -------
    argparse.ArgumentParser
        _description_
    """
    parser = argparse.ArgumentParser(
        description="Entrena modelos con parámetros específicos"
    )

    # Añadimos argumentos
    parser.add_argument(
        "--data",
        help="Nombre del archivo del dataset sobre el que entrenar el modelo. \
            Debe estar en la carpeta data/processed",
        required=True,
    )
    parser.add_argument(
        "--save",
        help="Guarda el modelo serializado con joblib en /models",
        action="store_true",
    )
    # Creamos subparser para los modelos
    subparsers = parser.add_subparsers(
        title="models",
        description="Modelos disponibles para entrenamiento",
        required=True,
        dest="model",
    )
    subparsers.required = True

    # Subparser para XGBoost
    xgb_parser = subparsers.add_parser(
        "xgb", help="Entrena un modelo xgb con sus parámetros"
    )
    xgb_parser.add_argument(
        "--gamma",
        type=float,
        default=0,
        help="Minimum loss reduction required to make a further partition \
            on a leaf node of the tree. The larger gamma is, \
                the more conservative the algorithm will be.",
    )
    xgb_parser.add_argument(
        "--max_depth",
        type=int,
        help="Maximum depth of a tree. Increasing this value will make \
            the model more complex and more likely to overfit. \
                0 indicates no limit on depth.",
        default=6,
    )
    xgb_parser.add_argument(
        "--alpha",
        help="L1 regularization term on weights. \
            Increasing this value will make model more conservative.",
        default=0,
        type=int,
    )
    xgb_parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    xgb_parser.add_argument(
        "--n_estimators", type=int, default=100, help="Number of trees"
    )

    # Subparser para RandomForest
    rf_parser = subparsers.add_parser(
        "randomforest", help="Entrena un modelo RF con sus parámetros"
    )
    rf_parser.add_argument(
        "--n_estimators", type=int, default=100, help="Number of trees"
    )
    rf_parser.add_argument(
        "--criterion",
        type=str,
        default="gini",
        help="The function to measure the quality of a split. \
            Supported criteria are “gini” for the Gini impurity \
                and “log_loss” and “entropy” both for the Shannon \
                    information gain, see Mathematical formulation. \
                        Note: This parameter is tree-specific.",
        choices=["gini", "entropy", "log_loss"],
    )
    rf_parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="The maximum depth of the tree. If None, then nodes are \
            expanded until all leaves are pure or until all \
                leaves contain less than min_samples_split samples.",
    )
    rf_parser.add_argument(
        "--class_weight",
        type=str,
        default=None,
        help="Weights associated with classes in the form \
            {class_label: weight}. If not given, all classes \
                are supposed to have weight one. \
                    For multi-output problems, a list of dicts \
                        can be provided in the same order as the columns of y",
        choices=["balanced", "balanced_subsample", None],
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
        df_train = is_valid_dataframe(settings.FOLDER_DATA_PROCESSED, args.data)
    except NonValidDataset as exc:
        print(f"Dataset erróneo. Error: {exc}")
        return

    print(f"Validando modelo con CV y {settings.SPLITS_FOR_CV} splits:")
    if args.model == "xgb":
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            gamma=args.gamma,
            max_depth=args.max_depth,
            alpha=args.alpha,
            learning_rate=args.learning_rate,
        )
    elif args.model == "randomforest":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            criterion=args.criterion,
            max_depth=args.max_depth,
            class_weight=args.class_weight,
        )
    model_name = model
    print(model_name)

    model = SerializableClassifier(model)

    X = df_train.drop(columns=[settings.TARGET_FEATURE])
    y = df_train[settings.TARGET_FEATURE]

    # Codificamos labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Usamos cross validation para estimar una accuracy media
    results_cv = cross_val_score(
        model,
        X,
        y_encoded,
        n_jobs=-1,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=settings.SPLITS_FOR_CV),
    )

    y_preds = cross_val_predict(
        model,
        X,
        y_encoded,
        cv=StratifiedKFold(n_splits=settings.SPLITS_FOR_CV),
        n_jobs=-1,
    )

    print(f"Resultados de modelo {model_name}")
    print(
        f"Accuracy media en CV con {settings.SPLITS_FOR_CV} splits: {results_cv.mean():.3%}"
    )
    print(classification_report(y_true=y_encoded, y_pred=y_preds, zero_division=0))

    # Solo entrenamos el modelo si han pasado '--save'
    if args.save:
        # Entrenamos en el dataset completo
        model.fit(X, y_encoded)
        # Guardamos
        try:
            model_filename = generate_model_name(args)
            model.save(settings.FOLDER_MODELS_SERIALISED / model_filename)
        except Exception as exc:
            print(f"Se ha producido un error al guardar el modelo: {exc}")
        else:
            print("Guardado correctamente modelo:")
            print(model_filename)


if __name__ == "__main__":
    main()
