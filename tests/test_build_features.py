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

import argparse

import numpy as np
import pandas as pd
import pytest

from aidtecsolutions.features.custom_transformers import WineDatasetTransformer
from aidtecsolutions.features.build_features import setup_parser
from aidtecsolutions.features.utils import generate_dataset_name, parse_col_name
from aidtecsolutions.custom_exceptions import (
    WrongColumnName,
    WrongColumnType
)

def test_feature_parser_with_valid_args() -> None:
    parser = setup_parser()
    args = parser.parse_args([
        '--con', 'train.csv',
        '--alcohol',
        '--densidad',
        '--densidad_alcohol',
        '--save',
    ])
    
    assert args.con == 'train.csv'
    assert args.alcohol == True
    assert args.densidad == True
    assert args.densidad_alcohol == True
    assert args.save == True

def test_feature_parser_with_incorrect_args() -> None:
    parser = setup_parser()    
    with pytest.raises(SystemExit):
        args = parser.parse_args([
            #'--con', 'train.csv',
            '--alcohol',
            '--densidad',
            '--densidad_alcohol',
            '--save',
        ])


def test_feature_parser_with_incorrect_args_error_msg(capfd) -> None:
    parser = setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--alcohol'])  # Intencionalmente omitiendo '--con'

    out, err = capfd.readouterr()
    assert "the following arguments are required: --con" in err


def test_corregir_alcohol_type_float(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=True,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed = wt.fit_transform(train_raw)
    assert train_transformed['alcohol'].dtype == 'float64'


def test_corregir_densidad_type_float(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=True,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed = wt.fit_transform(train_raw)
    assert train_transformed['densidad'].dtype == 'float64'


def test_corregir_densidad_max_value(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=True,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed = wt.fit_transform(train_raw)
    # La densidad del vino tiene que ser inferior a 2 g/cm³. Debe rondar 1
    assert train_transformed['densidad'].max() < 2


def test_binarizar_color_por_defecto(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    # La variable color tiene que estar binarizada
    assert train_transformed['color'].max() == 1 and (len(train_transformed['color'].value_counts()) == 2)


def test_drop_good_columns(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=['color', 'alcohol'],
        shuffle=False,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    # La variable color tiene que estar binarizada
    assert ('color' not in train_transformed.columns) and ('alcohol' not in train_transformed.columns)


def test_densidad_alcohol_interaction(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=True,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=True,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    assert 'densidad_alcohol' in train_transformed.columns, "'densidad_alcohol' no se encuentra en las columnas"
    expected = (train_transformed['alcohol'] * train_transformed['densidad'])
    assert np.isclose(train_transformed['densidad_alcohol'], expected, atol=1e-8).all()


def test_remove_outliers_menos_muestras(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=True,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=True,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    assert hasattr(wt, 'outlier_pred'), "WineDatasetTransformer no tiene el parámetro 'outlier_pred"
    assert len(train_transformed) < len(train_raw), 'No se ha producido correctamente el recorte de outliers'


def test_remove_outliers_sin_corregir_alcohol(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=True,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    with pytest.raises(WrongColumnType):
        train_transformed: pd.DataFrame = wt.fit_transform(train_raw)


def test_densidad_alcohol_interaction_sin_corregir_alcohol(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=True,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    with pytest.raises(WrongColumnType):
        train_transformed: pd.DataFrame = wt.fit_transform(train_raw)


def test_estandarizar_mismo_num_columnas_sin_corregir_alcohol_densidad(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=True,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    assert len(train_transformed.columns) == len(train_raw.columns), f'El número de columnas no coincide: \
        {len(train_transformed.columns)} != {len(train_raw.columns)}'


def test_estandarizar_mismo_num_columnas_corrigiend_alcohol_densidad(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=True,
        corregir_densidad=True,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=True,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    assert len(train_transformed.columns) == len(train_raw.columns), f'El número de columnas no coincide: \
        {len(train_transformed.columns)} != {len(train_raw.columns)}'


def test_ninguna_transformacion_comprobar_indices(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    assert (train_transformed.index == train_raw.index).all(), 'Los índices no coinciden'


def test_shuffle_comprobar_distintos_indices_misma_len(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=True,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    # Comprobamos que haya distintos indices
    assert (train_transformed.index != train_raw.index).any(), 'Los índices coinciden, no se ha shuffleado'
    # Comprobamos que la longitud sea la misma
    assert len(train_transformed) == len(train_raw), 'Las longitudes no coinciden y deberían'


def test_ratio_diox_correcto(train_raw):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=True,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=None,
        shuffle=False,
    )
    train_transformed: pd.DataFrame = wt.fit_transform(train_raw)
    assert 'SO2_l / SO2_tot' in train_transformed.columns, "'ratio_diox' no se encuentra en las columnas"
    expected = (
        train_transformed['dioxido de azufre libre'] / train_transformed['dioxido de azufre total'])
    assert np.isclose(train_transformed['SO2_l / SO2_tot'], expected, atol=1e-8).all()


def test_log_column_wrong(train_raw):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=['bob', 'foo'],
        drop_columns=None,
        shuffle=False,
    )
    with pytest.raises(WrongColumnName):
        train_transformed = wt.fit_transform(train_raw)


def test_drop_column_wrong(train_raw: pd.DataFrame):
    wt = WineDatasetTransformer(
        corregir_alcohol=False,
        corregir_densidad=False,
        color_interactions=False,
        densidad_alcohol_interaction=False,
        ratio_diox=False,
        rbf_diox=False,
        remove_outliers=False,
        standardize=False,
        log_transformation=None,
        drop_columns=['bob', 'foo'],
        shuffle=False,
    )
    with pytest.raises(WrongColumnName):
        train_transformed = wt.fit_transform(train_raw)


def test_generate_dataset_alcohol_densidad_drop():
    # Simula los argumentos
    args = argparse.Namespace(
        con="train.csv",
        alcohol=True,
        densidad=True,
        color=False,
        densidad_alcohol=False,
        ratiodiox=False,
        rbfdiox=False,
        outliers=False,
        estandarizar=False,
        shuffle=False,
        drop=["color", "year"],
        log=None
    )

    # Llamada a la función
    result = generate_dataset_name(args)

    # Verificar el resultado
    expected = "train.csv-corregir_alcohol-corregir_densidad-drop=color-year.csv"
    assert result == expected


def test_generate_dataset_name_some_false():
    # Configuración con algunos argumentos False
    args = argparse.Namespace(
        con="dataset_base.csv",
        alcohol=False,
        densidad=False,
        color=True,
        densidad_alcohol=False,
        ratiodiox=False,
        rbfdiox=False,
        outliers=True,
        estandarizar=False,
        shuffle=False,
        drop=[],
        log=None
    )

    # Llamada a la función
    result = generate_dataset_name(args)

    # Verificar el resultado
    expected = "dataset_base.csv-color_interactions-remove_outliers.csv"
    assert result == expected


def good_parsed_col_name() -> None:
    """Parsea correctamente el nombre de las columnas
    """
    list_col_names = ['color', 'alcohol', 'dioxido_de_azufre_libre ']  # Extra espacio al final  
    expected = ['color', 'alcohol', 'dioxido de azufre libre']
    assert parse_col_name(list_col_names) == expected
