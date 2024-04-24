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

import numpy as np
import pandas as pd
import pytest

from aidtecsolutions.features.custom_transformers import WineDatasetTransformer
from aidtecsolutions.custom_exceptions import (
    WrongColumnName,
    WrongColumnType
)


def test_corregir_alcohol_type_float(train_raw):
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


def test_corregir_densidad_type_float(train_raw):
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


def test_corregir_densidad_max_value(train_raw):
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


def test_binarizar_color_por_defecto(train_raw):
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


def test_drop_good_columns(train_raw):
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


def test_densidad_alcohol_interaction(train_raw):
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


def test_remove_outliers_menos_muestras(train_raw):
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


def test_remove_outliers_sin_corregir_alcohol(train_raw):
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


def test_densidad_alcohol_interaction_sin_corregir_alcohol(train_raw):
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


def test_drop_column_wrong(train_raw):
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