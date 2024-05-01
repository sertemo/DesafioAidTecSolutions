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

import pytest

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from aidtecsolutions.models.train_model import setup_parser
from aidtecsolutions.models.utils import generate_model_name
from aidtecsolutions.models.custom_models import SerializableClassifier

def test_model_parser_with_valid_args() -> None:
    parser = setup_parser()
    args = parser.parse_args([
        '--data', 'data.csv',
        'xgb',
        '--gamma', '0.1',
        '--max_depth', '5',
        '--alpha', '0.5',
        '--learning_rate', '0.01',
        '--n_estimators', '150'
    ])
    
    assert args.data == 'data.csv'
    assert args.model == 'xgb'
    assert args.gamma == 0.1
    assert args.max_depth == 5
    assert args.alpha == 0.5
    assert args.learning_rate == 0.01
    assert args.n_estimators == 150

def test_parser_with_invalid_args() -> None:
    parser = setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--data', 'data.csv', 'xgb', '--gamma'])

def test_parser_bad_arg_type() -> None:
    parser = setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--data', 'data.csv', 'xgb', '--gamma', 'no_valid'])

def test_parser_bad_arg_randomfirest() -> None:
    parser = setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
        '--data', 'data.csv',
        'randomforest',
        '--gamma', '0.1',
        '--max_depth', '5',
        '--alpha', '0.5', # bad for rf
        '--learning_rate', '0.01',
        '--n_estimators', '150'
    ])

def test_correct_model_name() -> None:
    parser = setup_parser()
    args = parser.parse_args([
        '--data', 'train-corregir_alcohol.csv',
        'randomforest',
        '--n_estimators', '900',
    ])
    name = generate_model_name(args)
    assert name == 'model_data=train-corregir_alcohol.csv_model=randomforest_n_estimators=900_criterion=gini.joblib'

def test_correct_model_name_with_save() -> None:
    parser = setup_parser()
    args = parser.parse_args([
        '--data', 'train-corregir_alcohol.csv',
        '--save',
        'randomforest',
        '--n_estimators', '900',
    ])
    name = generate_model_name(args)
    assert name == 'model_data=train-corregir_alcohol.csv_model=randomforest_n_estimators=900_criterion=gini.joblib'

def test_incorrect_model_name() -> None:
    parser = setup_parser()
    args = parser.parse_args([
        '--data', 'train-corregir_alcohol.csv',
        'randomforest',
        '--n_estimators', '900',
        '--criterion', 'gini'
    ])
    name = generate_model_name(args)
    assert name != 'model_data=train-corregir_alcohol.csv_save=False_model=randomforest_n_estimators=950.joblib'


def test_initialization():
    """Prueba que el clasificador se inicializa correctamente."""
    dt = RandomForestClassifier()
    wrapper = SerializableClassifier(dt)
    assert wrapper.classifier is dt

def test_fit_and_predict():
    """Prueba los métodos fit y predict."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    dt = RandomForestClassifier()
    wrapper = SerializableClassifier(dt)

    # Test fit method
    trained_wrapper = wrapper.fit(X, y)
    assert trained_wrapper is wrapper  # tiene que devolverse él mismo

    # Test predict method
    predictions = trained_wrapper.predict(X)
    assert np.array_equal(predictions, y)  # suponemos fit perfecto

def test_delegation_to_underlying_classifier():
    """Prueba que los atributos no definidos en el wrapper se deleguen correctamente."""
    dt = RandomForestClassifier(n_estimators=123)
    wrapper = SerializableClassifier(dt)
    
    # Acceder directamente a un atributo del clasificador subyacente
    assert wrapper.n_estimators == 123

def test_not_fitted_error():
    """Prueba que la verificación de modelo ajustado funcione correctamente."""
    dt = RandomForestClassifier()
    wrapper = SerializableClassifier(dt)
    X = np.array([[1, 2], [3, 4]])

    with pytest.raises(NotFittedError):
        _ = wrapper.predict(X)

def test_cross_val_score_integration():
    """Prueba que SerializableClassifier funcione con cross_val_score."""
    data = load_iris()
    X, y = data.data, data.target
    dt = DecisionTreeClassifier()
    wrapper = SerializableClassifier(dt)

    # Probar que no hay errores al usar cross_val_score
    scores = cross_val_score(wrapper, X, y, cv=3)
    assert scores.size == 3
    assert scores.mean() > 0  # Comprobar que los scores tienen sentido

def test_cross_val_predict_integration():
    """Prueba que SerializableClassifier funcione con cross_val_predict."""
    data = load_iris()
    X, y = data.data, data.target
    dt = DecisionTreeClassifier()
    wrapper = SerializableClassifier(dt)

    # Probar que no hay errores al usar cross_val_predict
    predictions = cross_val_predict(wrapper, X, y, cv=3)
    assert len(predictions) == len(y)