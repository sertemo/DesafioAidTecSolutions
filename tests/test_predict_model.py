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

from aidtecsolutions.models.predict_model import PredictModel
import settings

def test_parser_with_no_args():
    pm = PredictModel()
    parser = pm._setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # Debería fallar sin argumentos requeridos

def test_parser_configuration():
    pm = PredictModel()
    parser = pm._setup_parser()
    # Verifica que todas las opciones requeridas estén presentes
    assert 'data' in parser.format_help()
    assert 'model' in parser.format_help()
    assert 'merge' in parser.format_help()

def test_parser_with_all_args():
    pm = PredictModel()
    parser = pm._setup_parser()
    args = parser.parse_args(['--data', 'data.csv', '--model', 'model.joblib'])
    assert args.data == 'data.csv'
    assert args.model == 'model.joblib'

def test_file_check_and_predictions(predict_model, mocker):
    # Mocks para los checks de archivos y para cargar datos/modelos
    mocker.patch('aidtecsolutions.utils.is_valid_dataset', return_value=True)
    mocker.patch('aidtecsolutions.utils.is_valid_dataframe', return_value=pd.DataFrame())
    mocker.patch('pathlib.Path.exists', return_value=True)
    mocker.patch('builtins.print')  # suprimir la salida de print

    mock_model = mocker.Mock()
    mocker.patch('aidtecsolutions.wrappers.SerializableClassifier.load', return_value=mock_model)
    mock_transformer = mocker.Mock()
    mocker.patch('aidtecsolutions.wrappers.SerializableTransformer.load', return_value=mock_transformer)

    mock_model.predict.return_value = np.array([2, 3, 4])  # Simular predicciones
    transform_preds = np.array([5, 6, 7])
    mock_transformer.inverse_transform.return_value = transform_preds

    # Simular argumentos de línea de comandos
    mocker.patch('argparse.ArgumentParser.parse_args',
                return_value=mocker.Mock(data='data.csv', model='model.joblib', merge=None))

    predict_model.main()

    # Verificaciones
    assert mock_model.preds is not None