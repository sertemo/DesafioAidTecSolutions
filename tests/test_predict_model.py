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

import pandas as pd
import pytest
from aidtecsolutions.models.predict_model import setup_parser, main

def test_parser_with_no_args():
    parser = setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # Debería fallar sin argumentos requeridos

def test_parser_with_all_args():
    parser = setup_parser()
    args = parser.parse_args(['--data', 'data.csv', '--model', 'model.joblib'])
    assert args.data == 'data.csv'
    assert args.model == 'model.joblib'

def test_invalid_dataset(mocker):
    # Mock de la función is_valid_dataset para retornar False
    mocker.patch('aidtecsolutions.utils.is_valid_dataset', return_value=False)
    # Mock del print para evitar salidas durante el test
    mocker.patch('builtins.print')

    # Revisar si la función main maneja correctamente un dataset inválido
    with pytest.raises(SystemExit):
        main()

def test_valid_dataset(mocker):
    # Mock de la función is_valid_dataset para retornar True
    mocker.patch('aidtecsolutions.utils.is_valid_dataset', return_value=True)
    # Mock de la función is_valid_dataframe para retornar un DataFrame vacío
    mocker.patch('aidtecsolutions.utils.is_valid_dataframe', return_value=pd.DataFrame())
    # Mock del sistema de archivos para simular un archivo de modelo existente
    mocker.patch('pathlib.Path.exists', return_value=True)
    # Mock de la función load y predict del modelo
    mock_model = mocker.Mock()
    mocker.patch('aidtecsolutions.wrappers.SerializableClassifier.load', return_value=mock_model)
    mock_model.predict.return_value = ['prediction']
    # Mock del print para evitar salidas durante el test
    mocker.patch('builtins.print')

    # Ejecutar la función main para ver si procesa correctamente un dataset válido
    main()

    # Asegurar que el método predict fue llamado
    mock_model.predict.assert_called_once()