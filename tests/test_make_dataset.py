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
import requests_mock

from aidtecsolutions.custom_exceptions import DatasetDownloadError
from aidtecsolutions.data.utils import download_dataset
import settings


def test_download_dataset_success(tmp_path, train_url, monkeypatch):
    # Configura una URL de prueba y el nombre del archivo
    file_name = "train.csv"
    
    # Usar monkeypatch para establecer FOLDER_DATA_RAW temporalmente a tmp_path
    monkeypatch.setattr(settings, 'FOLDER_DATA_RAW', tmp_path)

    # Contenido simulado que se "descargaría"
    data = b"contenido del archivo dataset de la web de kopuru."

    # Configurar requests_mock para simular la respuesta de una solicitud GET
    with requests_mock.Mocker() as m:
        m.get(train_url, status_code=200, content=data)  # Genera respuesta falsa
        
        # Ejecutar la función a testear
        download_dataset(train_url, file_name)
        
        # Comprobar que el archivo fue creado y contiene el contenido correcto
        file_path = tmp_path / file_name
        assert file_path.exists()
        with open(file_path, "rb") as f:
            file_contents = f.read()
        print(f"Archivo {file_name} ha sido guardado exitosamente en {file_path}")
        assert file_contents == data

def test_download_dataset_failure(train_url):
    # URL de prueba y nombre de archivo
    file_name = "train.csv"
    train_url = train_url + 'ruido'

    # Configurar requests_mock para simular una falla en la descarga
    with requests_mock.Mocker() as m:
        m.get(train_url, status_code=404)

        # Simular stdout para capturar el output de la función
        with pytest.raises(DatasetDownloadError) as excinfo:
            download_dataset(train_url, file_name)
        assert "Error al descargar el archivo. Estado: 404" in str(excinfo.value)
