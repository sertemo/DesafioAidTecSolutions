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

"""Scripts con funciones auxiliares relacionadas make_dataset"""

import requests

from aidtecsolutions.custom_exceptions import DatasetDownloadError
import settings


def download_dataset(url: str, nombre_archivo: str) -> None:
    """Hace una solicitud GET y descarga un archivo
    de la url. Lo guarda en data/raw

    Parameters
    ----------
    url : str
        _description_
    nombre_archivo : str
        _description_

    Raises
    -------
    DatasetDoenloadError
        Si la conexión no se ha realizado
        correctamente
    """
    respuesta = requests.get(url)
    print(f"GET {url}")

    # Verificar que la solicitud fue exitosa
    if respuesta.status_code == 200:
        # Abrir un archivo en modo escritura binaria
        ruta_completa = settings.FOLDER_DATA_RAW / nombre_archivo
        with open(ruta_completa, "wb") as f:
            f.write(respuesta.content)
        print(f"Archivo guardado en {ruta_completa}")
    else:
        err = f"Error al descargar el archivo. Estado: {respuesta.status_code}"
        print(err)
        raise DatasetDownloadError(err)
