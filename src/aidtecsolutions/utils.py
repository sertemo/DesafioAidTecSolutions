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

" Script con funciones auxiliares para todo el proyectos"

from pathlib import Path

import pandas as pd

from aidtecsolutions.custom_exceptions import NonValidDataset


def is_valid_dataset(file_name: str, folder: Path) -> bool:
    """Comprueba que el archivo esté en un determinado
    directorio

    Parameters
    ----------
    file_name : str
        _description_
    folder : Path
        _description_

    Returns
    -------
    bool
        _description_
    """
    lista_archivos = [archivo.name for archivo in folder.iterdir()]
    return file_name in lista_archivos


def is_valid_dataframe(file_name_path: Path, file_name: str) -> pd.DataFrame:
    """Comprueba si un dataset es válido

    Parameters
    ----------
    file_name_path : Path
        _description_
    file_name : str
        _description_

    Returns
    -------
    pd.DataFrame | None
        Devuelve el dataset

    Raises
    ------
    NonValidDataset
        Si el dataset no es válido
    """
    try:
        df = pd.read_csv(file_name_path / file_name, index_col=0)
    except Exception as err:
        raise NonValidDataset(f"El dataset no es válido. Error: {err}")
    else:
        return df
