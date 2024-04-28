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


def generate_model_name(args: argparse.Namespace) -> str:
    """Crea el nombre de archivo del modelo
    para parsearlo en función de los
    argumentos pasados

    Parameters
    ----------
    args : argparse.Namespace
        _description_

    Returns
    -------
    str
        _description_
    """
    filename_parts = ["model"]
    for key, value in vars(args).items():
        if (value is not None) and (key != "save"):
            part = f"{key}={value}"
            filename_parts.append(part)

    filename = "_".join(filename_parts) + ".joblib"
    return filename
