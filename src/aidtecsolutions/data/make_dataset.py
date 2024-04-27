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

"""Scripts para descargar o generar la data"""

import argparse

from aidtecsolutions.custom_exceptions import DatasetDownloadError
from aidtecsolutions.data.utils import download_dataset
import settings


def main() -> None:
    parser = argparse.ArgumentParser()

    # Añadimos argumentos
    parser.add_argument(
        "--train",
        help="Descarga el dataset de train de la web de kopuru",
        action="store_true",
    )

    parser.add_argument(
        "--test",
        help="Descarga el dataset de test la web de kopuru",
        action="store_true",
    )

    # Parseamos los argumentos
    args = parser.parse_args()

    if args.train:
        try:
            download_dataset(settings.TRAIN_URL, "train.csv")
        except DatasetDownloadError as exc:
            print(f"Error al descargar el dataset: {exc}")
    elif args.test:
        try:
            download_dataset(settings.TEST_URL, "test.csv")
        except DatasetDownloadError as exc:
            print(f"Error al descargar el dataset: {exc}")


if __name__ == "__main__":
    main()
