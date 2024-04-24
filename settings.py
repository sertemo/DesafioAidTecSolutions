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

"""Scripts con rutas y constantes del proyecto"""

from pathlib import Path

# Rutas
FOLDER_DATA_RAW = Path("data/raw")
FOLDER_DATA_PROCESSED = Path("data/processed")
FOLDER_DATA_INTERIM = Path("data/interim")
FOLDER_MODELS_SERIALISED = Path("models")

TRAIN_DATASET = "calidad_vino_AT-_train.csv"
TEST_DATASET = "calidad_vino_AT-_test.csv"
RUTA_TRAIN_DATASET_RAW = FOLDER_DATA_RAW / TRAIN_DATASET
RUTA_TEST_DATASET_RAW = FOLDER_DATA_RAW / TEST_DATASET
