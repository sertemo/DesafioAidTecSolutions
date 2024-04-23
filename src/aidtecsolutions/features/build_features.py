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

"""Scripts to turn raw data into features for modeling"""

import pandas as pd

import aidtecsolutions.settings as settings
from aidtecsolutions.features.custom_transformers import WineDatasetTransformer


def main() -> None:
    wt = WineDatasetTransformer(
        shuffle=True,
        color_interactions=False,
        densidad_alcohol_interaction=True,
        standardize=False,
        ratio_diox=True,
        rbf_diox=True,
        drop_columns=["color", "year", "densidad", "alcohol"],
    )
    X = pd.read_csv(settings.RUTA_TRAIN_DATASET)

    X_transformed = wt.fit_transform(X)
    print(X_transformed)


if __name__ == "__main__":
    main()
