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
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class MultiStageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, model_stage1=None, model_extreme=None, model_middle=None
    ) -> None:
        """inicializa los modelos de las 3 etapas.
        Tienen que ser modelos con la firma de Scikit-Learn"""
        self.model_stage1 = model_stage1
        self.model_extreme = model_extreme
        self.model_middle = model_middle

    def fit(
        self, X: NDArray[np.float_] | pd.DataFrame, y: NDArray[np.float_] | pd.DataFrame
    ) -> "MultiStageClassifier":
        self.model_stage1_ = self.model_stage1.fit(X, (y < 5) | (y > 7))
        self.model_extreme_ = self.model_extreme.fit(
            X[(y < 5) | (y > 7)], y[(y < 5) | (y > 7)]
        )
        self.model_middle_ = self.model_middle.fit(
            X[(y >= 5) & (y <= 7)], y[(y >= 5) & (y <= 7)]
        )
        return self

    def predict(self, X: NDArray[np.float_] | pd.DataFrame) -> NDArray[np.float_]:
        # Usar check_is_fitted para asegurar que el modelo ha sido ajustado
        check_is_fitted(self, ["model_stage1_", "model_extreme_", "model_middle_"])

        stage1_pred = self.model_stage1_.predict(X)
        final_pred = np.where(
            stage1_pred, self.model_extreme_.predict(X), self.model_middle_.predict(X)
        )
        return final_pred
