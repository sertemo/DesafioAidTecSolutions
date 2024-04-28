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

import joblib
from typing import cast, Any

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


class SerializableMixin:
    """Clase que implementa el método save
    para serializar un modelo"""

    def save(self, model_path: str) -> None:
        """Serializa el modelo usando joblib"""
        with open(model_path, "wb") as f:
            joblib.dump(self, f)


class DeserializableMixin(SerializableMixin):
    """Clase para cargar y deserializar
    un modelo guardado

    Parameters
    ----------
    SerializableMixin : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    @classmethod
    def load(cls, model_path: str) -> "DeserializableMixin":
        with open(model_path, "rb") as f:
            classifier = cast(DeserializableMixin, joblib.load(f))
        return classifier


class SerializableClassifier(
    BaseEstimator,
    ClassifierMixin,
    DeserializableMixin,
):
    """Wrapper de un clasificador para
    que sea serializable y deserializable
    con joblib

    Parameters
    ----------
    BaseEstimator : _type_
        _description_
    ClassifierMixin : _type_
        _description_
    DeserializableMixin : _type_
        _description_
    """

    def __init__(self, classifier: BaseEstimator) -> None:
        self.classifier: BaseEstimator = classifier

    def fit(
        self,
        X: NDArray[np.float64] | pd.DataFrame,
        y: NDArray[np.float64] | pd.DataFrame,
    ) -> "SerializableClassifier":
        self.classifier.fit(X, y)
        return self

    def predict(self, X: NDArray[np.float64] | pd.DataFrame) -> NDArray[np.int64]:
        check_is_fitted(self.classifier)
        predictions: NDArray[np.int64] = self.classifier.predict(X)
        return predictions

    def __getattr__(self, attr: Any) -> Any:
        """Delega atributos al clasificador subyacente si no se encuentran en 'self'."""
        return getattr(self.classifier, attr)
