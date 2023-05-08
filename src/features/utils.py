"""Module containing a class to transform features for the pipeline."""
from typing import Any

import pandas as pd
from sklearn import base


class TabularToDict(base.BaseEstimator, base.TransformerMixin):
    """Class to transform tabular data into dictionary or list of dictionaries."""

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None
    ) -> "TabularToDict":
        """Return self instance.

        Args:
            X: features.
            y: target.

        Returns:
            Self instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> dict[str, Any] | list[dict[str, Any]]:
        """Transform features into dictionary or list of dictionaries.

        Args:
            X: features.

        Returns:
            Transformed features.
        """
        return X.to_dict(orient="records")
