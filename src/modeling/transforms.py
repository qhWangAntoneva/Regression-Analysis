"""Variable transformation engine.

Provides the ``VariableTransformer`` class that applies common data
transformations (log, standardize, center, square) to selected columns.
Each transform returns both the modified DataFrame and metadata describing
what was done, so that downstream display code can report results clearly.

**Important:**  Column names produced by this module avoid characters that
patsy would interpret as function calls (parentheses) or interaction syntax
(colons).  For example, ``log(price)`` becomes column ``price_log``.
The helper :meth:`display_name` converts internal names back to
human-readable form for UI display.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class VariableTransformer:
    """Apply common variable transformations to a DataFrame.

    Supported transforms (accessible via ``SUPPORTED_TRANSFORMS``):

        - ``'log'``:         Natural log, ``ln(x + 1e-10)``.
        - ``'standardize'``: Z-score standardisation (zero mean, unit variance).
        - ``'center'``:      Subtract the mean (zero mean, original std).
        - ``'square'``:      Square the variable.

    The public method :meth:`transform` accepts a dictionary that maps
    variable names to transform types and returns a new DataFrame with
    additional columns as well as a metadata dictionary for later display.

    Usage::

        transformer = VariableTransformer()
        data_out, meta = transformer.transform(
            data,
            {"price": "log", "income": "standardize"},
        )
        # meta == {
        #     "price": {"log": "price_log"},
        #     "income": {"standardize": "income_z"},
        # }
        # transformer.display_name("price_log") -> "log(price)"
    """

    SUPPORTED_TRANSFORMS: tuple[str, ...] = (
        "log",
        "standardize",
        "center",
        "square",
    )

    @staticmethod
    def display_name(col_name: str) -> str:
        """Convert an internal column name to a human-readable form.

        Examples::

            "price_log"    -> "log(price)"
            "income_z"     -> "z(income)"
            "age_c"        -> "c(age)"
            "x1_sq"        -> "x1_sq"
            "x1_x_x2"      -> "x1:x2"
            "x1"           -> "x1"   (unchanged)
        """
        # Strip suffix patterns in reverse-priority order
        if col_name.endswith("_log") and "_log" in col_name:
            base = col_name[:-4]
            return f"log({base})"
        if col_name.endswith("_z") and "_z" in col_name:
            base = col_name[:-2]
            return f"z({base})"
        if col_name.endswith("_c") and "_c" in col_name:
            base = col_name[:-2]
            return f"c({base})"
        if col_name.endswith("_sq") and "_sq" in col_name:
            base = col_name[:-3]
            return f"{base}²"
        # Interaction: var1_x_var2
        if "_x_" in col_name:
            parts = col_name.split("_x_", 1)
            if len(parts) == 2:
                return f"{parts[0]}:{parts[1]}"
        return col_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        data: pd.DataFrame,
        transforms: dict[str, str],
    ) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Apply transformations to the specified variables.

        Args:
            data: Input DataFrame.  It is **not** modified in place; a copy
                is returned with additional columns appended.
            transforms: Mapping of ``{variable_name: transform_type}``.
                Example: ``{"price": "log", "income": "standardize"}``.

        Returns:
            A tuple ``(transformed_df, metadata)`` where:

            - ``transformed_df`` is a new DataFrame containing all original
              columns plus the transformed columns.
            - ``metadata`` is a nested dict keyed by original variable name,
              then by transform type, with the new column name as value.
              Example::

                  {
                      "price": {"log": "price_log"},
                      "income": {"standardize": "income_z"},
                  }

        Raises:
            ValueError: If a variable is not present in ``data`` or if an
                unsupported transform type is requested.
        """
        # Validate inputs -------------------------------------------------
        for var in transforms:
            if var not in data.columns:
                raise ValueError(
                    f"Variable '{var}' not found in data columns."
                )
            ttype = transforms[var]
            if ttype not in self.SUPPORTED_TRANSFORMS:
                raise ValueError(
                    f"Unsupported transform '{ttype}' for '{var}'. "
                    f"Supported: {self.SUPPORTED_TRANSFORMS}"
                )

        df = data.copy()
        metadata: dict[str, dict[str, str]] = {}

        for var in transforms:
            ttype = transforms[var]
            new_col = self._apply(df, var, ttype)
            if var not in metadata:
                metadata[var] = {}
            metadata[var][ttype] = new_col

        return df, metadata

    # ------------------------------------------------------------------
    # Per-transform methods
    # ------------------------------------------------------------------

    @staticmethod
    def _apply(data: pd.DataFrame, var: str, ttype: str) -> str:
        """Apply a single transform to *var* in-place on *data*.

        Returns the new column name.
        """
        col: pd.Series = data[var]

        if ttype == "log":
            new_name = f"{var}_log"
            # Safe log: log(x + epsilon) to avoid -inf for zeros
            data[new_name] = np.log(np.maximum(col, 0) + 1e-10)

        elif ttype == "standardize":
            new_name = f"{var}_z"
            mean = col.mean()
            std = col.std(ddof=0)
            data[new_name] = (col - mean) / (std if std > 0 else 1.0)

        elif ttype == "center":
            new_name = f"{var}_c"
            mean = col.mean()
            data[new_name] = col - mean

        elif ttype == "square":
            new_name = f"{var}_sq"
            data[new_name] = col ** 2

        else:
            # Should not be reachable after validation
            raise ValueError(f"Unknown transform: {ttype}")

        return new_name

    # ------------------------------------------------------------------
    # Interaction-term helpers
    # ------------------------------------------------------------------

    @staticmethod
    def add_interactions(
        data: pd.DataFrame,
        interaction_pairs: list[tuple[str, str]],
    ) -> tuple[pd.DataFrame, list[str]]:
        """Add interaction (product) columns for the given variable pairs.

        .. note::
            This method is a **utility** that creates product columns in
            the DataFrame.  The fitter does **not** currently use these
            columns directly; ``build_formula`` in ``specification.py``
            adds ``var1:var2`` terms that patsy expands automatically.

        Args:
            data: Input DataFrame (not modified in place).
            interaction_pairs: List of ``(var1, var2)`` tuples.

        Returns:
            A tuple ``(new_df, interaction_col_names)`` where
            ``new_df`` contains all original columns plus the new
            product columns.

        Raises:
            ValueError: If any variable in a pair is not a column.
        """
        df = data.copy()
        col_names: list[str] = []

        for v1, v2 in interaction_pairs:
            if v1 not in df.columns:
                raise ValueError(
                    f"Interaction variable '{v1}' not found in data."
                )
            if v2 not in df.columns:
                raise ValueError(
                    f"Interaction variable '{v2}' not found in data."
                )
            new_col = f"{v1}_x_{v2}"
            df[new_col] = df[v1] * df[v2]
            col_names.append(new_col)

        return df, col_names
