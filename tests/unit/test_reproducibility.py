# encoding: utf-8
"""单元测试：分析复现包导出。

测试 DataExporter.export_reproducibility_package() 方法。
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import pytest

from src.data_io.exporter import DataExporter
from src.results.table import CoefficientRow, ModelResult


# =========================================================================
# Helper: minimal model spec
# =========================================================================


@dataclass
class FakeModelSpec:
    """模拟 ModelSpec 用于测试。"""
    dep_var: str
    indep_vars: list = field(default_factory=list)
    control_vars: list = field(default_factory=list)
    has_intercept: bool = True


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """创建一个简单的回归数据集。"""
    return pd.DataFrame({
        "y": [3.5, 5.2, 7.1, 8.8, 10.5],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def model_result() -> ModelResult:
    """创建一个模型结果。"""
    return ModelResult(
        model_type="OLS",
        coefficients=[
            CoefficientRow(
                name="Intercept", coef=2.0, se=0.5, t_stat=4.0,
                pvalue=0.001, ci_lower=1.0, ci_upper=3.0,
            ),
            CoefficientRow(
                name="x1", coef=0.5, se=0.1, t_stat=5.0,
                pvalue=0.0001, ci_lower=0.3, ci_upper=0.7,
            ),
        ],
        n_obs=5,
        n_params=2,
        df_resid=3,
        r_squared=0.95,
        adj_r_squared=0.93,
        rmse=0.35,
        dep_var="y",
        aic=10.0,
        bic=12.0,
        specification="y ~ x1",
        f_statistic=(57.0, 0.0001),
    )


@pytest.fixture
def temp_dir() -> str:
    """创建一个临时目录。"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # 清理
    for f in Path(tmpdir).glob("*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                import shutil
                shutil.rmtree(f)
        except Exception:
            pass
    try:
        Path(tmpdir).rmdir()
    except Exception:
        pass


# =========================================================================
# Test: export_reproducibility_package
# =========================================================================


class TestExportReproducibilityPackage:
    """测试分析复现包导出。"""

    def test_export_basic(
        self, sample_data: pd.DataFrame, model_result: ModelResult, temp_dir: str
    ) -> None:
        """基本复现包导出。"""
        spec = FakeModelSpec(dep_var="y", indep_vars=["x1"])
        zip_path = DataExporter.export_reproducibility_package(
            sample_data, spec, model_result, temp_dir,
        )
        assert os.path.exists(zip_path)
        assert zip_path.endswith(".zip")

    def test_zip_contents(
        self, sample_data: pd.DataFrame, model_result: ModelResult, temp_dir: str
    ) -> None:
        """ZIP 应包含预期的所有文件。"""
        spec = FakeModelSpec(dep_var="y", indep_vars=["x1"])
        zip_path = DataExporter.export_reproducibility_package(
            sample_data, spec, model_result, temp_dir,
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "data.csv" in names
            assert "model_config.json" in names
            assert "reproduce.py" in names
            assert "results_summary.txt" in names

    def test_data_csv_content(
        self, sample_data: pd.DataFrame, model_result: ModelResult, temp_dir: str
    ) -> None:
        """data.csv 应包含相同的数据。"""
        spec = FakeModelSpec(dep_var="y", indep_vars=["x1"])
        zip_path = DataExporter.export_reproducibility_package(
            sample_data, spec, model_result, temp_dir,
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("data.csv") as f:
                loaded = pd.read_csv(f, encoding="utf-8-sig")
        assert loaded.shape == sample_data.shape
        assert list(loaded.columns) == list(sample_data.columns)

    def test_model_config_json(
        self, sample_data: pd.DataFrame, model_result: ModelResult, temp_dir: str
    ) -> None:
        """model_config.json 应包含正确字段。"""
        spec = FakeModelSpec(
            dep_var="y", indep_vars=["x1"], control_vars=["x2"], has_intercept=True,
        )
        zip_path = DataExporter.export_reproducibility_package(
            sample_data, spec, model_result, temp_dir,
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            config = json.loads(zf.read("model_config.json"))
            assert config["dep_var"] == "y"
            assert config["indep_vars"] == ["x1"]
            assert config["control_vars"] == ["x2"]
            assert config["has_intercept"] is True
            assert config["model_type"] == "OLS"

    def test_reproduce_script_content(
        self, sample_data: pd.DataFrame, model_result: ModelResult, temp_dir: str
    ) -> None:
        """reproduce.py 应包含 statsmodels 代码。"""
        spec = FakeModelSpec(dep_var="y", indep_vars=["x1"])
        zip_path = DataExporter.export_reproducibility_package(
            sample_data, spec, model_result, temp_dir,
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            script = zf.read("reproduce.py").decode("utf-8")

        assert "import statsmodels" in script
        assert "import pandas" in script
        assert "data.csv" in script
        assert "y ~ x1" in script or "formula" in script

    def test_results_summary_exists(
        self, sample_data: pd.DataFrame, model_result: ModelResult, temp_dir: str
    ) -> None:
        """results_summary.txt 应存在且非空。"""
        spec = FakeModelSpec(dep_var="y", indep_vars=["x1"])
        zip_path = DataExporter.export_reproducibility_package(
            sample_data, spec, model_result, temp_dir,
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            content = zf.read("results_summary.txt").decode("utf-8")

        assert len(content) > 0

    def test_export_empty_data(self, temp_dir: str) -> None:
        """空 DataFrame 应抛出 ValueError。"""
        empty_df = pd.DataFrame()
        spec = FakeModelSpec(dep_var="y", indep_vars=["x1"])
        from src.results.table import ModelResult

        with pytest.raises(ValueError, match="为空"):
            DataExporter.export_reproducibility_package(
                empty_df, spec, ModelResult(
                    model_type="OLS", coefficients=[], n_obs=0, n_params=0, df_resid=0,
                ),
                temp_dir,
            )

    def test_zip_not_empty(
        self, sample_data: pd.DataFrame, model_result: ModelResult, temp_dir: str
    ) -> None:
        """ZIP 文件不应为空。"""
        spec = FakeModelSpec(dep_var="y", indep_vars=["x1"])
        zip_path = DataExporter.export_reproducibility_package(
            sample_data, spec, model_result, temp_dir,
        )

        file_size = os.path.getsize(zip_path)
        assert file_size > 0, "ZIP 文件不应为空"
