# encoding: utf-8
"""
样本数据集生成模块

提供预定义的数据集，用户可一键加载到应用中进行探索和分析。
所有数据集通过 pandas/numpy 实时生成，无需外部文件。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def get_sample_datasets() -> dict[str, dict[str, Any]]:
    """返回所有可用样本数据集的元信息。

    Returns:
        {
            '房价数据': {'description': ..., 'source': ..., 'n_rows': ..., 'n_cols': ...},
            '工资数据': {...},
            '空气质量数据': {...},
        }
    """
    return {
        "房价数据": {
            "description": "房屋价格与特征数据，适合用于多元线性回归分析房价的影响因素。",
            "source": "模拟生成（基于常见房价数据模式）",
            "n_rows": 500,
            "n_cols": 8,
        },
        "工资数据": {
            "description": "工资收入与个人特征数据，可用于分析教育、经验等因素对收入的影响。",
            "source": "模拟生成（基于劳动经济学常见变量）",
            "n_rows": 400,
            "n_cols": 7,
        },
        "空气质量数据": {
            "description": "空气质量指数与气象数据，可用于分析污染物与气象条件的关系。",
            "source": "模拟生成（基于环境监测常见指标）",
            "n_rows": 300,
            "n_cols": 6,
        },
    }


def load_housing_data() -> pd.DataFrame:
    """生成房价模拟数据集（500行 x 8列）。"""
    rng = np.random.default_rng(42)

    n = 500
    sqft = rng.normal(1500, 500, n).clip(500, 5000)
    bedrooms = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.2, 0.4, 0.25, 0.1])
    age = rng.uniform(0, 50, n)
    location_score = rng.uniform(1, 10, n)
    floor = rng.choice([1, 2, 3, 4, 5, 6, 7, 8], n)
    has_garage = rng.choice([0, 1], n, p=[0.3, 0.7])

    # 房价生成公式
    price = (
        50000
        + 200 * sqft
        + 15000 * bedrooms
        - 1000 * age
        + 30000 * location_score
        + 10000 * floor
        + 20000 * has_garage
        + rng.normal(0, 50000, n)
    ).clip(50000, None)

    df = pd.DataFrame({
        "price": price.round(0).astype(int),
        "sqft": sqft.round(0).astype(int),
        "bedrooms": bedrooms.astype(int),
        "age": age.round(1),
        "location_score": location_score.round(2),
        "floor": floor.astype(int),
        "has_garage": has_garage.astype(int),
    })

    # 模拟少量缺失值
    missing_idx = rng.choice(n, size=8, replace=False)
    df.loc[missing_idx, "age"] = np.nan

    return df


def load_wages_data() -> pd.DataFrame:
    """生成工资模拟数据集（400行 x 7列）。"""
    rng = np.random.default_rng(42)

    n = 400
    education = rng.choice(["高中以下", "高中", "本科", "硕士", "博士"], n, p=[0.1, 0.25, 0.35, 0.2, 0.1])
    experience = rng.uniform(0, 40, n)
    gender = rng.choice(["男", "女"], n, p=[0.5, 0.5])
    industry = rng.choice(["制造业", "服务业", "IT", "金融", "教育", "医疗"], n, p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
    hours_per_week = rng.normal(40, 8, n).clip(20, 80)
    years_at_company = rng.uniform(0, 25, n)

    # 教育回报映射
    edu_map = {"高中以下": 0, "高中": 1, "本科": 2, "硕士": 3, "博士": 4}
    edu_years = np.array([edu_map[e] for e in education])
    edu_premium = edu_years * 5000

    industry_map = {"制造业": 0, "服务业": 0, "IT": 1, "金融": 1, "教育": 0, "医疗": 0}
    industry_premium = np.array([industry_map[i] for i in industry]) * 10000

    wage = (
        3000
        + edu_premium
        + 500 * experience
        + industry_premium
        + 200 * (hours_per_week - 40)
        + 200 * years_at_company
        + rng.normal(0, 8000, n)
    ).clip(2500, None)

    df = pd.DataFrame({
        "wage": wage.round(0).astype(int),
        "education": education,
        "experience": experience.round(1),
        "gender": gender,
        "industry": industry,
        "hours_per_week": hours_per_week.round(1),
        "years_at_company": years_at_company.round(1),
    })

    # 模拟少量缺失值
    missing_idx = rng.choice(n, size=5, replace=False)
    df.loc[missing_idx, "years_at_company"] = np.nan

    return df


def load_air_quality_data() -> pd.DataFrame:
    """生成空气质量模拟数据集（300行 x 6列）。"""
    rng = np.random.default_rng(42)

    n = 300
    temp = rng.normal(20, 10, n)
    humidity = rng.uniform(30, 90, n)
    wind = rng.weibull(2, n) * 5

    # AQI 生成
    aqi = (
        30
        + 0.5 * np.abs(temp - 20)
        - 0.3 * wind
        + 0.2 * (humidity - 60)
        + rng.normal(0, 15, n)
    ).clip(0, 300)

    pm25 = (
        20
        + 0.4 * aqi
        + 0.1 * (humidity - 50)
        - 0.5 * wind
        + rng.normal(0, 10, n)
    ).clip(0, None)

    pm10 = pm25 * rng.uniform(1.2, 2.0, n)

    df = pd.DataFrame({
        "AQI": aqi.round(0).astype(int),
        "PM2.5": pm25.round(1),
        "PM10": pm10.round(1),
        "temp": temp.round(1),
        "humidity": humidity.round(1),
        "wind": wind.round(1),
    })

    # 模拟少量缺失值
    missing_idx = rng.choice(n, size=6, replace=False)
    df.loc[missing_idx, "PM2.5"] = np.nan

    missing_idx2 = rng.choice(n, size=4, replace=False)
    df.loc[missing_idx2, "wind"] = np.nan

    return df


def load_sample_dataset(name: str) -> pd.DataFrame:
    """按名称加载样本数据集。

    Args:
        name: 数据集名称，可选 '房价数据'、'工资数据'、'空气质量数据'。

    Returns:
        对应的 pandas DataFrame。
    """
    loaders = {
        "房价数据": load_housing_data,
        "工资数据": load_wages_data,
        "空气质量数据": load_air_quality_data,
    }

    loader = loaders.get(name)
    if loader is None:
        raise ValueError(f"未知数据集: {name}。可选: {list(loaders.keys())}")

    return loader()
