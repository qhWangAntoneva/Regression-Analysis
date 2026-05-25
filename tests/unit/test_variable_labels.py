# encoding: utf-8
"""Tests for build_variable_labels function."""
import pytest
from src.modeling.specification import ModelSpec, build_variable_labels


def _make_spec():
    return ModelSpec(dep_var="y", indep_vars=["x1"])


class TestBuildVariableLabels:
    def test_intercept(self):
        labels = build_variable_labels(_make_spec(), ["Intercept", "x1"])
        assert labels["Intercept"] == "Intercept"
        assert labels["x1"] == "x1"

    def test_simple_categorical_main_effect(self):
        labels = build_variable_labels(_make_spec(), ["education[T.本科]"])
        assert labels["education[T.本科]"] == "education: 本科"

    def test_categorical_level_with_hyphen(self):
        labels = build_variable_labels(_make_spec(), ["status[T.High-risk]"])
        assert labels["status[T.High-risk]"] == "status: High-risk"

    def test_categorical_x_numeric_interaction(self):
        labels = build_variable_labels(_make_spec(), ["education[T.本科]:income"])
        assert labels["education[T.本科]:income"] == "education: 本科 × income"

    def test_categorical_x_categorical_interaction(self):
        labels = build_variable_labels(_make_spec(), ["gender[T.Male]:education[T.本科]"])
        assert labels["gender[T.Male]:education[T.本科]"] == "gender: Male × education: 本科"

    def test_interaction_only_categorical_no_T_prefix(self):
        labels = build_variable_labels(_make_spec(), ["cat[a]:x"])
        assert labels["cat[a]:x"] == "cat: a × x"

    def test_three_way_interaction(self):
        labels = build_variable_labels(_make_spec(), ["cat[T.a]:cat2[T.b]:x"])
        assert labels["cat[T.a]:cat2[T.b]:x"] == "cat: a × cat2: b × x"

    def test_continuous_only_interaction(self):
        labels = build_variable_labels(_make_spec(), ["income:age"])
        assert labels["income:age"] == "income × age"

    def test_continuous_variable(self):
        labels = build_variable_labels(_make_spec(), ["income"])
        assert labels["income"] == "income"

    def test_mixed_categorical_and_continuous(self):
        labels = build_variable_labels(_make_spec(),
            ["Intercept", "education[T.本科]", "income", "education[T.本科]:income"])
        assert labels["Intercept"] == "Intercept"
        assert labels["education[T.本科]"] == "education: 本科"
        assert labels["income"] == "income"
        assert labels["education[T.本科]:income"] == "education: 本科 × income"

    def test_variable_name_with_underscore(self):
        labels = build_variable_labels(_make_spec(), ["edu_cat[T.high]"])
        assert labels["edu_cat[T.high]"] == "edu_cat: high"
