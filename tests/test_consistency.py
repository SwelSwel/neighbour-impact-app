"""Training/production consistency: the app must rebuild the notebook's features and predictions exactly."""
import math

import numpy as np
import pytest

from app.config import settings
from app.predictor import Predictor, prediction_slot
from datetime import datetime


@pytest.fixture(scope="module")
def predictor():
    return Predictor(settings.model_path)


def test_feature_order_matches_bundle(predictor):
    assert len(predictor.features) == 66
    assert len(set(predictor.features)) == 66


def test_features_match_notebook(predictor, cases):
    for i, c in enumerate(cases):
        got = np.array(predictor.build_row(**c["inputs"]), dtype=float)
        exp = np.array([np.nan if v is None else v for v in c["expected_features"]], dtype=float)
        both_nan = np.isnan(got) & np.isnan(exp)
        assert np.array_equal(np.isnan(got), np.isnan(exp)), f"case {i}: missing values differ"
        assert np.allclose(got[~both_nan], exp[~both_nan], rtol=1e-5, atol=1e-6), f"case {i}: features differ"


def test_predictions_match_notebook(predictor, cases):
    rows = [predictor.build_row(**c["inputs"]) for c in cases]
    preds = predictor.predict_rows(rows)
    for t in predictor.targets:
        exp = np.array([c["expected_predictions"][t] for c in cases])
        assert np.allclose(preds[t], exp, rtol=1e-6, atol=1e-6), f"{t}: predictions differ from the notebook"


def test_predictions_are_reasonable(predictor, cases):
    """Sanity check on real outages: predictions are finite and close to the observed values on average."""
    rows = [predictor.build_row(**c["inputs"]) for c in cases]
    preds = predictor.predict_rows(rows)
    for t in predictor.targets:
        actual = np.array([c["actual"][t] for c in cases])
        assert np.all(np.isfinite(preds[t]))
        assert np.mean(np.abs(preds[t] - actual)) < np.mean(np.abs(actual))   # far better than predicting zero


@pytest.mark.parametrize("start, slot", [
    ("2025-06-10T10:40:00", "2025-06-10T12:00:00"),   # the report's worked example
    ("2025-06-10T12:00:00", "2025-06-10T12:00:00"),   # starting exactly on a reading
    ("2025-06-10T22:15:00", "2025-06-11T00:00:00"),   # rolls over midnight
])
def test_prediction_slot(start, slot):
    assert prediction_slot(datetime.fromisoformat(start)) == datetime.fromisoformat(slot)


def test_missing_last_week_is_allowed(predictor, cases):
    inputs = dict(cases[0]["inputs"], nb_last_week=None)
    preds = predictor.predict_rows([predictor.build_row(**inputs)])
    assert all(math.isfinite(float(preds[t][0])) for t in predictor.targets)
