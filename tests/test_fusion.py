import numpy as np

from src.fusion_engine import FusionEngine


def test_fuse_scores_with_defaults_and_bounds():
	fusion = FusionEngine()
	score = fusion.fuse_scores(None, 1.0, 0.0)
	assert 0.0 <= score <= 1.0


def test_calibrate_weights_normalized():
	fusion = FusionEngine()
	fusion.calibrate_weights(0.9, 0.8, 0.7)
	assert np.isclose(fusion.weights.sum(), 1.0)
	assert all(weight > 0 for weight in fusion.weights)


def test_predict_label_threshold():
	fusion = FusionEngine()
	assert fusion.predict_label(0.7, threshold=0.5) == 1
	assert fusion.predict_label(0.2, threshold=0.5) == 0
