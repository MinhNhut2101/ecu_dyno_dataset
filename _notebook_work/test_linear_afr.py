"""Synthetic invariants for the separate interpolation/extrapolation layers."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from afr_interpolation import LinearAFRSurface, complete_matrix


def affine(points):
    points = np.asarray(points, dtype=float)
    return 14.0 + 0.3 * points[:, 0] - 0.2 * points[:, 1]


class LinearAFRChecks(unittest.TestCase):
    def test_affine_surface_exact_inside_and_outside(self):
        source = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=float)
        values = affine(source)
        query = np.array([[1, 1], [0.5, 0.75], [4, 3], [-2, 1]], dtype=float)
        model = LinearAFRSurface(source, values)
        strict, full, audit = model.predict(query)
        np.testing.assert_allclose(full, affine(query), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(strict[:2], affine(query[:2]), rtol=1e-12, atol=1e-12)
        self.assertTrue(np.isnan(strict[2:]).all())
        self.assertEqual(audit.origin.tolist(), [
            'linear_interpolation', 'linear_interpolation',
            'linear_extrapolation', 'linear_extrapolation',
        ])
        self.assertEqual(audit.inside_source_hull.tolist(), [True, True, False, False])

    def test_supported_values_preserved_exactly_and_input_unchanged(self):
        source = pd.DataFrame(np.nan, index=[0, 1, 2, 3], columns=[0, 1, 2, 3])
        for row, col, value in [(0, 0, 14.123456789), (0, 2, 14.728374657),
                                (2, 0, 13.222333444), (2, 2, 14.555666777)]:
            source.loc[row, col] = value
        before = source.copy(deep=True)
        result = complete_matrix(source)
        pd.testing.assert_frame_equal(source, before)
        pd.testing.assert_frame_equal(result['source'], before)
        known = source.notna().to_numpy()
        np.testing.assert_array_equal(result['full'].to_numpy()[known], source.to_numpy()[known])
        np.testing.assert_array_equal(result['strict'].to_numpy()[known], source.to_numpy()[known])
        self.assertTrue(np.isfinite(result['full'].to_numpy()).all())
        audit = result['audit']
        self.assertEqual(audit.origin.eq('supported').sum(), 4)
        self.assertTrue(audit.loc[audit.origin.ne('supported'), 'source_value'].isna().all())
        self.assertTrue(audit.loc[audit.origin.eq('supported'), 'nearest_anchor_distance_bins'].eq(0).all())
        self.assertTrue(result['strict'].iloc[:3, :3].notna().all().all())
        self.assertTrue(result['strict'].iloc[3, :].isna().all())
        self.assertTrue(result['strict'].iloc[:, 3].isna().all())

    def test_local_fit_expands_beyond_nearest_eight_for_full_rank(self):
        # The first eight neighbors lie on y=0; a ninth provides the second dimension.
        points = np.vstack([np.c_[np.arange(8), np.zeros(8)], [[0, 10]]])
        model = LinearAFRSurface(points, affine(points), neighbors=8)
        query = np.array([[3, -1]], dtype=float)
        strict, full, audit = model.predict(query)
        self.assertTrue(np.isnan(strict).all())
        np.testing.assert_allclose(full, affine(query), rtol=1e-12, atol=1e-12)
        self.assertEqual(audit.neighbors_used.iloc[0], 9)
        self.assertEqual(audit.fit_rank.iloc[0], 3)
        self.assertLess(audit.fit_condition.iloc[0], 1e8)

    def test_rejects_collinear_all_missing_and_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            LinearAFRSurface([[0, 0], [1, 0], [2, 0]], [14, 14.1, 14.2])
        with self.assertRaises(ValueError):
            complete_matrix(pd.DataFrame(np.nan, index=[0, 1], columns=[0, 1]))
        with self.assertRaises(ValueError):
            LinearAFRSurface([[0, 0], [1, 0], [0, 1]], [14, np.inf, 14.2])
        model = LinearAFRSurface([[0, 0], [1, 0], [0, 1]], [14, 14.1, 14.2])
        with self.assertRaises(ValueError):
            model.predict([[np.nan, 0]])

    def test_generated_values_never_increase_the_source_anchor_count(self):
        source = pd.DataFrame(np.nan, index=range(5), columns=range(5))
        source.loc[1, 1], source.loc[1, 3], source.loc[3, 1], source.loc[3, 3] = 14.0, 14.2, 14.4, 14.6
        result = complete_matrix(source)
        self.assertEqual(result['source'].notna().sum().sum(), 4)
        self.assertEqual(result['audit'].origin.eq('supported').sum(), 4)
        self.assertEqual(result['audit'].source_value.notna().sum(), 4)
        self.assertEqual(len(result['audit']), source.size)
        self.assertTrue(result['full'].notna().all().all())
        unsupported = result['audit'].origin.ne('supported')
        self.assertTrue(result['audit'].loc[unsupported, 'source_value'].isna().all())
        self.assertTrue(result['audit'].loc[unsupported, 'estimated_value'].notna().all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
