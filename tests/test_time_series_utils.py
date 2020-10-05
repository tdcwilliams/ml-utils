#! /usr/bin/env python
import os
import datetime as dt
import unittest

from mock import patch, MagicMock, call, DEFAULT
import numpy as np

from ml_utils.time_series_utils import(
        rolling_window,
        )

class TimeSeriesUtilsTest(unittest.TestCase):
    ''' Class for testing NextsimBin '''

    def test_rolling_window(self):
        ns = 10
        a = np.arange(ns)

        # window size = 0
        r = rolling_window(a,0)
        self.assertEqual(r.shape, (ns+1,0))

        # window size = 1
        r = rolling_window(a,1)
        self.assertEqual(r.shape, (ns,1))
        self.assertEqual(list(r.flatten()), list(a.flatten()))

        # window size = 2
        r = rolling_window(a,2)
        self.assertEqual(r.shape, (ns-1,2))
        r2 = np.array([a[:-1], a[1:]]).T
        self.assertEqual(list(r.flatten()), list(r2.flatten()))

if __name__ == "__main__":
    unittest.main()
