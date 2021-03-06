#! /usr/bin/env python
import os
import datetime as dt
import unittest

from mock import patch, MagicMock, call, DEFAULT
import numpy as np

from ml_utils.time_series_utils import(
        rolling_window,
        split_sequence,
        match_sequences,
        split_sequences,
        )

class TimeSeriesUtilsTest(unittest.TestCase):
    ''' Class for testing NextsimBin '''

    def test_rolling_window(self):
        """ test rolling_window works for a variety of cases """
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

    def test_split_sequence(self):
        """ test split_sequence works for a variety of cases """
        ns = 10
        a = np.arange(ns)

        x, y, index = split_sequence(a, n_in=1, n_out=1)
        x2 = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])
        y2 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
        i2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(list(x.flatten()), list(x2.flatten()))
        self.assertEqual(list(y.flatten()), list(y2.flatten()))
        self.assertEqual(list(index.flatten()), list(i2.flatten()))
        nr = len(index)
        self.assertEqual(x.shape, (nr,1))
        self.assertEqual(y.shape, (nr,1))

        x, y, index = split_sequence(a, n_in=2, n_out=1)
        x3 = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        y3 = y2[1:]
        i3 = i2[1:]
        self.assertEqual(list(x.flatten()), list(x3.flatten()))
        self.assertEqual(list(y.flatten()), list(y3.flatten()))
        self.assertEqual(list(index.flatten()), list(i3.flatten()))
        i3 = i2[:-1]
        nr = len(index)
        self.assertEqual(x.shape, (nr,2))
        self.assertEqual(y.shape, (nr,1))

        x, y, index = split_sequence(a, n_in=1, n_out=2)
        x3 = x2[:-1]
        y3 = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
        i3 = i2[:-1]
        nr = len(index)
        self.assertEqual(x.shape, (nr,1))
        self.assertEqual(y.shape, (nr,2))

        x, y, index = split_sequence(a, n_in=0, n_out=2)
        y3 = rolling_window(a, 2)
        i3 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        nr = len(index)
        self.assertEqual(x.shape, (nr,0))
        self.assertEqual(y.shape, (nr,2))
        self.assertEqual(list(y.flatten()), list(y3.flatten()))
        self.assertEqual(list(index.flatten()), list(i3.flatten()))

        x, y, index = split_sequence(a, n_in=3, n_out=0)
        x3 = rolling_window(a, 3)
        i3 = np.array([3, 4, 5, 6, 7, 8, 9, 10])
        nr = len(index)
        self.assertEqual(y.shape, (nr,0))
        self.assertEqual(x.shape, (nr,3))
        self.assertEqual(list(x.flatten()), list(x3.flatten()))
        self.assertEqual(list(index.flatten()), list(i3.flatten()))

    def test_match_sequences(self):
        seqs = [np.array([10, 20, 30]), np.array([25, 35, 45, 55])]
        inds = [np.array([0, 1, 2]), np.array([1, 2, 3, 4])]
        s1, s2 = match_sequences(seqs, inds, [1,2])
        self.assertEqual(list(s1), [20, 30])
        self.assertEqual(list(s2), [25, 35])

    def test_split_sequences(self):
        # simple test
        s1 = np.arange(10,100,10)
        s2 = 5 + s1
        xx, yy, index = split_sequences([s1,s2],[3,3], n_steps_out=[1,1])
        xx2 = [
                np.array([
                    [10, 20, 30],
                    [20, 30, 40],
                    [30, 40, 50],
                    [40, 50, 60],
                    [50, 60, 70],
                    [60, 70, 80],
                    ]),
                np.array([
                    [15, 25, 35],
                    [25, 35, 45],
                    [35, 45, 55],
                    [45, 55, 65],
                    [55, 65, 75],
                    [65, 75, 85],
                    ]),
                ]
        yy2 =[
                np.array([[40], [50], [60], [70], [80], [90]]),
                np.array([[45], [55], [65], [75], [85], [95]]),
                ]
        i2 = np.array([3, 4, 5, 6, 7, 8])
        for a,a2 in zip(xx, xx2):
            self.assertEqual(list(a.flatten()), list(a2.flatten()))
        for a,a2 in zip(yy, yy2):
            self.assertEqual(list(a.flatten()), list(a2.flatten()))
        self.assertEqual(list(index.flatten()), list(i2.flatten()))

        # simple test
        yy2 = [
                np.array([[40], [50], [60], [70], [80]]),
                np.array([[45, 55], [55, 65], [65, 75], [75, 85], [85, 95]]),
                ]
        i2 = np.array([3, 4, 5, 6, 7])
        xx, yy, index = split_sequences([s1,s2],[3,3], n_steps_out=[1,2])
        for a,a2 in zip(xx, xx2):
            self.assertEqual(list(a.flatten()), list(a2[:-1].flatten()))
        for a,a2 in zip(yy, yy2):
            self.assertEqual(list(a.flatten()), list(a2.flatten()))
        self.assertEqual(list(index.flatten()), list(i2.flatten()))

        # some of n_steps_in and n_steps_out are 0
        (x,), (y,), index = split_sequences([s1,s2],[3,0],[0,2])
        x2 = np.array([
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50],
            [40, 50, 60],
            [50, 60, 70],
            ])
        y2 = np.array([
            [45, 55],
            [55, 65],
            [65, 75],
            [75, 85],
            [85, 95],
            ])
        i2 = np.array([3, 4, 5, 6, 7])
        self.assertEqual(list(x.flatten()), list(x2.flatten()))
        self.assertEqual(list(y.flatten()), list(y2.flatten()))
        self.assertEqual(list(index.flatten()), list(i2.flatten()))


if __name__ == "__main__":
    unittest.main()
