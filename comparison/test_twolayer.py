#!/usr/bin/env python3
"""
Tests for the TwoLayer neural network.

Author: Sebastian Goldt <goldt.sebastian@gmail.com>

February 2020
"""

import math
import unittest

import torch
import torch.nn.functional as F

from mlp.twolayer import TwoLayer, erfscaled, identity


class TwoLayerTests(unittest.TestCase):
    def test_output_standard(self):
        N = 784
        M = 3
        num_cats = 10

        student = TwoLayer(F.relu, N, M, num_cats)

        w = student.fc1.weight.data
        v = student.fc2.weight.data

        xs = torch.randn(2, 784)

        ys = torch.t(F.relu(v.mm(F.relu(w.mm(xs.T)))))
        preds = student(xs)

        self.assertTrue(torch.allclose(ys, preds))

    def test_output_committee(self):
        N = 784
        M = 3
        num_cats = 10

        gs = (F.relu, identity)
        student = TwoLayer(gs, N, M, num_cats)

        w = student.fc1.weight.data
        v = student.fc2.weight.data

        xs = torch.randn(2, 784)

        ys = torch.t(v @ F.relu(w @ xs.T))
        preds = student(xs)

        self.assertTrue(torch.allclose(ys, preds))

    def test_output_committee_erf(self):
        N = 784
        M = 3
        num_cats = 10
        gs = (erfscaled, identity)

        student = TwoLayer(gs, N, M, num_cats)

        w = student.fc1.weight.data
        v = student.fc2.weight.data

        xs = torch.randn(2, 784)

        ys = torch.t(v @ gs[0](w @ xs.T))
        preds = student(xs)

        self.assertTrue(torch.allclose(ys, preds))

    def test_output_committee_normalise1(self):
        N = 784
        M = 3
        num_cats = 10
        g = (F.relu, identity)

        student = TwoLayer(g, N, M, num_cats, normalise1=True)

        w = student.fc1.weight.data
        v = student.fc2.weight.data

        xs = torch.randn(2, 784)

        ys = torch.t(v @ F.relu(w @ xs.T / math.sqrt(N)))
        preds = student(xs)

        self.assertTrue(torch.allclose(ys, preds))

    def test_output_committee_normalise2(self):
        N = 784
        M = 3
        num_cats = 10
        gs = (F.relu, identity)

        student = TwoLayer(gs, N, M, num_cats, normalise2=True)

        w = student.fc1.weight.data
        v = student.fc2.weight.data

        xs = torch.randn(2, 784)

        ys = torch.t(v @ F.relu(w @ xs.T) / math.sqrt(M))
        preds = student(xs)

        self.assertTrue(torch.allclose(ys, preds))

    def test_output_committee_normaliseBoth(self):
        N = 784
        M = 3
        num_cats = 10
        gs = (F.relu, identity)

        student = TwoLayer(gs, N, M, num_cats, normalise1=True, normalise2=True)

        w = student.fc1.weight.data
        v = student.fc2.weight.data

        xs = torch.randn(2, 784)

        ys = torch.t(v @ F.relu(w @ xs.T / math.sqrt(N)) / math.sqrt(M))
        preds = student(xs)

        self.assertTrue(torch.allclose(ys, preds))

    def test_selfoverlap(self):
        N = 784
        M = 3
        num_cats = 10

        student = TwoLayer(F.relu, N, M, num_cats)

        w = student.fc1.weight.data
        Q = w @ w.T / N

        self.assertTrue(torch.allclose(Q, student.selfoverlap()))


if __name__ == "__main__":
    unittest.main()
