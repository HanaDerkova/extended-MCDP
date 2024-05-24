import datetime
import itertools

import numpy as np
import pytest
import scipy.stats
from scipy.special import logsumexp

# import simple_model
import src.simple_model
from helpers import load_trained_model


class TestEvalProbsSingleDirect:

    funs = {
        "simulation": src.simple_model.Model.eval_probs_single_simulation,
        "hanka": src.simple_model.Model.eval_probs_hankas_method,
        "direct": src.simple_model.Model.eval_probs_single_direct_lm,
    }

    testdata = [
        # ([(1, 2)], [(1, 2)], 3),
        # ([(1, 2)], [(2, 3)], 3),
        # ([(1, 2)], [(2, 3)], 10),
        # ([(1, 2)], [(2, 3), (5, 9)], 10),
        # ([(1, 2), (6, 7)], [(2, 3), (5, 9)], 20),
        # ([(1, 2), (6, 7), (10, 11), (15, 16), (17, 18)], [(2, 3), (5, 9), (19, 20)], 20),
        ([(1, 99)], [(1, 99)], 100),
        ([(1, 999)], [(1, 999)], 1_000),
        ([(1, 9999)], [(1, 9999)], 10_000),
        ([(1, 990)], [(1, 990)], 1_000_000),
        #----------------------------------------- matrix multipl
        ([(1, 99999)], [(1, 99999)], 100_000),
        ([(1, 999999)], [(1, 999999)], 1_000_000),
        ([(1, 9999999)], [(1, 9999999)], 10_000_000),
        #------------------------------------------------------ orig mcdp
        ([(1, 99999999)], [(1, 99999999)], 100_000_000),
        ([(1, 999999999)], [(1, 999999999)], 1_000_000_000),
    ]

    @pytest.mark.parametrize("r,q,c", testdata)
    def test_sum_is_one(self, r, q, c):
        gap_matrix = np.array([[0.3, 0.3,0.4], [0,0.2,0.8],[0,0,0]])
        interval_matrix = np.array([[0.5, 0.5,0], [0,0.8,0.2],[0,0,0]])
        fun = self.funs['hanka']
        chr_size = c
        result = fun(r, q, chr_size, gap_matrix, interval_matrix)
        assert logsumexp(result) == pytest.approx(0)

    @pytest.mark.parametrize("r,q,chr_size", testdata)
    @pytest.mark.flaky(reruns=5)
    def test_close_to_simulation(self, r, q, chr_size):
        gap_matrix = np.array([[0.3, 0.3,0.4], [0,0.2,0.8],[0,0,0]])
        interval_matrix = np.array([[0.5, 0.5,0], [0,0.8,0.2],[0,0,0]])
        result_sim = self.funs["simulation"](r, q, chr_size, tries=100)
        result_direct = self.funs["hanka"](r, q, chr_size, gap_matrix, interval_matrix)
        assert np.exp(result_direct) == pytest.approx(np.exp(result_sim), abs=0.05)

# class TestNewEvalProbsSingleDirect:

#     funs = {
#         "simulation": src.simple_model.Model.eval_probs_single_simulation,
#         "hanka": src.simple_model.Model.eval_probs_hankas_method,
#         "direct": src.simple_model.Model.eval_probs_single_direct_lm,
#     }

#     testdata = [
#         ([(1, 9999)], [(1, 9999)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(10)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(200)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(300)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(400)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(500)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(600)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(700)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(800)], 100_000_000),
        
#     ]

#     def test_me():
#         assert -1.7603885375187377e-08 == pytest.approx(0, abs=1e-7)

#     @pytest.mark.parametrize("r,q,c", testdata)
#     def test_sum_is_one(self, r, q, c):
#         gap_matrix = np.array([[0.3, 0.3,0.4], [0,0.2,0.8],[0,0,0]])
#         interval_matrix = np.array([[0.5, 0.5,0], [0,0.8,0.2],[0,0,0]])
#         fun = self.funs['hanka']
#         chr_size = c
#         result = fun(r, q, chr_size, gap_matrix, interval_matrix)
#         assert logsumexp(result) == pytest.approx(0,rel=1e-8, abs=1e-8)

# class TestNewEvalProbsSingleDirect:

#     funs = {
#         "simulation": src.simple_model.Model.eval_probs_single_simulation,
#         "hanka": src.simple_model.Model.eval_probs_hankas_method,
#         "direct": src.simple_model.Model.eval_probs_single_direct_lm,
#     }

#     testdata = [
#         ([(1, 9999)], [(1, 9999)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(10)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(200)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(300)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(400)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(500)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(600)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(700)], 100_000_000),
#         ([(i * 9999 + 1, (i + 1) * 9999) for i in range(100)], [(i * 9999 + 1, (i + 1) * 9999) for i in range(800)], 100_000_000),
        
#     ]

#     def test_me():
#         assert -1.7603885375187377e-08 == pytest.approx(0, abs=1e-7)

#     @pytest.mark.parametrize("r,q,c", testdata)
#     def test_sum_is_one(self, r, q, c):
#         gap_matrix = load_trained_model("../../resources/example_github/trained_model_gaps.txt")
#         interval_matrix = load_trained_model("../../resources/example_github/trained_model_intervals.txt")
#         fun = self.funs['hanka']
#         chr_size = c
#         result = fun(r, q, chr_size, gap_matrix, interval_matrix)
#         assert logsumexp(result) == pytest.approx(0,rel=1e-8, abs=1e-8)

