import numpy as np

from src.dca_bootstrap import (
    net_benefit,
    treat_all_net_benefit,
)


def test_treat_none_reference_logic():

    y_true = np.array([
        0, 0, 1, 1
    ])

    y_prob = np.array([
        0.1,
        0.2,
        0.8,
        0.9,
    ])

    nb = net_benefit(
        y_true,
        y_prob,
        threshold=0.5,
    )

    assert nb > 0


def test_treat_all_formula():

    y_true = np.array([
        0, 0, 1, 1
    ])

    nb = treat_all_net_benefit(
        y_true,
        threshold=0.5,
    )

    assert abs(nb) < 1e-8
