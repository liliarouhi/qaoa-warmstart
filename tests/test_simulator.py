"""Validate the NumPy simulator against PennyLane (thesis Ch. 5, 'simulator validation')."""
import numpy as np
import pytest

from qws.graphs import erdos_renyi, from_edge_list
from qws.simulator import QAOASimulator


def test_single_edge_p1_analytic():
    # For one edge: F = 1/2 + 1/2 sin(4 beta) sin(gamma)  (special case of Lemma 3, d_u = d_v = 0)
    sim = QAOASimulator(from_edge_list(2, [(0, 1)]))
    for gm, bt in [(0.3, 0.2), (1.1, -0.4), (2.0, 0.7)]:
        assert np.isclose(sim.expectation([gm], [bt]), 0.5 + 0.5 * np.sin(4 * bt) * np.sin(gm))


def test_zero_angles_give_half_total_weight(rng):
    g = erdos_renyi(6, 0.5, rng)
    sim = QAOASimulator(g)
    assert np.isclose(sim.expectation([0.0], [0.0]), g.weights.sum() / 2)


def test_state_normalised(rng):
    sim = QAOASimulator(erdos_renyi(6, 0.5, rng))
    assert np.isclose(np.sum(sim.probabilities([0.4, 0.1], [0.2, -0.3])), 1.0)


def test_nfev_counter(rng):
    sim = QAOASimulator(erdos_renyi(5, 0.5, rng))
    for _ in range(4):
        sim.expectation([0.1], [0.1])
    assert sim.nfev == 4


def test_matches_pennylane(rng):
    qml = pytest.importorskip("pennylane")
    g = erdos_renyi(6, 0.6, rng)
    n = g.n
    # Cost Hamiltonian as an observable: C = sum w_uv (1 - Z_u Z_v)/2
    coeffs = [-w / 2 for w in g.weights]
    obs = [qml.PauliZ(int(u)) @ qml.PauliZ(int(v)) for u, v in g.edges]
    const = g.weights.sum() / 2
    H = qml.Hamiltonian(coeffs, obs)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(gammas, betas):
        for w in range(n):
            qml.Hadamard(w)
        for gm, bt in zip(gammas, betas):
            for (u, v), wt in zip(g.edges, g.weights):
                # exp(-i gm * wt * (1 - Z Z)/2) = global phase * exp(+i gm*wt/2 ZZ) = IsingZZ(-gm*wt)
                qml.IsingZZ(-gm * wt, wires=[int(u), int(v)])
            for w in range(n):
                qml.RX(2 * bt, wires=w)  # RX(2b) = exp(-i b X)
        return qml.expval(H)

    sim = QAOASimulator(g)
    for _ in range(3):
        gs, bs = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
        assert np.isclose(sim.expectation(gs, bs), float(circuit(gs, bs)) + const, atol=1e-8)
