import numpy as np

from qcom.controls import TimeSeries
from qcom.solvers import evolve_state


class TwoLevelAdapter:
    required_channels = ("Omega",)
    dimension = 2

    def hamiltonian_at(self, t, controls):
        omega = controls["Omega"]
        return np.array([[0.0, omega / 2.0], [omega / 2.0, 0.0]])


time_series = TimeSeries(Omega=([0.0, 1.0], [1.0, 1.0]))
psi0 = np.array([1.0, 0.0], dtype=np.complex128)
psi_t, _ = evolve_state(time_series, TwoLevelAdapter(), psi0, n_steps=4, show_progress=False)
print(np.round(np.abs(psi_t) ** 2, 6))
