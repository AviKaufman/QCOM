from .progress import ProgressManager
import numpy as np
from scipy.sparse import csr_matrix, kron, identity


"""
Halitonain constructor for QCOM Project. Allows users to build specific hamiltonians. Over time I hope to add more.
"""


def build_rydberg_hamiltonian_chain(
    num_atoms, Omega, Delta, a, pbc=False, show_progress=False
):
    """
    Constructs the Hamiltonian for the Rydberg model on a single-chain configuration.

    Args:
        num_atoms (int): Number of atoms in the system.
        Omega (float): Rabi frequency (driving term with sigma_x), in MHz.
        Delta (float): Detuning (shifts the energy of the Rydberg state relative to the ground state), in MHz.
        a (float): Lattice spacing in μm.
        pbc (bool): Whether to use periodic boundary conditions (PBC).
        show_progress (bool): Whether to display progress updates (default: False).

    Returns:
        hamiltonian (scipy.sparse.csr_matrix): The Hamiltonian matrix.
    """

    C6 = 5420503  # Hard‑coded Van der Waals interaction constant

    sigma_x = csr_matrix([[0, 1], [1, 0]])  # Pauli‑X
    sigma_z = csr_matrix([[1, 0], [0, -1]])  # Pauli‑Z
    identity_2 = identity(2, format="csr")

    hamiltonian = csr_matrix((2**num_atoms, 2**num_atoms))

    # Now only count the choose‑2 interactions (since PBC just changes distance, not pair count)
    total_steps = num_atoms + num_atoms + (num_atoms * (num_atoms - 1)) // 2
    step = 0

    with (
        ProgressManager.progress("Building Rydberg Hamiltonian (Chain)", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # 1) driving term
        for k in range(num_atoms):
            op_x = identity(1, format="csr")
            for j in range(num_atoms):
                op_x = kron(op_x, sigma_x if j == k else identity_2, format="csr")
            hamiltonian += (Omega / 2) * op_x

            step += 1
            if show_progress:
                ProgressManager.update_progress(min(step, total_steps))

        # 2) detuning term
        for k in range(num_atoms):
            op_detune = identity(1, format="csr")
            for j in range(num_atoms):
                op_detune = kron(
                    op_detune,
                    (identity_2 - sigma_z) / 2 if j == k else identity_2,
                    format="csr",
                )
            hamiltonian -= Delta * op_detune

            step += 1
            if show_progress:
                ProgressManager.update_progress(min(step, total_steps))

        # helper to build the two‑site operator
        def construct_interaction(i, j, distance):
            V_ij = C6 / (distance**6)
            op_ni = identity(1, format="csr")
            op_nj = identity(1, format="csr")
            for m in range(num_atoms):
                op_ni = kron(
                    op_ni,
                    (identity_2 - sigma_z) / 2 if m == i else identity_2,
                    format="csr",
                )
                op_nj = kron(
                    op_nj,
                    (identity_2 - sigma_z) / 2 if m == j else identity_2,
                    format="csr",
                )
            return V_ij * op_ni * op_nj

        # 3) van der Waals interactions (with optional PBC distance)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                delta = abs(j - i)
                if pbc:
                    # wrap‑around distance
                    delta_pbc = abs(num_atoms - delta)
                    distance_pbc = delta_pbc * a
                    hamiltonian += construct_interaction(i, j, distance_pbc)
                distance = delta * a

                hamiltonian += construct_interaction(i, j, distance)

                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        # finish up progress
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return hamiltonian


def build_rydberg_hamiltonian_ladder(
    num_atoms, Omega, Delta, a, rho=2, pbc=False, show_progress=False
):
    """
    Same API as before, but much faster by precomputing all local ops
    and all distances exactly once.
    """
    assert num_atoms % 2 == 0, "Need even num_atoms for ladder."

    # constants
    C6 = 5420503
    # single-site building blocks
    σx = csr_matrix([[0, 1], [1, 0]])
    σz = csr_matrix([[1, 0], [0, -1]])
    I2 = identity(2, format="csr")
    nm = (I2 - σz) * 0.5  # projector onto |r>

    Ncol = num_atoms // 2

    # 1) precompute all single-site X_k and n_k operators
    X_ops = []
    n_ops = []
    for k in range(num_atoms):
        # build list of length num_atoms, only slot k is non-identity
        ops_x = [I2] * num_atoms
        ops_n = [I2] * num_atoms
        ops_x[k] = σx
        ops_n[k] = nm

        # kron them all
        A = ops_x[0]
        B = ops_n[0]
        for m in range(1, num_atoms):
            A = kron(A, ops_x[m], format="csr")
            B = kron(B, ops_n[m], format="csr")
        X_ops.append(A)
        n_ops.append(B)

    # 2) precompute all pairwise distances (and their interaction strengths)
    pairs = []
    for i in range(num_atoms):
        ci, ri = divmod(i, 2)
        for j in range(i + 1, num_atoms):
            cj, rj = divmod(j, 2)
            dx = abs(ci - cj)
            dy = abs(ri - rj)

            # direct
            if dy == 0 and dx > 0:
                d1 = dx * a
            elif dx == 0 and dy == 1:
                d1 = rho * a
            else:
                d1 = np.hypot(dx * a, rho * a)
            V1 = C6 / (d1**6)
            pairs.append((i, j, V1))

            # PBC wrap in x direction
            if pbc and dx > 0:
                dx_wrap = abs(Ncol - dx)
                if dy == 0:
                    d2 = dx_wrap * a
                else:
                    d2 = np.hypot(dx_wrap * a, rho * a)
                V2 = C6 / (d2**6)
                pairs.append((i, j, V2))

    # 3) build H by summing the three parts
    H = csr_matrix((2**num_atoms, 2**num_atoms))
    #   a) driving
    for Xk in X_ops:
        H += (Omega / 2) * Xk
    #   b) detuning
    for nk in n_ops:
        H -= Delta * nk
    #   c) interactions
    for i, j, Vij in pairs:
        H += Vij * (n_ops[i].multiply(n_ops[j]))  # element-wise is fine

    return H


def build_ising_hamiltonian(num_spins, J, h, pbc=False, show_progress=False):
    """
    Constructs the Hamiltonian for the 1D Quantum Ising Model in a transverse field.

    Args:
        num_spins (int): Number of spins (sites) in the chain.
        J (float): Coupling strength between neighboring spins (interaction term).
        h (float): Strength of the transverse magnetic field (field term).
        pbc (bool): Whether to use periodic boundary conditions (PBC).
        show_progress (bool): Whether to display progress updates (default: False).

    Returns:
        hamiltonian (scipy.sparse.csr_matrix): The Hamiltonian matrix in sparse form.
    """

    sigma_x = csr_matrix([[0, 1], [1, 0]])  # Pauli-X
    sigma_z = csr_matrix([[1, 0], [0, -1]])  # Pauli-Z
    identity_2 = identity(2, format="csr")

    hamiltonian = csr_matrix((2**num_spins, 2**num_spins))

    total_steps = (2 * num_spins - 1) + (1 if pbc else 0)
    step = 0

    with (
        ProgressManager.progress("Building Ising Hamiltonian (1D Chain)", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for i in range(num_spins):
            op_z = identity(1, format="csr")
            for j in range(num_spins):
                op_z = kron(op_z, sigma_z if j == i else identity_2, format="csr")
            hamiltonian += -h * op_z
            step += 1
            if show_progress:
                ProgressManager.update_progress(min(step, total_steps))

        for i in range(num_spins - 1):
            op_xx = identity(1, format="csr")
            for j in range(num_spins):
                op_xx = kron(
                    op_xx, sigma_x if j in [i, i + 1] else identity_2, format="csr"
                )
            hamiltonian += -J * op_xx
            step += 1
            if show_progress:
                ProgressManager.update_progress(min(step, total_steps))

        if pbc:
            op_x_pbc = identity(1, format="csr")
            for j in range(num_spins):
                op_x_pbc = kron(
                    op_x_pbc,
                    sigma_x if j in [0, num_spins - 1] else identity_2,
                    format="csr",
                )
            hamiltonian += -J * op_x_pbc
            step += 1
            if show_progress:
                ProgressManager.update_progress(min(step, total_steps))

        if show_progress:
            ProgressManager.update_progress(total_steps)

    return hamiltonian


def build_ising_hamiltonian_ladder(
    num_spins, J, h, pbc=False, include_diagonal=True, show_progress=False
):
    """
    Constructs the Hamiltonian for the 1D Quantum Ising Model on a ladder geometry
    with horizontal, vertical, and optional diagonal interactions.

    Args:
        num_spins (int): Number of spins in the system (must be even for the ladder).
        J (float): Coupling strength between neighboring spins (interaction term).
        h (float): Strength of the transverse magnetic field (field term).
        pbc (bool): Whether to use periodic boundary conditions (PBC).
        include_diagonal (bool): Whether to include diagonal interactions.
        show_progress (bool): Whether to display progress updates.

    Returns:
        hamiltonian (scipy.sparse.csr_matrix): The Hamiltonian matrix in sparse form.
    """

    assert num_spins % 2 == 0, "Number of spins must be even for a ladder."

    sigma_x = csr_matrix([[0, 1], [1, 0]])
    sigma_z = csr_matrix([[1, 0], [0, -1]])
    identity_2 = identity(2, format="csr")

    # Precompute number of columns in the ladder
    ncol = num_spins // 2

    # Count how many interaction terms we’ll add
    num_interactions = 0
    for i in range(num_spins):
        col_i, row_i = divmod(i, 2)
        for j in range(i + 1, num_spins):
            col_j, row_j = divmod(j, 2)
            raw = abs(col_i - col_j)
            # wrap horizontally if PBC
            col_diff = min(raw, ncol - raw) if pbc else raw
            row_diff = abs(row_i - row_j)

            # horizontal, vertical, or (opt) diagonal?
            if (
                (row_i == row_j and col_diff == 1)  # horizontal
                or (col_diff == 0 and row_diff == 1)  # vertical
                or (include_diagonal and row_diff == 1 and col_diff == 1)
            ):
                num_interactions += 1

    total_steps = num_spins + num_interactions
    step = 0

    hamiltonian = csr_matrix((2**num_spins, 2**num_spins))

    with (
        ProgressManager.progress("Building Ising Hamiltonian (Ladder)", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # 1) Transverse‐field on Z
        for i in range(num_spins):
            op_z = identity(1, format="csr")
            for j in range(num_spins):
                op_z = kron(op_z, sigma_z if j == i else identity_2, format="csr")
            hamiltonian += -h * op_z

            step += 1
            if show_progress:
                ProgressManager.update_progress(step)

        # helper for σ_x ⊗ σ_x
        def construct_xx(i, j):
            op = identity(1, format="csr")
            for k in range(num_spins):
                op = kron(op, sigma_x if k in (i, j) else identity_2, format="csr")
            return -J * op

        # 2) Couplings (horizontal, vertical, optional diagonal, with optional wrap)
        for i in range(num_spins):
            col_i, row_i = divmod(i, 2)
            for j in range(i + 1, num_spins):
                col_j, row_j = divmod(j, 2)
                raw = abs(col_i - col_j)
                col_diff = min(raw, ncol - raw) if pbc else raw
                row_diff = abs(row_i - row_j)

                if (
                    (row_i == row_j and col_diff == 1)
                    or (col_diff == 0 and row_diff == 1)
                    or (include_diagonal and row_diff == 1 and col_diff == 1)
                ):
                    hamiltonian += construct_xx(i, j)
                    step += 1
                    if show_progress:
                        ProgressManager.update_progress(step)

        # finalize progress
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return hamiltonian
