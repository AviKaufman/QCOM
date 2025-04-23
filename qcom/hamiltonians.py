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
    Constructs the Hamiltonian for the Rydberg model on a ladder configuration with horizontal,
    vertical, and diagonal interactions between atoms, including both direct and periodic images
    when pbc=True.
    """

    assert (
        num_atoms % 2 == 0
    ), "Number of atoms must be even for a ladder configuration."

    C6 = 5420503
    sigma_x = csr_matrix([[0, 1], [1, 0]])
    sigma_z = csr_matrix([[1, 0], [0, -1]])
    identity_2 = identity(2, format="csr")

    # precompute columns
    ncol = num_atoms // 2

    # count how many interactions we’ll insert:
    total_pairs = num_atoms * (num_atoms - 1) // 2
    verticals = ncol  # one vertical per column
    # every non-vertical pair gets TWO terms if pbc, otherwise 1
    extra_images = (total_pairs - verticals) if pbc else 0
    total_steps = num_atoms + num_atoms + total_pairs + extra_images
    step = 0

    hamiltonian = csr_matrix((2**num_atoms, 2**num_atoms))

    with (
        ProgressManager.progress("Building Rydberg Hamiltonian (Ladder)", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        # 1) driving
        for k in range(num_atoms):
            op = identity(1, format="csr")
            for j in range(num_atoms):
                op = kron(op, sigma_x if j == k else identity_2, format="csr")
            hamiltonian += (Omega / 2) * op
            step += 1
            if show_progress:
                ProgressManager.update_progress(step)

        # 2) detuning
        for k in range(num_atoms):
            op = identity(1, format="csr")
            for j in range(num_atoms):
                op = kron(
                    op,
                    (identity_2 - sigma_z) / 2 if j == k else identity_2,
                    format="csr",
                )
            hamiltonian -= Delta * op
            step += 1
            if show_progress:
                ProgressManager.update_progress(step)

        # helper
        def construct_interaction(i, j, dist):
            V = C6 / (dist**6)
            op_i = identity(1, format="csr")
            op_j = identity(1, format="csr")
            for m in range(num_atoms):
                nm = (identity_2 - sigma_z) / 2
                op_i = kron(op_i, nm if m == i else identity_2, format="csr")
                op_j = kron(op_j, nm if m == j else identity_2, format="csr")
            return V * op_i * op_j

        # 3) all‐to‐all interactions + periodic images
        for i in range(num_atoms):
            col_i, row_i = divmod(i, 2)
            for j in range(i + 1, num_atoms):
                col_j, row_j = divmod(j, 2)

                dx_raw = abs(col_i - col_j)
                dy = abs(row_i - row_j)

                # direct distance
                if dy == 0 and dx_raw > 0:
                    d1 = dx_raw * a  # horizontal
                elif dx_raw == 0 and dy == 1:
                    d1 = rho * a  # vertical
                else:
                    d1 = np.sqrt((dx_raw * a) ** 2 + (rho * a) ** 2)  # diagonal

                hamiltonian += construct_interaction(i, j, d1)
                step += 1
                if show_progress:
                    ProgressManager.update_progress(step)

                # periodic‐image distance (only wrap in x-direction)
                if pbc and dx_raw > 0:
                    dx_wrap = abs(ncol - dx_raw)
                    # same pattern: horizontal vs diag
                    if dy == 0:
                        d2 = dx_wrap * a
                    elif dx_raw == 0:
                        # vertical has no wrap
                        continue
                    else:
                        d2 = np.sqrt((dx_wrap * a) ** 2 + (rho * a) ** 2)

                    hamiltonian += construct_interaction(i, j, d2)
                    step += 1
                    if show_progress:
                        ProgressManager.update_progress(step)

        # finish up
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return hamiltonian


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
