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

    C6 = 5420503  # Hard-coded Van der Waals interaction constant

    sigma_x = csr_matrix([[0, 1], [1, 0]])  # Pauli-X
    sigma_z = csr_matrix([[1, 0], [0, -1]])  # Pauli-Z
    identity_2 = identity(2, format="csr")

    hamiltonian = csr_matrix((2**num_atoms, 2**num_atoms))

    total_steps = (
        num_atoms
        + num_atoms
        + (num_atoms * (num_atoms - 1)) // 2
        + (num_atoms if pbc else 0)
    )
    step = 0

    with (
        ProgressManager.progress("Building Rydberg Hamiltonian (Chain)", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for k in range(num_atoms):
            op_x = identity(1, format="csr")
            for j in range(num_atoms):
                op_x = kron(op_x, sigma_x if j == k else identity_2, format="csr")
            hamiltonian += (Omega / 2) * op_x
            step += 1
            if show_progress:
                ProgressManager.update_progress(min(step, total_steps))

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

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                distance = abs(j - i) * a
                hamiltonian += construct_interaction(i, j, distance)
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        if pbc:
            for i in range(num_atoms):
                j = (i + 1) % num_atoms
                distance = a
                hamiltonian += construct_interaction(i, j, distance)
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        if show_progress:
            ProgressManager.update_progress(total_steps)

    return hamiltonian


def build_rydberg_hamiltonian_ladder(
    num_atoms, Omega, Delta, a, rho=2, pbc=False, show_progress=False
):
    """
    Constructs the Hamiltonian for the Rydberg model on a ladder configuration with horizontal,
    vertical, and diagonal interactions between atoms.

    Args:
        num_atoms (int): Number of atoms in the system (must be even for the ladder).
        Omega (float): Rabi frequency (driving term with sigma_x), in MHz.
        Delta (float): Detuning (shifts the energy of the Rydberg state relative to the ground state), in MHz.
        a (float): Lattice spacing in μm (x-spacing).
        rho (float): Ratio of y-spacing to x-spacing (default is 2, meaning y-spacing = 2 * a).
        pbc (bool): Whether to use periodic boundary conditions (PBC).
        show_progress (bool): Whether to display progress updates (default: False).

    Returns:
        hamiltonian (scipy.sparse.csr_matrix): The Hamiltonian matrix.
    """

    assert (
        num_atoms % 2 == 0
    ), "Number of atoms must be even for a ladder configuration."

    C6 = 5420503  # Hard-coded Van der Waals interaction constant

    sigma_x = csr_matrix([[0, 1], [1, 0]])  # Pauli-X
    sigma_z = csr_matrix([[1, 0], [0, -1]])  # Pauli-Z
    identity_2 = identity(2, format="csr")

    hamiltonian = csr_matrix((2**num_atoms, 2**num_atoms))

    total_steps = 2 * num_atoms + (num_atoms * (num_atoms - 1)) // 2 + (2 if pbc else 0)
    step = 0

    with (
        ProgressManager.progress("Building Rydberg Hamiltonian (Ladder)", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for k in range(num_atoms):
            op_x = identity(1, format="csr")
            for j in range(num_atoms):
                op_x = kron(op_x, sigma_x if j == k else identity_2, format="csr")
            hamiltonian += (Omega / 2) * op_x
            step += 1
            if show_progress:
                ProgressManager.update_progress(min(step, total_steps))

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

        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                column_i, row_i = i // 2, i % 2
                column_j, row_j = j // 2, j % 2

                if row_i == row_j:
                    distance = abs(column_i - column_j) * a
                elif column_i == column_j:
                    distance = rho * a
                else:
                    horizontal_distance = abs(column_i - column_j) * a
                    vertical_distance = rho * a
                    distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)

                hamiltonian += construct_interaction(i, j, distance)
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        if pbc:
            for row_start in [0, 1]:
                i = row_start
                j = row_start + 2 * (num_atoms // 2 - 1)
                distance = a
                hamiltonian += construct_interaction(i, j, distance)
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

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
        include_diagonal (bool): Whether to include diagonal interactions (default: True).
        show_progress (bool): Whether to display progress updates (default: False).

    Returns:
        hamiltonian (scipy.sparse.csr_matrix): The Hamiltonian matrix in sparse form.
    """

    assert (
        num_spins % 2 == 0
    ), "Number of spins must be even for a ladder configuration."

    sigma_x = csr_matrix([[0, 1], [1, 0]])  # Pauli-X
    sigma_z = csr_matrix([[1, 0], [0, -1]])  # Pauli-Z
    identity_2 = identity(2, format="csr")

    hamiltonian = csr_matrix((2**num_spins, 2**num_spins))

    num_interactions = 0
    for i in range(num_spins):
        for j in range(i + 1, num_spins):
            column_i, row_i = i // 2, i % 2
            column_j, row_j = j // 2, j % 2
            if row_i == row_j and abs(column_i - column_j) == 1:
                num_interactions += 1
            elif column_i == column_j and row_i != row_j:
                num_interactions += 1
            elif (
                include_diagonal
                and abs(column_i - column_j) == 1
                and abs(row_i - row_j) == 1
            ):
                num_interactions += 1

    total_steps = num_spins + num_interactions + (2 if pbc else 0)
    step = 0

    with (
        ProgressManager.progress("Building Ising Hamiltonian (Ladder)", total_steps)
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

        def construct_interaction(i, j):
            op_xx = identity(1, format="csr")
            for m in range(num_spins):
                op_xx = kron(
                    op_xx, sigma_x if m in [i, j] else identity_2, format="csr"
                )
            return -J * op_xx

        for i in range(num_spins):
            for j in range(i + 1, num_spins):
                column_i, row_i = i // 2, i % 2
                column_j, row_j = j // 2, j % 2

                if row_i == row_j and abs(column_i - column_j) == 1:
                    hamiltonian += construct_interaction(i, j)
                elif column_i == column_j and row_i != row_j:
                    hamiltonian += construct_interaction(i, j)
                elif (
                    include_diagonal
                    and abs(column_i - column_j) == 1
                    and abs(row_i - row_j) == 1
                ):
                    hamiltonian += construct_interaction(i, j)

                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        if pbc:
            for row_start in [0, 1]:
                i = row_start
                j = row_start + 2 * (num_spins // 2 - 1)
                hamiltonian += construct_interaction(i, j)
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        if show_progress:
            ProgressManager.update_progress(total_steps)

    return hamiltonian
