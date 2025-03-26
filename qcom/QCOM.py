"""
QCOM.py
Author: Avi Kaufman

This file is the main file for the QCOM project. It will contain all the imports and the main function for the project.

QCOM is short for Quantum Computation and is a package developed as part of Avi Kaufman's 2025 honor's thesis for undergraduate physics

QCOM is a package that is intended to aid the Meurice Research group in analysis of quantum systoms. Particularly thermodynamic properties of neutral atom (rydberg) systems. The package allows for users to compute exact values, and work with external values from DMRG or Quantum Devices such as Aquilla.

This package is a work in progress and should be updated to fit the needs of the group or any other users.

Last updated: 3-24-2025
"""

"""
All Imports for QCOM Project
"""

import random
import os
import pandas as pd
import sys
import time
from contextlib import contextmanager
import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

"""
Progress Manager for QCOM Project. Helps to track progress of long running tasks.
"""


class ProgressManager:
    """Manages progress updates for long-running tasks with real-time feedback."""

    active_task = None  # Keeps track of the currently active task
    total_steps = None  # Total number of steps for the current task
    start_time = None  # Start time of the current task

    @staticmethod
    @contextmanager
    def progress(task_name, total_steps=None):
        if ProgressManager.active_task is not None:
            # If there's already an active task, skip nested progress tracking
            yield
            return

        # Initialize progress tracking
        ProgressManager.active_task = task_name
        ProgressManager.total_steps = total_steps
        ProgressManager.start_time = time.time()
        sys.stdout.write(f"Starting: {task_name}...\n")
        sys.stdout.flush()

        try:
            yield
        finally:
            # Compute total elapsed time
            elapsed_time = time.time() - ProgressManager.start_time
            sys.stdout.write(
                f"\nCompleted: {task_name}. Elapsed time: {elapsed_time:.2f} seconds.\n"
            )
            sys.stdout.flush()

            # Reset state
            ProgressManager.active_task = None
            ProgressManager.total_steps = None
            ProgressManager.start_time = None

    @staticmethod
    def update_progress(current_step):
        if ProgressManager.active_task is None or ProgressManager.total_steps is None:
            return

        elapsed_time = time.time() - ProgressManager.start_time
        percent_finished = (current_step / ProgressManager.total_steps) * 100
        estimated_total_time = elapsed_time / (
            current_step / ProgressManager.total_steps
        )
        estimated_remaining_time = estimated_total_time - elapsed_time

        message = (
            f"Task: {ProgressManager.active_task} | "
            f"Progress: {percent_finished:.2f}% | "
            f"Elapsed: {elapsed_time:.2f}s | "
            f"Remaining: {estimated_remaining_time:.2f}s"
        )
        # Clear the line before writing the new message
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.write(message)
        sys.stdout.flush()

    @staticmethod
    @contextmanager
    def dummy_context():
        """No-op context manager for when progress tracking is disabled."""
        yield


"""
Processing functions for analyzing and saving external data files
"""


def parse_file(
    file_path, sample_size=None, update_interval=500000, show_progress=False
):
    """
    Parse the file and optionally sample data while reading.

    This version streams the file line by line and updates progress only every
    update_interval lines based on the file's byte size.

    Args:
        file_path (str): Path to the input file.
        sample_size (int, optional): Number of samples to retain (None means full processing).
        update_interval (int, optional): Number of lines before updating progress.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        data (dict): A dictionary mapping binary sequences to their raw counts.
        total_count (float): The sum of counts across all sequences.
    """
    data = {}
    total_count = 0.0
    valid_lines = 0

    file_size = os.path.getsize(file_path)
    bytes_read = 0

    with open(file_path, "r") as file:
        with (
            ProgressManager.progress("Parsing file", total_steps=file_size)
            if show_progress
            else ProgressManager.dummy_context()
        ):
            for idx, line in enumerate(file):
                bytes_read += len(line)
                if show_progress and idx % update_interval == 0:
                    ProgressManager.update_progress(bytes_read)
                try:
                    line = line.strip()
                    if not line:
                        continue

                    binary_sequence, count_str = line.split()
                    count = float(count_str)
                    total_count += count
                    valid_lines += 1

                    if sample_size and len(data) < sample_size:
                        data[binary_sequence] = count
                    elif sample_size:
                        # Reservoir sampling using the count of valid lines.
                        replace_idx = random.randint(0, valid_lines - 1)
                        if replace_idx < sample_size:
                            keys = list(data.keys())
                            data[keys[replace_idx]] = count
                    else:
                        data[binary_sequence] = count

                except Exception as e:
                    print(f"Error reading line '{line}' in {file_path}: {e}")

            if show_progress:
                ProgressManager.update_progress(file_size)

    return data, total_count


def normalize_to_probabilities(data, total_count):
    """
    Convert raw counts to probabilities.

    Returns:
        normalized_data (dict): A dictionary with probabilities.
    """
    if total_count == 0:
        raise ValueError("Total count is zero; cannot normalize to probabilities.")
    normalized_data = {key: value / total_count for key, value in data.items()}
    return normalized_data


def sample_data(
    data, total_count, sample_size, update_interval=100, show_progress=False
):
    """
    Sample bit strings based on their probabilities.

    Args:
        data (dict): Dictionary of raw counts.
        total_count (float): Sum of all counts (used for normalization).
        sample_size (int): Number of samples to generate.
        update_interval (int, optional): Number of samples before updating progress.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        dict: A dictionary mapping sampled bit strings to their probabilities.
    """
    normalized_data = normalize_to_probabilities(data, total_count)
    sequences = list(normalized_data.keys())
    probabilities = list(normalized_data.values())

    sampled_dict = {}

    with (
        ProgressManager.progress("Sampling data", total_steps=sample_size)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        sampled_sequences = random.choices(
            sequences, weights=probabilities, k=sample_size
        )

        for idx, sequence in enumerate(sampled_sequences):
            sampled_dict[sequence] = sampled_dict.get(sequence, 0) + 1
            if show_progress and idx % update_interval == 0:
                ProgressManager.update_progress(idx + 1)

        if show_progress:
            ProgressManager.update_progress(sample_size)  # Ensure 100% completion

    total_sampled_count = sum(sampled_dict.values())
    return {key: count / total_sampled_count for key, count in sampled_dict.items()}


def introduce_error_data(
    data,
    total_count,
    ground_rate=0.01,
    excited_rate=0.08,
    update_interval=100,
    show_progress=False,
):
    """
    Introduce bit-flipping errors to the dataset with separate error rates for ground and excited states.

    If a bit is '1', it has an 'excited_rate' chance of being flipped to '0'.
    Conversely, if a bit is '0', it has a 'ground_rate' chance of being flipped to '1'.

    Args:
        data (dict): Dictionary of raw counts.
        total_count (float): Sum of all counts (used for normalization).
        ground_rate (float, optional): Probability of a '0' flipping to '1'. Default is 0.01.
        excited_rate (float, optional): Probability of a '1' flipping to '0'. Default is 0.08.
        update_interval (int, optional): Number of sequences before updating progress.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        dict: A dictionary with probabilities after errors are introduced.
    """
    print("Introducing errors to the data...")
    normalized_data = normalize_to_probabilities(data, total_count)
    new_data = {}
    sequences = list(normalized_data.keys())

    with (
        ProgressManager.progress("Introducing errors", total_steps=len(sequences))
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for idx, sequence in enumerate(sequences):
            modified_sequence = list(sequence)

            for i in range(len(modified_sequence)):
                if modified_sequence[i] == "1" and random.random() < excited_rate:
                    modified_sequence[i] = "0"
                elif modified_sequence[i] == "0" and random.random() < ground_rate:
                    modified_sequence[i] = "1"

            new_sequence = "".join(modified_sequence)
            new_data[new_sequence] = new_data.get(new_sequence, 0) + 1

            if show_progress and idx % update_interval == 0:
                ProgressManager.update_progress(idx + 1)

        if show_progress:
            ProgressManager.update_progress(len(sequences))  # Ensure 100% completion

    total_new_count = sum(new_data.values())
    return {key: count / total_new_count for key, count in new_data.items()}


def print_most_probable_data(normalized_data, n=10):
    """
    Print the n most probable bit strings with evenly spaced formatting.
    """
    sorted_data = sorted(normalized_data.items(), key=lambda x: x[1], reverse=True)
    print(f"Most probable {n} bit strings:")

    # Find max index width (for up to 99, this is 2)
    max_index_width = len(str(n))

    for idx, (sequence, probability) in enumerate(sorted_data[:n], start=1):
        print(
            f"{str(idx).rjust(max_index_width)}.  Bit string: {sequence}, Probability: {probability:.8f}"
        )


def save_data(data, savefile, update_interval=100, show_progress=False):
    """
    Save the data to a file using the same convention as in parse_file, with optional progress tracking.

    Each line in the file will contain:
        <state> <value>
    where 'state' is the binary sequence and 'value' is the associated count or probability.

    Args:
        data (dict): Dictionary with keys as states and values as counts or probabilities.
        savefile (str): The path to the file where the data will be saved.
        update_interval (int, optional): Frequency at which progress updates occur.
        show_progress (bool, optional): Whether to display progress updates.
    """
    states = list(data.keys())
    total_states = len(states)

    with open(savefile, "w") as f:
        with (
            ProgressManager.progress("Saving data", total_steps=total_states)
            if show_progress
            else ProgressManager.dummy_context()
        ):
            for idx, state in enumerate(states):
                f.write(f"{state} {data[state]}\n")

                if show_progress and idx % update_interval == 0:
                    ProgressManager.update_progress(idx + 1)

            if show_progress:
                ProgressManager.update_progress(total_states)  # Ensure 100% completion


def combine_datasets(data1, data2, tol=1e-6, update_interval=100, show_progress=False):
    """
    Combine two datasets (dictionaries mapping states to counts or probabilities).

    If both datasets are probabilities (sum ≈ 1), combine and renormalize the result so that it sums to 1.
    If both datasets are counts (i.e. neither sums to 1), simply combine the counts without normalization.
    If one dataset is probabilities and the other is counts, raise an error.

    Args:
        data1, data2 (dict): The datasets to combine.
        tol (float, optional): Tolerance for checking if a dataset is probabilities (default is 1e-6).
        update_interval (int, optional): Frequency at which progress updates occur.
        show_progress (bool, optional): Whether to display progress updates.

    Returns:
        combined (dict): The combined dataset.
            - If both inputs are probabilities, the returned dataset is normalized.
            - If both inputs are counts, the returned dataset is not normalized.

    Raises:
        ValueError: If one dataset is probabilities and the other is counts.
    """
    total1 = sum(data1.values())
    total2 = sum(data2.values())

    is_prob1 = abs(total1 - 1.0) < tol
    is_prob2 = abs(total2 - 1.0) < tol

    if is_prob1 and is_prob2:
        data_type = "probabilities"
    elif (is_prob1 and not is_prob2) or (not is_prob1 and is_prob2):
        raise ValueError(
            "Cannot combine a dataset of probabilities with a dataset of counts. "
            "Please convert one to the other before combining."
        )
    else:
        data_type = "counts"

    combined = {}
    all_keys = set(data1.keys()).union(data2.keys())
    total_keys = len(all_keys)

    with (
        ProgressManager.progress("Combining datasets", total_steps=total_keys)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for idx, key in enumerate(all_keys):
            combined[key] = data1.get(key, 0) + data2.get(key, 0)

            if show_progress and idx % update_interval == 0:
                ProgressManager.update_progress(idx + 1)

        if show_progress:
            ProgressManager.update_progress(total_keys)  # Ensure 100% completion

    if data_type == "probabilities":
        combined_total = sum(combined.values())
        combined = {key: value / combined_total for key, value in combined.items()}

    return combined


def save_dict_to_parquet(data_dict, file_name):
    """
    Saves a dictionary of key-value pairs (e.g., {"state": prob}) to a Parquet file.

    Parameters:
        data_dict (dict): A dictionary where keys are states and values are probabilities.
        file_name (str): The name of the Parquet file to save.
    """
    total_steps = 3
    with ProgressManager.progress(
        "Saving dictionary to Parquet", total_steps=total_steps
    ):
        # Step 1: Convert dictionary to a list of items.
        items = list(data_dict.items())
        ProgressManager.update_progress(1)

        # Step 2: Create a DataFrame from the items.
        df = pd.DataFrame(items, columns=["state", "probability"])
        ProgressManager.update_progress(2)

        # Step 3: Save the DataFrame as a Parquet file.
        df.to_parquet(file_name, engine="pyarrow", index=False)
        ProgressManager.update_progress(3)

    print(f"Dictionary saved to {file_name}")


def parse_parq(file_name):
    """
    Reads a Parquet file and converts it back into a dictionary.

    Parameters:
        file_name (str): The Parquet file name to read.

    Returns:
        dict: A dictionary where keys are states and values are probabilities.
    """
    total_steps = 2
    with ProgressManager.progress("Parsing Parquet file", total_steps=total_steps):
        # Step 1: Read the Parquet file into a DataFrame.
        df = pd.read_parquet(file_name, engine="pyarrow")
        ProgressManager.update_progress(1)

        # Step 2: Convert the DataFrame into a dictionary.
        data_dict = dict(zip(df["state"], df["probability"]))
        ProgressManager.update_progress(2)

    return data_dict


"""
Halitonain Solver for QCOM Project. Allows users to build and solve specific hamiltonians. Over time I hope to add more. 
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


"""
Functions for computing quantum information measures in quantum systems. This will be an every growing list of functions.
"""


def von_neumann_entropy_from_rdm(rdm):
    """Computes the Von Neumann Entanglement Entropy given a reduced density matrix."""
    eigenvalues = np.linalg.eigvalsh(rdm)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Avoid log(0)
    return -np.sum(eigenvalues * np.log(eigenvalues))


def von_neumann_entropy_from_hamiltonian(
    hamiltonian, configuration, state_index=0, show_progress=False
):
    """Computes VNEE given a Hamiltonian and partition specification."""
    if not isinstance(hamiltonian, np.ndarray):
        hamiltonian = hamiltonian.toarray()  # Convert sparse to dense

    num_atoms = int(np.log2(hamiltonian.shape[0]))  # Number of atoms in the system
    subsystem_atoms = [i for i, included in enumerate(configuration) if included == 1]
    subsystem_size = len(subsystem_atoms)
    total_steps = (
        5 + num_atoms
    )  # Decomposition, reshaping, tracing steps, and entropy computation
    step = 0

    with (
        ProgressManager.progress("Computing Von Neumann Entropy", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        chosen_eigenvalue, chosen_state = find_eigenstate(
            hamiltonian, state_index, show_progress
        )
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        density_matrix = np.outer(chosen_state, chosen_state.conj())
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        reshaped_matrix = density_matrix.reshape([2] * (2 * num_atoms))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        current_dim = num_atoms
        for atom in reversed(range(num_atoms)):
            if configuration[atom] == 0:
                reshaped_matrix = np.trace(
                    reshaped_matrix, axis1=atom, axis2=atom + current_dim
                )
                current_dim -= 1
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        dim_subsystem = 2**subsystem_size
        reduced_density_matrix = reshaped_matrix.reshape((dim_subsystem, dim_subsystem))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        entropy = von_neumann_entropy_from_rdm(reduced_density_matrix)
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return entropy


def get_eigenstate_probabilities(hamiltonian, state_index=0, show_progress=False):
    """
    Computes the probability distribution of the chosen eigenstate in the computational basis.
    """
    if not isinstance(hamiltonian, np.ndarray):
        hamiltonian = hamiltonian.toarray()

    num_qubits = int(np.log2(hamiltonian.shape[0]))
    hilbert_dim = 2**num_qubits
    total_steps = 4 + hilbert_dim
    step = 0

    with (
        ProgressManager.progress("Computing Eigenstate Probabilities", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        chosen_eigenvalue, chosen_state = find_eigenstate(
            hamiltonian, state_index, show_progress
        )
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        probabilities = np.abs(chosen_state) ** 2
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        state_prob_dict = {
            format(i, f"0{num_qubits}b"): probabilities[i] for i in range(hilbert_dim)
        }
        step += hilbert_dim
        if show_progress:
            ProgressManager.update_progress(total_steps)

    return state_prob_dict


def find_eigenstate(hamiltonian, state_index=0, show_progress=False):
    """
    Computes a specific eigenstate of the Hamiltonian efficiently.
    """
    if not isinstance(hamiltonian, np.ndarray):
        hamiltonian = hamiltonian.toarray()

    with (
        ProgressManager.progress("Finding Eigenstate", 1)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        if show_progress:
            print(
                "\rFinding Eigenstate... This may take some time. Please wait.",
                end="",
                flush=True,
            )
        start_time = time.time()

        if state_index == 0:
            eigenvalues, eigenvectors = eigsh(hamiltonian, k=1, which="SA", tol=1e-10)
        else:
            eigenvalues, eigenvectors = eigsh(
                hamiltonian, k=state_index + 1, which="SA", tol=1e-10
            )

        chosen_eigenvalue = eigenvalues[state_index]
        chosen_eigenvector = eigenvectors[:, state_index]
        end_time = time.time()

        if show_progress:
            print("\r" + " " * 80, end="")
            print(
                f"\rEigenstate {state_index} found in {end_time - start_time:.2f} seconds.",
                flush=True,
            )
            ProgressManager.update_progress(1)

    return chosen_eigenvalue, chosen_eigenvector


def create_density_matrix(eigenvector, show_progress=False):
    """
    Constructs the density matrix from a given eigenvector.
    """
    with (
        ProgressManager.progress("Constructing Density Matrix", 1)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        density_matrix = np.outer(eigenvector, np.conj(eigenvector))
        if show_progress:
            ProgressManager.update_progress(1)
    return density_matrix


def compute_reduced_density_matrix(density_matrix, configuration, show_progress=False):
    """
    Computes the reduced density matrix by tracing out sites marked as 0 in the configuration.
    """
    num_qubits = int(np.log2(density_matrix.shape[0]))
    subsystem_atoms = [i for i, included in enumerate(configuration) if included == 1]
    subsystem_size = len(subsystem_atoms)
    total_steps = 2 + num_qubits
    step = 0

    with (
        ProgressManager.progress("Computing Reduced Density Matrix", total_steps)
        if show_progress
        else ProgressManager.dummy_context()
    ):
        if show_progress:
            print(
                "\rReshaping Density Matrix for Partial Trace... Please wait.",
                end="",
                flush=True,
            )
        start_time = time.time()

        reshaped_matrix = density_matrix.reshape([2] * (2 * num_qubits))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        current_dim = num_qubits
        for atom in reversed(range(num_qubits)):
            if configuration[atom] == 0:
                reshaped_matrix = np.trace(
                    reshaped_matrix, axis1=atom, axis2=atom + current_dim
                )
                current_dim -= 1
                step += 1
                if show_progress:
                    ProgressManager.update_progress(min(step, total_steps))

        dim_subsystem = 2**subsystem_size
        reduced_density_matrix = reshaped_matrix.reshape((dim_subsystem, dim_subsystem))
        step += 1
        if show_progress:
            ProgressManager.update_progress(min(step, total_steps))

        end_time = time.time()
        if show_progress:
            print("\r" + " " * 80, end="")
            print(
                f"\rReduced Density Matrix computed in {end_time - start_time:.2f} seconds.",
                flush=True,
            )
            ProgressManager.update_progress(total_steps)

    return reduced_density_matrix


"""
Functions for computing classical information measures in quantum systems. This will be an every growing list of functions.
"""


def order_dict(inp_dict):
    """
    Orders a dictionary based on binary keys interpreted as integers.

    Args:
        inp_dict (dict): Dictionary where keys are binary strings.

    Returns:
        dict: Ordered dictionary sorted by integer values of binary keys.
    """
    ordered_items = sorted(inp_dict.items(), key=lambda item: int(item[0], 2))
    return dict(ordered_items)


def part_dict(inp_dict, indices):
    """
    Extracts a subset of bits from each binary string based on given indices.

    Args:
        inp_dict (dict): Dictionary where keys are binary strings.
        indices (list): List of indices specifying which bits to extract.

    Returns:
        dict: New dictionary where keys contain only the extracted bits.
    """
    new_dict = {}

    for key, value in inp_dict.items():
        extracted_bits = "".join(
            key[i] for i in indices
        )  # Extract only relevant indices
        if extracted_bits in new_dict:
            new_dict[extracted_bits] += value  # Sum probabilities for duplicates
        else:
            new_dict[extracted_bits] = value

    return new_dict


def compute_shannon_entropy(prob_dict):
    """
    Computes the Shannon entropy of a probability distribution.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.

    Returns:
        float: Shannon entropy.
    """
    total_prob = sum(prob_dict.values())
    entropy = -sum(
        (p / total_prob) * np.log(p / total_prob) for p in prob_dict.values() if p > 0
    )
    return entropy


def compute_reduced_shannon_entropy(prob_dict, configuration, target_region):
    """
    Computes the reduced Shannon entropy for a given region.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to which region (0 for A, 1 for B).
        target_region (int): The target region for entropy calculation (0 for A, 1 for B).

    Returns:
        float: Reduced Shannon entropy.
    """
    # Extract the indices corresponding to the target region (0 for A, 1 for B)
    target_indices = [
        i for i, region in enumerate(configuration) if region == target_region
    ]

    # Obtain the reduced dictionary using the extracted indices
    reduced_dict = part_dict(prob_dict, target_indices)
    reduced_dict = order_dict(reduced_dict)  # Order for consistency

    # Get sorted sequences and corresponding probabilities
    sorted_sequences = list(reduced_dict.keys())
    sorted_probabilities = list(reduced_dict.values())

    # Compute total probability for normalization
    total_prob = sum(sorted_probabilities)

    # Identify unique leftmost parts of the sequences
    unique_leftmost_parts = []
    previous_leftmost = None

    for sequence in sorted_sequences:
        leftmost_part = sequence[: len(target_indices)]  # Extract relevant part
        if leftmost_part != previous_leftmost:
            unique_leftmost_parts.append(leftmost_part)
            previous_leftmost = leftmost_part

    # Initialize reduced probabilities array
    reduced_probabilities = np.zeros(len(unique_leftmost_parts))

    # Sum probabilities for each unique leftmost part
    current_index = -1
    previous_leftmost = None

    for index, sequence in enumerate(sorted_sequences):
        leftmost_part = sequence[: len(target_indices)]
        if leftmost_part != previous_leftmost:
            current_index += 1
            reduced_probabilities[current_index] += sorted_probabilities[index]
            previous_leftmost = leftmost_part
        else:
            reduced_probabilities[current_index] += sorted_probabilities[index]

    # Compute the reduced Shannon entropy
    reduced_entropy = -sum(
        (p / total_prob) * np.log(p / total_prob)
        for p in reduced_probabilities
        if p > 0
    )

    return reduced_entropy


def compute_mutual_information(prob_dict, configuration):
    """
    Computes the classical mutual information between two regions.

    Args:
        prob_dict (dict): A dictionary mapping states to their probabilities.
        configuration (list): A binary list specifying which sites belong to which region (0 for A, 1 for B).

    Returns:
        tuple: (mutual information, Shannon entropy of A, Shannon entropy of B, Shannon entropy of AB)
    """
    # Compute full system entropy
    shan_AB = compute_shannon_entropy(prob_dict)

    # Compute reduced Shannon entropy for region A and B
    shan_A = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=0)
    shan_B = compute_reduced_shannon_entropy(prob_dict, configuration, target_region=1)

    # Compute mutual information
    mutual_information = shan_A + shan_B - shan_AB

    return mutual_information


def cumulative_distribution(binary_dict):
    """
    Compute the cumulative probability distribution from a given binary probability dictionary.

    Args:
        binary_dict (dict): A dictionary where keys are binary strings representing states,
                            and values are their corresponding probabilities.

    Returns:
        tuple: (x_axis, y_axis) representing the cumulative probability distribution.
    """
    # Extract and sort probabilities from the dictionary
    probabilities = np.array(list(binary_dict.values()))
    sorted_probs = np.sort(probabilities)

    # Compute cumulative distribution
    unique_probs, counts = np.unique(sorted_probs, return_counts=True)
    cumulative_prob = np.cumsum(unique_probs * counts)

    # Normalize cumulative probability to ensure it ranges from 0 to 1
    cumulative_prob /= cumulative_prob[-1]

    # Ensure x-axis spans from the smallest probability to 1
    x_axis = np.append(unique_probs, [1])
    y_axis = np.append(cumulative_prob, [1])  # Ensure y-axis ends at 1

    return x_axis, y_axis


def compute_N_of_p_all(probabilities, p_delta=0.1, show_progress=False):
    """
    Efficiently compute N(p) for each unique nonzero probability.

    Args:
        probabilities (array-like): List or array of probabilities.
        p_delta (float): Width in log10 space for the neighborhood.
        show_progress (bool): Whether to show progress updates.

    Returns:
        tuple: (unique_probs, N_values)
    """
    probs = np.array(probabilities)
    probs = probs[probs > 0]
    sorted_probs = np.sort(probs)
    cumulative_probs = np.cumsum(sorted_probs)
    unique_probs = np.unique(sorted_probs)

    def compute_single_N(p):
        log_p = np.log10(p)
        lower = 10 ** (log_p - p_delta / 2)
        upper = 10 ** (log_p + p_delta / 2)

        lower_idx = np.searchsorted(sorted_probs, lower, side="left")
        upper_idx = np.searchsorted(sorted_probs, upper, side="right")

        sigma_lower = cumulative_probs[lower_idx - 1] if lower_idx > 0 else 0.0
        sigma_upper = cumulative_probs[upper_idx - 1] if upper_idx > 0 else 0.0

        return (sigma_upper - sigma_lower) / ((upper - lower) * p)

    N_values = []

    with (
        ProgressManager.progress("Computing N(p)", total_steps=len(unique_probs))
        if show_progress
        else ProgressManager.dummy_context()
    ):
        for i, p in enumerate(unique_probs):
            N_values.append(compute_single_N(p))
            if show_progress:
                ProgressManager.update_progress(i + 1)

    return unique_probs, N_values


def compute_N_of_p(p, sorted_probs, cumulative_probs, p_delta=0.1):
    """
    Compute N(p) at a single value using precomputed arrays.

    Args:
        p (float): Probability to evaluate N(p).
        sorted_probs (np.array): Sorted array of nonzero probabilities.
        cumulative_probs (np.array): Cumulative sum of sorted_probs.
        p_delta (float): Width in log10 space.

    Returns:
        float: N(p)
    """
    if p <= 0:
        return 0.0

    log_p = np.log10(p)
    lower = 10 ** (log_p - p_delta / 2)
    upper = 10 ** (log_p + p_delta / 2)

    lower_idx = np.searchsorted(sorted_probs, lower, side="left")
    upper_idx = np.searchsorted(sorted_probs, upper, side="right")

    sigma_lower = cumulative_probs[lower_idx - 1] if lower_idx > 0 else 0.0
    sigma_upper = cumulative_probs[upper_idx - 1] if upper_idx > 0 else 0.0

    return (sigma_upper - sigma_lower) / ((upper - lower) * p)
