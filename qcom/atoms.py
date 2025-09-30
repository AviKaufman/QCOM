"""
This module contains the source code for the AtomRegister class, which is used to manage and manipulate
atoms in a quantum system. The AtomRegister class provides methods for adding, removing, and manipulating atoms,
as well as prebuilt geometries for common builds such as a chain, ladder, and square lattice.

It also includes methods for visualizing the atom register and saving it to a file.
Future improvements may include adding more geometries, providing more advanced manipulation methods, and improving performance.
For more information, please refer to the documentation.
"""

import numpy as np
import matplotlib.pyplot as plt


class AtomRegister:
    """
    Stores 2D atom positions and provides utilities for lattice generation, distance calculations,
    editing, merging, and plotting.
    """

    def __init__(self, positions=None):
        """
        Initialize an AtomRegister.

        Args:
            positions (Optional[Union[List[Tuple[float, float]], numpy.ndarray]]):
                An iterable of 2-tuples `(x, y)` or an array of shape `(N, 2)`
                giving the coordinates of each atom in the register. If None,
                creates an empty register to which atoms can be added.

        Raises:
            ValueError: If `positions` is provided but cannot be converted to an Nx2 array.

        Attributes:
            positions (numpy.ndarray of shape (N, 2)):
                The internal array of atom coordinates.
        """
        if positions is None:
            self.positions = np.zeros((0, 2), dtype=float)
        else:
            arr = np.array(positions, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    "positions must be convertible to an array of shape (N,2)"
                )
            # --- duplicate check ---
            seen = {tuple(row) for row in arr}
            if len(seen) != arr.shape[0]:
                raise ValueError("Duplicate atom positions are not allowed")
            # -----------------------
            self.positions = arr

    def __len__(self):
        """Return number of atoms."""
        return len(self.positions)

    @classmethod
    def chain(cls, n, a, y=0.0):
        """
        Create a 1D chain of atoms spaced along x at height y.

        Args:
            cls (Type[AtomRegister]):
                The class on which this method is called. Used so that
                subclasses calling this method return an instance of themselves.
            n (int): Number of atoms in the chain.
            a (float): Spacing between adjacent atoms.
            y (float): Vertical offset for all atoms (default: 0.0).

        Returns:
            AtomRegister:
                A new AtomRegister containing the generated chain of atom positions.
        """
        positions = [(i * a, y) for i in range(n)]
        return cls(positions)

    @classmethod
    def ladder(cls, ncol, a, rho, y0=0.0):
        """
        Create a 2D Ladder of atoms spaced by a along x at height y0.

        Args:
            cls (Type[AtomRegister]):
                The class on which this method is called. Used so that
                subclasses calling this method return an instance of themselves.
            ncol (int): Number of columns in the ladder.
            a (float): Spacing between adjacent atoms in the x direction.
            rho (float): Spacing between adjacent atoms in the y direction.
            y0 (float): Vertical offset for all atoms (default: 0.0).

        Returns:
            AtomRegister:
                A new AtomRegister containing the generated ladder of atom positions.
        """

        positions = []
        for i in range(ncol):
            positions.append((i * a, y0))
            positions.append((i * a, y0 + rho))
        return cls(positions)

    @classmethod
    def grid(cls, nx, ny, a, x0=0.0, y0=0.0):
        """
        Create a rectangular grid of nx by ny atoms spaced by a,

        Args:
            cls (Type[AtomRegister]):
                The class on which this method is called. Used so that
                subclasses calling this method return an instance of themselves.
            nx (int): Number of atoms in the x direction.
            ny (int): Number of atoms in the y direction.
            a (float): Spacing between adjacent atoms.
            x0 (float): Horizontal offset for all atoms (default: 0.0).
            y0 (float): Vertical offset for all atoms (default: 0.0).
        Returns:
            AtomRegister:
                A new AtomRegister containing the generated grid of atom positions.

        """

        positions = []
        for i in range(nx):
            for j in range(ny):
                positions.append((x0 + i * a, y0 + j * a))
        return cls(positions)

    def add(self, position):
        """
        Add a new atom at the specified (x, y) positon.
        Args:
            position (Tuple[float, float]): The (x, y) coordinates of the new atom.
        Raises:
            ValueError: If position is not a 2-tuple or cannot be converted to a float.
        """
        if not isinstance(position, (tuple, list)) or len(position) != 2:
            raise ValueError("position must be a 2-tuple or list")
        pos = np.array(position, dtype=float)
        # --- duplicate check ---
        if np.any(np.all(self.positions == pos, axis=1)):
            raise ValueError(f"Atom already exists at position {position}")
        # -----------------------
        self.positions = np.vstack([self.positions, pos])

    def remove(self, index):
        """
        Remove the atom at the specified index.

        Args:
            index (int): The index of the atom to remove.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= len(self.positions):
            raise IndexError("index out of bounds")
        self.positions = np.delete(self.positions, index, axis=0)

    def translate(self, dx, dy):
        """
        Translate all atoms by (dx, dy).

        Args:
            dx (float): Change in x-coordinate.
            dy (float): Change in y-coordinate.
        """
        self.positions += np.array([dx, dy])

    def distance(self, i, j):
        """Compute Euclidean distance between atom i and atom j.

        Args:
            i (int): Index of the first atom.
            j (int): Index of the second atom.
        Returns:
            float: The Euclidean distance between the two atoms.
        """
        xi, yi = self.positions[i]
        xj, yj = self.positions[j]
        return np.hypot(xj - xi, yj - yi)

    def distances(self):
        """
        Return full NxN matrix of pairwise distances.

        Returns:
            numpy.ndarray: A 2D array of shape (N, N) where D[i, j] is the distance between atom i and atom j.
        """
        N = len(self)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = self.distance(i, j)
        return D

    def extend(self, other):
        """
        Append all atoms from another AtomRegister into this one.

        Args:
            other (AtomRegister): Another AtomRegister instance whose atoms will be added.
        """
        if not isinstance(other, AtomRegister):
            raise ValueError("extend() argument must be an AtomRegister")
        # --- duplicate check ---
        for pos in other.positions:
            if np.any(np.all(self.positions == pos, axis=1)):
                raise ValueError(f"Atom already exists at position {tuple(pos)}")
        # -----------------------
        self.positions = np.vstack([self.positions, other.positions])

    def __add__(self, other):
        """
        Return a new AtomRegister combining self and another.

        Args:
            other (AtomRegister): The other register to combine.

        Returns:
            AtomRegister: New register containing atoms from both.
        """
        if not isinstance(other, AtomRegister):
            return NotImplemented
        combined = np.vstack([self.positions, other.positions])
        return self.__class__(combined)

    def plot(self, ax=None, show_index=True, default_s=350, **kwargs):
        """
        Scatter-plot the atom register in 2D, optionally annotating each atom
        with its index. Markers are larger by default.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
            show_index (bool): If True, draw the atom index in the center of each marker.
            default_s (float): Default marker size (area) if 's' not provided in kwargs.
            **kwargs: Additional keyword arguments passed to `ax.scatter` (e.g. c, marker, edgecolors).
        Returns:
            matplotlib.axes.Axes: The axes containing the scatter plot.
        """
        # force serif font
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["mathtext.fontset"] = "stix"

        if ax is None:
            fig, ax = plt.subplots()

        x, y = self.positions[:, 0], self.positions[:, 1]
        # pull out size, or use our larger default
        s = kwargs.pop("s", default_s)
        scatter = ax.scatter(x, y, s=s, **kwargs)

        if show_index:
            # choose a contrasting color for the text (fall back to white)
            text_color = kwargs.get("edgecolors", "white")
            # scale font size proportional to marker diameter (~ sqrt(area))
            font_sz = (s**0.5) * 0.6
            for i, (xi, yi) in enumerate(self.positions):
                ax.text(
                    xi,
                    yi,
                    str(i),
                    ha="center",
                    va="center",
                    fontsize=font_sz,
                    color=text_color,
                    weight="bold",
                )

        ax.set_aspect("equal")
        ax.set_title("Atom Register", fontsize=18)
        ax.set_xlabel(r"$\mu$m", fontsize=18)
        ax.set_ylabel(r"$\mu$m", fontsize=18)
        return ax
