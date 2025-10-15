"""
LatticeRegister — explicit lattice-site layouts for Hamiltonian builders (1D/2D/3D)

Goal
-----
Provide a transparent way to define a register of lattice sites that is guaranteed
to be compatible with QCOM's Hamiltonian builders. The design makes it clear:

  • Where each site lives in space.
  • Which qubit/bit in the computational basis corresponds to which site.

Key guarantees
--------------
• Dimensionality: Internally always 3D (shape (N, 3)), using SI meters.
  If you only need 1D or 2D, supply coordinates with zeros in the unused axes.

• Bitstring ↔ index mapping: Insertion order defines the index.
  If you add sites in the order A, B, C, then the resulting bitstring
  (MSB=site 0, by our convention) will unambiguously reflect that order.
  There is no hidden reordering.

• No conventions, no “rows/columns”: LatticeRegister makes no attempt to define
  grids, ladders, or canonical layouts. Arbitrary positions are supported,
  and the user lives and dies by the order in which sites were added.

• No duplicates: exact coordinate duplicates are rejected.

Hamiltonian-builder contract
----------------------------
A LatticeRegister exposes:
    - 'positions': np.ndarray, shape (N, 3), dtype float64, in meters
    - '__len__()': returns N
    - 'distances()': pairwise distance matrix in meters (NxN)

Builders should depend only on this contract.

Future extensions (non-breaking)
--------------------------------
• Tolerance-based duplicate detection or minimum-spacing checks.
• Named sites / tags with index maps (e.g., {"A0": 0, "A1": 1}).
• Export/import: JSON, CSV, or QuEra/AHS-compatible formats.
• Visualization helpers (2D/3D plots, blockade graphs).
"""

# ------------------------------------------ Imports ------------------------------------------

import numpy as np
from collections.abc import Iterable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

# ------------------------------------------ LatticeRegister Class ------------------------------------------

class LatticeRegister:
    """
    LatticeRegister — explicit lattice-site layouts for Hamiltonian builders (1D/2D/3D).

    Conventions:
      • Coordinates are 3D (x, y, z) in SI meters.
      • Indexing is insertion order. (MSB ↔ site 0 by QCOM convention.)
    """
    # ------------------------------------------ Construction and Attributes ------------------------------------------

    def __init__(self, positions: Iterable[tuple[float, float, float]] | None = None):
        """
        Initialize a LatticeRegister.

        Args:
            positions (Optional[Iterable[Tuple[float, float, float]]]):
                A sequence of 3D coordinates (x, y, z) in meters.
                Each entry must be a tuple/list of three floats.
                Example for one site: [(0.0, 0.0, 0.0)]
                Example for two sites: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
                If None, an empty register is created.

        Raises:
            ValueError:
                - If 'positions' cannot be converted to an array of shape (N, 3).
                - If any coordinate is non-finite (NaN or Inf).
                - If duplicate coordinates are provided.

        Attributes:
            _pos (np.ndarray, shape (N, 3), dtype float64):
                Internal mutable array of site coordinates.

            positions (np.ndarray, shape (N, 3), dtype float64, read-only view):
                Read-only property exposing the site coordinates to users.
                Attempting in-place modification raises an error.
        """
        if positions is None:
            # Empty register: 0 rows, 3 columns (no sites yet).
            arr = np.zeros((0, 3), dtype=np.float64)
        else:
            rows = list(positions)                       # <-- accept generators / any iterable
            arr = np.array(rows, dtype=np.float64)

            # Require a 2D matrix with 3 columns: (N, 3).
            # If you intended a single site, call LatticeRegister([(x, y, z)]).
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    "positions must be a 2D array of shape (N, 3). "
                    "For a single site, pass [(x, y, z)], not (x, y, z)."
                )

            # Require finite numeric coordinates (no NaN/Inf).
            if not np.isfinite(arr).all():
                raise ValueError("positions must contain only finite numbers (no NaN/Inf).")

            # Exact duplicate check (no coordinate may appear twice).
            seen = {tuple(row) for row in arr}
            if len(seen) != arr.shape[0]:
                raise ValueError("Duplicate site positions are not allowed.")

        # Normalize storage (internal, mutable).
        self._pos = np.ascontiguousarray(arr, dtype=np.float64)

    # ------------------------------------------ Properties ------------------------------------------

    @property
    def positions(self) -> np.ndarray:
        """
        Read-only view of site positions.

        Returns:
            np.ndarray: Array of shape (N, 3), dtype float64, in meters.
                        Attempting to modify in place raises an error.
        """
        view = self._pos.view()
        view.setflags(write=False)  # freeze the view, not the internal array
        return view

    # ------------------------------------------ Methods ------------------------------------------

    def __len__(self) -> int:
        """Return the number of sites in the register."""
        return self._pos.shape[0]

    # ------------------------------------------ New Method ------------------------------------------

    def add(self, position: tuple[float, float, float]) -> int:
        """
        Append a single site to the register in insertion order.

        Args:
            position (Tuple[float, float, float]):
                The (x, y, z) coordinates in meters for the new site.

        Returns:
            int:
                The index assigned to the newly added site (0-based).
                Indexing follows insertion order (MSB ↔ site 0 by QCOM convention).

        Raises:
            ValueError:
                - If 'position' is not a length-3 tuple/list.
                - If any coordinate is non-finite (NaN or Inf).
                - If a site with exactly the same coordinates already exists.

        Notes:
            - This method enforces strict 3D coordinates in SI units.
            - Exact duplicates are not allowed (no tolerance).
              If you need tolerance-based deduplication later, add a separate API.
        """
        # Validate container and length
        if not isinstance(position, (tuple, list)) or len(position) != 3:
            raise ValueError("add() expects a 3-tuple/list (x, y, z) in meters.")

        # Convert and validate finiteness
        p = np.asarray(position, dtype=np.float64)
        if not np.isfinite(p).all():
            raise ValueError("add() received non-finite coordinate (NaN/Inf).")

        # Exact-duplicate check (strict, no tolerance).
        # self._pos == p broadcasts p against each row of self._pos, producing an (N,3) boolean array.
        # np.all(..., axis=1) yields an (N,) mask: True where a row matches p in all three coords.
        # np.any(...) is True if any row matches => duplicate.
        if self._pos.size and np.any(np.all(self._pos == p, axis=1)):
            raise ValueError(f"Site already exists at position {tuple(p)}")

        # Append in insertion order (maintain contiguous float64 storage)
        # _pos should always have shape (N, 3) with N being the number of sites. If N=0 then we need to reshape p to be (1, 3) so that vstack works correctly.
        if self._pos.shape[0] == 0:
            self._pos = p.reshape(1, 3)
        else:
            # Stack row-wise so the new point p becomes the last row (preserving insertion order).
            # np.vstack([self._pos, p]) requires p to be (1,3) or (3,) — NumPy will broadcast (3,) into a row.
            # np.ascontiguousarray(...) enforces C-contiguous float64 storage (performance/interop); logic is unchanged. 
            self._pos = np.ascontiguousarray(np.vstack([self._pos, p]), dtype=np.float64)

        # Return index of the newly added site
        return self._pos.shape[0] - 1

    # ------------------------------------------ New Method ------------------------------------------

    def remove(self, index: int) -> int:
        """
        Remove the site at the specified index.

        Args:
            index (int):
                Zero-based index of the site to remove.

        Returns:
            int:
                The index that was removed (useful for logging).

        Raises:
            IndexError:
                If 'index' is out of bounds.
        """
        n = self._pos.shape[0]
        if index < 0 or index >= n:
            raise IndexError(f"remove(): index {index} out of bounds for N={n}")

        # Delete row and keep internal storage contiguous float64.
        self._pos = np.ascontiguousarray(np.delete(self._pos, index, axis=0), dtype=np.float64)
        return index
    
    # ------------------------------------------ New Method ------------------------------------------

    def position(self, index: int) -> tuple[float, float, float]:
        """
        Return the (x, y, z) coordinates (meters) of the site at 'index'.

        Raises:
            IndexError: If 'index' is out of bounds.
        """
        n = self._pos.shape[0]
        if index < 0 or index >= n:
            raise IndexError(f"position(): index {index} out of bounds for N={n}")
        x, y, z = map(float, self._pos[index])
        return (x, y, z)

    # ------------------------------------------ New Method ------------------------------------------

    def as_array(self) -> np.ndarray:
        """
        Return a **copy** of the positions array with shape (N, 3).

        Notes:
            - This is a copy; modifying it will not affect the register.
            - Use `.positions` property for a read-only view instead.
        """
        return self._pos.copy()
    
    # ------------------------------------------ New Method ------------------------------------------

    def index_map(self, max_rows: int | None = None) -> str:
        """
        Return a formatted string mapping indices → (x, y, z) meters.

        Args:
            max_rows (Optional[int]):
                If provided, truncate the output to the first `max_rows` rows.

        Example output:
            index  x (m)           y (m)           z (m)
            -----  --------------  --------------  --------------
            0      0.000000e+00    0.000000e+00    0.000000e+00
            1      5.000000e-06    0.000000e+00    0.000000e+00
        """
        X = self._pos
        n = X.shape[0]
        end = n if max_rows is None else min(max_rows, n)
        lines = [
            "index  x (m)           y (m)           z (m)",
            "-----  --------------  --------------  --------------",
        ]
        for i in range(end):
            x, y, z = X[i]
            lines.append(f"{i:<5d}  {x:>14.6e}  {y:>14.6e}  {z:>14.6e}")
        if end < n:
            lines.append(f"... ({n - end} more)")
        return "\n".join(lines)

    # ------------------------------------------ New Method ------------------------------------------

    def __repr__(self):
        """
        Return the official string representation of the LatticeRegister.

        Returns:
            str:
                A summary string including the number of sites and a preview
                of their coordinates. Called automatically by `repr(obj)` or `print(obj)`.
        """
        n = self._pos.shape[0]
        head = min(3, n)
        preview = ", ".join(
            [f"({x:.3e},{y:.3e},{z:.3e})" for x, y, z in self._pos[:head]]
        )
        more = "" if head == n else f", ... +{n - head} more"
        return f"LatticeRegister(N={n}, positions=[{preview}{more}])"

    # ------------------------------------------ New Method ------------------------------------------

    def distance(self, i: int, j: int) -> float:
        """
        Euclidean distance between site i and site j (meters).

        Args:
            i (int): Index of the first site.
            j (int): Index of the second site.

        Returns:
            float: Distance in meters.

        Raises:
            IndexError: If either index is out of bounds.
        """
        n = self._pos.shape[0]
        if i < 0 or i >= n or j < 0 or j >= n:
            raise IndexError(f"distance(): indices ({i}, {j}) out of bounds for N={n}")

        xi, yi, zi = self._pos[i]
        xj, yj, zj = self._pos[j]
        dx, dy, dz = (xj - xi), (yj - yi), (zj - zi)
        return float(np.sqrt(dx*dx + dy*dy + dz*dz))

    # ------------------------------------------ New Method ------------------------------------------

    def distances(self) -> np.ndarray:
        """
        Full pairwise Euclidean distance matrix (meters).

        Returns:
            np.ndarray: (N, N) array where D[i, j] is the distance between sites i and j.

        Notes:
            - Uses vectorized broadcasting for efficiency.
            - Output dtype is float64; diagonal entries are exactly 0.0.
        """
        X = self._pos  # shape (N, 3)
        # Compute all pairwise differences: shape (N, N, 3)
        diff = X[:, None, :] - X[None, :, :]
        # Row-wise dot product along the last axis -> (N, N) of squared distances
        D2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)
        # sqrt into float64 result
        return np.sqrt(D2, dtype=np.float64)

    # ------------------------------------------ New Method ------------------------------------------

    def clear(self) -> None:
        """
        Remove **all** sites from the register.

        After calling this method:
            • The register is empty (len(register) == 0).
            • Internal storage is reset to shape (0, 3).

        Returns:
            None
        """
        self._pos = np.zeros((0, 3), dtype=np.float64)

    # ------------------------------------------ New Method ------------------------------------------

    def plot(self, show_index: bool = True, default_s: float = 200, **kwargs):
        """
        Visualize the lattice register.

        Behavior:
            • If all z-coordinates are exactly 0.0 → use a 2D scatter plot.
            • Otherwise → use a 3D scatter plot.

        Args:
            show_index (bool):
                If True, annotate each site with its index number.
            default_s (float):
                Default marker size if 's' is not passed via kwargs.
            **kwargs:
                Additional keyword arguments forwarded to `ax.scatter`.

        Returns:
            matplotlib.axes.Axes or mpl_toolkits.mplot3d.Axes3D:
                The axes object used for plotting.

        Notes:
            - Uses Times/serif font for consistency with publications.
            - 2D plots use an equal aspect ratio when spans are non-degenerate.
              If the y-span is (near) zero, the axis is automatically padded and the
              aspect is relaxed to "auto" to keep points visible.
            - 3D plots cannot enforce aspect ratio equally in matplotlib,
              but the visual is still informative.
        """

        # Configure serif fonts globally for plots
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams["mathtext.fontset"] = "stix"

        X = self._pos
        s = kwargs.pop("s", default_s)

        # Case 0: empty register
        if X.shape[0] == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Empty LatticeRegister", ha="center", va="center",
                    fontsize=16, color="gray", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            return ax

        # Case 1: purely 2D (all z == 0)
        if np.allclose(X[:, 2], 0.0):
            fig, ax = plt.subplots()
            scatter = ax.scatter(X[:, 0], X[:, 1], s=s, **kwargs)

            if show_index:
                font_sz = (s**0.5) * 0.6  # scale label with marker diameter
                for i, (x, y, _) in enumerate(X):
                    ax.text(x, y, str(i), ha="center", va="center",
                            fontsize=font_sz, color="white", weight="bold")

            # --- Adaptive padding & aspect control -------------------------------------
            # Compute spans and apply padding on BOTH axes. If one axis is (near) 1D,
            # give the *used* axis a larger margin and the *unused* axis a small band
            # centered on its data so the markers don't sit on the frame.
            x = X[:, 0]; y = X[:, 1]
            xmin, xmax = float(x.min()), float(x.max())
            ymin, ymax = float(y.min()), float(y.max())
            xspan = xmax - xmin
            yspan = ymax - ymin

            # Detect (near) 1D along each axis
            tol_x = 1e-12 * max(1.0, abs(xmin), abs(xmax))
            tol_y = 1e-12 * max(1.0, abs(ymin), abs(ymax))
            near_zero_x = xspan <= tol_x
            near_zero_y = yspan <= tol_y

            # Fractions for padding
            frac_used = 0.05     # margin on the axis that actually varies
            frac_unused = 0.015  # narrow band for the axis that is effectively constant

            if near_zero_x and near_zero_y:
                # Single point: create a small square window around the point
                x0 = 0.5 * (xmin + xmax)
                y0 = 0.5 * (ymin + ymax)
                # Use a nominal size based on 1.0 (data units) so something is visible
                pad = 1.0
                ax.set_xlim(x0 - pad, x0 + pad)
                ax.set_ylim(y0 - pad, y0 + pad)
                ax.set_aspect("auto")

            elif near_zero_x and not near_zero_y:
                # Vertical geometry (x almost constant)
                x0 = 0.5 * (xmin + xmax)
                pad_x = max(frac_unused * (yspan if yspan > 0 else 1.0), tol_x if tol_x > 0 else 1.0)
                ax.set_xlim(x0 - pad_x, x0 + pad_x)

                pad_y = frac_used * yspan if yspan > 0 else 1.0
                ax.set_ylim(ymin - pad_y, ymax + pad_y)

                ax.set_aspect("auto")

            elif near_zero_y and not near_zero_x:
                # Horizontal geometry (y almost constant)
                y0 = 0.5 * (ymin + ymax)
                pad_y = max(frac_unused * (xspan if xspan > 0 else 1.0), tol_y if tol_y > 0 else 1.0)
                ax.set_ylim(y0 - pad_y, y0 + pad_y)

                pad_x = frac_used * xspan if xspan > 0 else 1.0
                ax.set_xlim(xmin - pad_x, xmax + pad_x)

                ax.set_aspect("auto")

            else:
                # Genuinely 2D cloud
                pad_x = frac_used * xspan if xspan > 0 else 1.0
                pad_y = frac_used * yspan if yspan > 0 else 1.0
                ax.set_xlim(xmin - pad_x, xmax + pad_x)
                ax.set_ylim(ymin - pad_y, ymax + pad_y)
                ax.set_aspect("equal")

            ax.set_title("Lattice Register (2D)", fontsize=16)
            ax.set_xlabel(r"$x$ (m)", fontsize=14)
            ax.set_ylabel(r"$y$ (m)", fontsize=14)
            return ax

        # Case 2: general 3D
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=s, **kwargs)

            if show_index:
                font_sz = (s**0.5) * 0.4
                for i, (x, y, z) in enumerate(X):
                    ax.text(x, y, z, str(i), ha="center", va="center",
                            fontsize=font_sz, color="black")

            ax.set_title("Lattice Register (3D)", fontsize=16)
            ax.set_xlabel(r"$x$ (m)", fontsize=14)
            ax.set_ylabel(r"$y$ (m)", fontsize=14)
            ax.set_zlabel(r"$z$ (m)", fontsize=14)
            return ax
