from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NeutrinoTransferCorrection:
    """Tabulated massive-neutrino correction to the cb Poisson source.

    For cb perturbation theory, the linear Poisson source is proportional to
    total matter. The correction is

        mu_nu(k, eta) = delta_tot(k, eta) / delta_nonu(k, eta)
                      = delta_m(k, eta) / delta_cb(k, eta).

    In cosmoprimo/CAMB transfer-table naming:
        delta_m  -> delta_tot
        delta_cb -> delta_nonu

    This class is a data container, not a user-facing switch. The top-level
    theory wrapper owns the sole public ``include_neutrino_corrections`` flag:
    if enabled it constructs and passes this object; if disabled it passes None.
    """

    k: np.ndarray
    eta: np.ndarray
    mu_nu: np.ndarray
    log_interp: bool = True
    fill_value: float | None = None

    def __post_init__(self):
        self.k = np.asarray(self.k, dtype=float).reshape(-1)
        self.eta = np.asarray(self.eta, dtype=float).reshape(-1)
        self.mu_nu = np.asarray(self.mu_nu, dtype=float)

        if self.k.size == 0 or self.eta.size == 0:
            raise ValueError("k and eta must both contain at least one value.")

        expected = (self.eta.size, self.k.size)
        if self.mu_nu.shape != expected:
            raise ValueError(
                f"mu_nu must have shape (neta, nk) = {expected}, "
                f"got {self.mu_nu.shape}."
            )

        order_eta = np.argsort(self.eta)
        self.eta = self.eta[order_eta]
        self.mu_nu = self.mu_nu[order_eta]

        order_k = np.argsort(self.k)
        self.k = self.k[order_k]
        self.mu_nu = self.mu_nu[:, order_k]

        if np.any(self.k <= 0.0):
            raise ValueError("All k values must be positive for log-k interpolation.")
        if np.any(np.diff(self.k) == 0.0):
            raise ValueError("k values must be unique.")
        if np.any(np.diff(self.eta) == 0.0):
            raise ValueError("eta values must be unique.")

        self._xk = np.log(self.k) if self.log_interp else self.k

    @classmethod
    def from_cosmoprimo_transfer_table(
        cls,
        table: Any,
        *,
        numerator: str = "delta_tot",
        denominator: str = "delta_nonu",
        log_interp: bool = True,
        fill_value: float | None = None,
    ) -> "NeutrinoTransferCorrection":
        """Build ``mu_nu`` from ``cosmo.get_transfer().table()``."""
        names = table.dtype.names or ()
        for name in ("k", "z", numerator, denominator):
            if name not in names:
                raise ValueError(
                    f"Transfer table is missing {name!r}. Available columns: {names}."
                )

        k_arr = np.asarray(table["k"], dtype=float)
        z_arr = np.asarray(table["z"], dtype=float)
        numerator_arr = np.asarray(table[numerator], dtype=float)
        denominator_arr = np.asarray(table[denominator], dtype=float)

        # cosmoprimo/CAMB normally stores transfer quantities as (nk, nz).
        k = k_arr[:, 0] if k_arr.ndim == 2 else k_arr
        z = z_arr[0, :] if z_arr.ndim == 2 else z_arr
        eta = np.log(1.0 / (1.0 + z))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator_arr / denominator_arr

        # Store in the interpolation layout (neta, nk).
        if ratio.shape == (k.size, eta.size):
            ratio = ratio.T
        elif ratio.shape != (eta.size, k.size):
            raise ValueError(
                f"Could not interpret transfer-ratio shape {ratio.shape}; "
                f"expected {(k.size, eta.size)} or {(eta.size, k.size)}."
            )

        # A malformed transfer value must not inject NaN into the ODE.
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)

        return cls(
            k=k,
            eta=eta,
            mu_nu=ratio,
            log_interp=log_interp,
            fill_value=fill_value,
        )

    def __call__(self, eta, k):
        """Evaluate ``mu_nu(k, eta)`` with bilinear (eta, ln k) interpolation."""
        eta_in = np.asarray(eta, dtype=float)
        k_in = np.asarray(k, dtype=float)
        eta_b, k_b = np.broadcast_arrays(eta_in, k_in)

        flat_eta = eta_b.ravel()
        flat_k = k_b.ravel()

        if self.log_interp:
            xk = np.log(np.clip(flat_k, self.k[0], self.k[-1]))
        else:
            xk = np.clip(flat_k, self.k[0], self.k[-1])

        outside_eta = (flat_eta < self.eta[0]) | (flat_eta > self.eta[-1])
        eta_eval = (
            flat_eta
            if self.fill_value is not None
            else np.clip(flat_eta, self.eta[0], self.eta[-1])
        )

        out = np.empty_like(eta_eval, dtype=float)

        for i, (ee, xx) in enumerate(zip(eta_eval, xk)):
            if self.fill_value is not None and outside_eta[i]:
                out[i] = float(self.fill_value)
                continue

            if self.eta.size == 1:
                out[i] = np.interp(xx, self._xk, self.mu_nu[0])
                continue

            j = int(np.clip(
                np.searchsorted(self.eta, ee, side="right") - 1,
                0,
                self.eta.size - 2,
            ))
            e0, e1 = self.eta[j], self.eta[j + 1]
            t = 0.0 if e1 == e0 else (ee - e0) / (e1 - e0)

            y0 = np.interp(xx, self._xk, self.mu_nu[j])
            y1 = np.interp(xx, self._xk, self.mu_nu[j + 1])
            out[i] = (1.0 - t) * y0 + t * y1

        out = out.reshape(eta_b.shape)
        return float(out) if out.ndim == 0 else out
