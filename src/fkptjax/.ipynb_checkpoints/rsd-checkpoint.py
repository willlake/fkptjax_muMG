"""Redshift-space FKPT/FOLPS projection helpers.

This module is the boundary between FKPT table construction and the FOLPS RSD
model.  It is designed so that external likelihood wrappers can pass
already-built FKPT tables, AP-rescaled ``(k, mu)`` grids, and nuisance/bias
parameters, while the FOLPS API details stay inside ``fkptjax``.

The usual chain is::

    table, table_now, _ = Kfuncs_to_tables_*(...)
    table_all = table[1:28] + table_now[1:28]
    pars = pack_fkpt_bias(params, nd=nbar)
    poles = tables_to_poles(jac, kap, muap, pars, table_all, mu, wmu,
                            ncols=27, bias_scheme="folps")

``tables_to_poles`` is JAX-jitted.  All configuration choices that affect Python
control flow in FOLPS are static arguments.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping, MutableMapping, Sequence

import jax.numpy as jnp
from jax import jit


FKPT_BIAS_ORDER: tuple[str, ...] = (
    "b1", "b2", "bs2", "b3nl",
    "alpha0", "alpha2", "alpha4",
    "ctilde", "alpha0shot", "alpha2shot",
    "PshotP", "X_FoG_p",
)
"""Bias/nuisance order expected by the FOLPS EFT RSD calculator."""


FKPT_BIAS_ALIASES: dict[str, tuple[str, ...]] = {
    "b1": ("b1", "b1p"),
    "b2": ("b2", "b2p"),
    "bs2": ("bs2", "bK2", "bK2p"),
    "b3nl": ("b3nl", "btd", "btdp"),
    "alpha0": ("alpha0", "alpha0p"),
    "alpha2": ("alpha2", "alpha2p"),
    "alpha4": ("alpha4", "alpha4p"),
    "ctilde": ("ctilde", "ctildep"),
    "alpha0shot": ("alpha0shot", "alpha0shotp"),
    "alpha2shot": ("alpha2shot", "alpha2shotp"),
    "PshotP": ("PshotP",),
    "X_FoG_p": ("X_FoG_p", "X_FoG"),
}
"""Accepted aliases for common FKPT/AP-scaling parameter names."""


__all__ = [
    "FKPT_BIAS_ORDER",
    "FKPT_BIAS_ALIASES",
    "pack_fkpt_bias",
    "bias_dict_with_defaults",
    "project_to_poles",
    "tables_to_pkmu",
    "tables_to_poles",
]


def _lookup_alias(params: Mapping[str, Any], name: str, default: Any = None, *, required: bool = True) -> Any:
    """Return ``params[name]`` allowing the aliases in ``FKPT_BIAS_ALIASES``."""
    for candidate in FKPT_BIAS_ALIASES.get(name, (name,)):
        if candidate in params:
            return params[candidate]
    if required:
        aliases = ", ".join(FKPT_BIAS_ALIASES.get(name, (name,)))
        raise KeyError(f"Missing FKPT bias parameter {name!r}; accepted aliases: {aliases}")
    return default


def bias_dict_with_defaults(
    params: Mapping[str, Any],
    nd: float | None = None,
    *,
    defaults: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Return a canonical FKPT/FOLPS bias dictionary.

    Parameters
    ----------
    params
        Input parameter dictionary.  It may use either the canonical FOLPS names
        in :data:`FKPT_BIAS_ORDER` or the aliases in
        :data:`FKPT_BIAS_ALIASES`.
    nd
        Tracer number density.  If ``PshotP`` is absent and ``nd`` is provided,
        this sets ``PshotP = 1 / nd``.  Pass ``PshotP`` explicitly if your input
        already stores the physical shot-noise amplitude.
    defaults
        Optional default values.  Defaults are applied before the mandatory
        checks.  ``X_FoG_p`` defaults to zero.  ``PshotP`` defaults to ``1 / nd``
        when ``nd`` is provided.
    strict
        If ``True``, every parameter in :data:`FKPT_BIAS_ORDER` must be present
        after aliases/defaults are applied.  If ``False``, missing parameters are
        filled with zero, except ``PshotP`` which still uses ``1 / nd`` when
        available.

    Returns
    -------
    dict
        Canonical dictionary ordered according to :data:`FKPT_BIAS_ORDER`.
    """
    local: MutableMapping[str, Any] = dict(defaults or {})
    local.update(dict(params))

    if "PshotP" not in local and nd is not None:
        local["PshotP"] = 1.0 / nd
    local.setdefault("X_FoG_p", 0.0)

    out: dict[str, Any] = {}
    for name in FKPT_BIAS_ORDER:
        if strict:
            out[name] = _lookup_alias(local, name, required=True)
        else:
            out[name] = _lookup_alias(local, name, default=0.0, required=False)
    return out


def pack_fkpt_bias(
    params: Mapping[str, Any] | Sequence[Any],
    nd: float | None = None,
    *,
    order: Sequence[str] = FKPT_BIAS_ORDER,
    defaults: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> Any:
    """Pack FKPT/FOLPS bias parameters into a JAX array.

    If ``params`` is already a sequence/array rather than a mapping, it is simply
    converted with ``jnp.asarray``.  If it is a mapping, aliases are resolved and
    the output follows ``order``.

    This helper keeps the old compact API available while keeping downstream
    wrapper code thin::

        pars = pack_fkpt_bias(params, nd=nbar)

    Parameters are documented in :func:`bias_dict_with_defaults`.
    """
    if isinstance(params, Mapping):
        canonical = bias_dict_with_defaults(params, nd=nd, defaults=defaults, strict=strict)
        return jnp.asarray([canonical[name] for name in order])
    return jnp.asarray(params)


def project_to_poles(pkmu: Any, mu: Any, wmu: Any, ells: Sequence[int] = (0, 2, 4)) -> Any:
    """Project ``P(k, mu)`` to Legendre multipoles.

    Parameters
    ----------
    pkmu
        Array with shape ``(nk, nmu)``.
    mu, wmu
        Mu nodes and weights.  These should be the same nodes used for ``pkmu``.
    ells
        Multipoles to return.  Currently supports ``0, 2, 4``.

    Returns
    -------
    array
        Multipoles with shape ``(len(ells), nk)``.
    """
    pkmu = jnp.asarray(pkmu)
    mu = jnp.asarray(mu)
    wmu = jnp.asarray(wmu)

    out = []
    for ell in tuple(ells):
        if ell == 0:
            leg = jnp.ones_like(mu)
        elif ell == 2:
            leg = 0.5 * (3.0 * mu**2 - 1.0)
        elif ell == 4:
            leg = (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0
        else:
            raise ValueError(f"Unsupported ell={ell}; supported values are 0, 2, 4")
        out.append((2 * ell + 1) * jnp.sum(wmu[None, :] * pkmu * leg[None, :], axis=-1))
    return jnp.asarray(out)


@partial(
    jit,
    static_argnames=(
        "ncols",
        "bias_scheme",
        "IR_resummation",
        "damping",
        "A_full",
        "use_TNS_model",
    ),
)
def tables_to_pkmu(
    kap: Any,
    muap: Any,
    pars: Any,
    table_all: Sequence[Any],
    *,
    ncols: int,
    bias_scheme: str = "folps",
    IR_resummation: bool = True,
    damping: Any = None,
    A_full: bool = False,
    use_TNS_model: bool = False,
) -> Any:
    """Evaluate the FOLPS redshift-space ``P(k, mu)`` from FKPT tables.

    Parameters
    ----------
    kap, muap
        AP-distorted ``k`` and ``mu`` arrays, usually with shape ``(nk, nmu)``.
    pars
        Bias/nuisance vector or dictionary accepted by FOLPS
        ``RSDMultipolesPowerSpectrumCalculator.set_bias_scheme``.
    table_all
        Concatenated tuple/list ``table[1:28] + table_now[1:28]``.
    ncols
        Number of columns belonging to the wiggle table.  For the current
        A_full=False FKPT layout this is ``27``.
    bias_scheme, IR_resummation, damping
        Passed through to FOLPS.
    A_full, use_TNS_model
        Passed to ``folps.MatrixCalculator``.  The current FKPT layout uses
        ``A_full=False`` and ``use_TNS_model=False``.

    Returns
    -------
    array
        ``P(k, mu)`` on the AP grid.
    """
    import folps as folpsv2

    # Keep the FOLPS matrix configuration local to fkptjax.  It is executed at
    # JAX trace time.
    folpsv2.MatrixCalculator(A_full=A_full, use_TNS_model=use_TNS_model)

    calc = folpsv2.RSDMultipolesPowerSpectrumCalculator(model="EFT")
    pars2 = calc.set_bias_scheme(pars=pars, bias_scheme=bias_scheme)
    return calc.get_rsd_pkmu(
        kap,
        muap,
        pars2,
        table_all[:ncols],
        table_all[ncols:],
        IR_resummation=IR_resummation,
        damping=damping,
    )


@partial(
    jit,
    static_argnames=(
        "ncols",
        "bias_scheme",
        "IR_resummation",
        "damping",
        "ells",
        "A_full",
        "use_TNS_model",
    ),
)
def tables_to_poles(
    jac: Any,
    kap: Any,
    muap: Any,
    pars: Any,
    table_all: Sequence[Any],
    mu: Any,
    wmu: Any,
    *,
    ncols: int,
    bias_scheme: str = "folps",
    IR_resummation: bool = True,
    damping: Any = None,
    ells: Sequence[int] = (0, 2, 4),
    A_full: bool = False,
    use_TNS_model: bool = False,
) -> Any:
    """Evaluate and project the FOLPS RSD model to multipoles.

    This is the main function external wrappers should call after FKPT table
    construction.  It performs the block that previously lived downstream::

        folps.MatrixCalculator(A_full=False, use_TNS_model=False)
        calc = folps.RSDMultipolesPowerSpectrumCalculator(model="EFT")
        pars2 = calc.set_bias_scheme(...)
        pkmu = calc.get_rsd_pkmu(...)
        poles = to_poles(jac * pkmu)

    Parameters are the same as :func:`tables_to_pkmu`, with the addition of
    ``jac``, ``mu``, ``wmu`` and ``ells`` for AP Jacobian and multipole
    projection.
    """
    pkmu = tables_to_pkmu(
        kap,
        muap,
        pars,
        table_all,
        ncols=ncols,
        bias_scheme=bias_scheme,
        IR_resummation=IR_resummation,
        damping=damping,
        A_full=A_full,
        use_TNS_model=use_TNS_model,
    )
    return project_to_poles(jac * pkmu, mu, wmu, ells=ells)
