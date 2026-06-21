"""High-level FKPT table-to-RSD pipelines.

The purpose of this module is to give external likelihood wrappers a small,
stable API.  The caller provides linear spectra, AP grids, mu quadrature, and bias parameters; the
functions here choose the table builder, run FOLPS, and return power-spectrum
multipoles plus a lightweight state object useful for diagnostics/emulation.

Three routes are exposed:

``full_method_poles``
    Legacy/eager FKPT table construction through ``Kfuncs_to_tables``.

``binning_jax_poles``
    Fully JAX-traceable PHENOM/binning route through ``Kfuncs_to_tables_jax``.

``rescale_branch_poles``
    JAX-traceable rescaling-branch route through
    ``Kfuncs_to_tables_rescale_jax``.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

import jax.numpy as jnp

from .kfuncs_to_tables import (
    Kfuncs_to_tables,
    Kfuncs_to_tables_jax,
    Kfuncs_to_tables_rescale_jax,
    build_jax_static_ctx,
)
from .rsd import tables_to_pkmu, tables_to_poles


__all__ = [
    "FKPTTableState",
    "split_folps_tables",
    "make_table_state",
    "poles_from_tables",
    "full_method_poles",
    "binning_jax_poles",
    "rescale_branch_poles",
    "build_jax_static_ctx",
]


class FKPTTableState(NamedTuple):
    """Lightweight container for FKPT/FOLPS table products.

    Attributes
    ----------
    table_w, table_now
        Raw table tuples returned by a ``Kfuncs_to_tables*`` builder.
    table, table_now_terms
        Full FOLPS-ready raw tables ``table_w`` and ``table_now``, including
        the ``k`` grid at index ``0``.
    scalars, scalars_now
        Diagnostic scalar tails ``table_w[28:]`` and ``table_now[28:]``.
        These tails are also included in ``table`` and ``table_now_terms``.
    table_all
        Concatenated ``table + table_now_terms`` passed to FOLPS.
    ncols
        Number of columns in the wiggle half of ``table_all``.
    kt
        FKPT output grid.
    fk_norm, f0, fk
        Normalized growth rate, low-k growth rate, and full growth-rate grid.
    kernel_constants
        Tuple ``(A, Ap, CFD3, CFD3p)`` when requested from the builder;
        otherwise ``None``.
    """

    table_w: tuple[Any, ...]
    table_now: tuple[Any, ...]
    table: tuple[Any, ...]
    table_now_terms: tuple[Any, ...]
    scalars: tuple[Any, ...]
    scalars_now: tuple[Any, ...]
    table_all: tuple[Any, ...]
    ncols: int
    kt: Any
    fk_norm: Any
    f0: Any
    fk: Any
    kernel_constants: Any = None


def split_folps_tables(table_w: Sequence[Any], table_now: Sequence[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Split raw FKPT tables into FOLPS-ready wiggle/no-wiggle tables.

    The current A_full=False table layout is:

    - index ``0``: ``k`` grid;
    - indices ``1:``: FOLPS columns and scalar tails.

    FOLPS ``interp_table`` expects the full raw table, including ``k`` at
    ``table[0]``.  Do not drop the first element.
    """
    table = tuple(table_w)
    table_now_terms = tuple(table_now)
    scalars = tuple(table_w[28:])
    scalars_now = tuple(table_now[28:])
    return table, table_now_terms, scalars, scalars_now


def make_table_state(
    table_w: Sequence[Any],
    table_now: Sequence[Any],
    kernel_constants: Any = None,
) -> FKPTTableState:
    """Build an :class:`FKPTTableState` from raw builder outputs."""
    table_w = tuple(table_w)
    table_now = tuple(table_now)
    table, table_now_terms, scalars, scalars_now = split_folps_tables(table_w, table_now)
    f0 = table_w[-1]
    fk_norm = table_w[2]
    fk = jnp.asarray(fk_norm) * jnp.asarray(f0)
    return FKPTTableState(
        table_w=table_w,
        table_now=table_now,
        table=table,
        table_now_terms=table_now_terms,
        scalars=scalars,
        scalars_now=scalars_now,
        table_all=table + table_now_terms,
        ncols=len(table),
        kt=table_w[0],
        fk_norm=fk_norm,
        f0=f0,
        fk=fk,
        kernel_constants=kernel_constants,
    )


def _call_table_builder(
    builder: Callable[..., Any],
    *,
    k: Any,
    pk: Any,
    pk_now: Any,
    return_kernel_constants: bool = True,
    **fkpt_params: Any,
) -> FKPTTableState:
    """Call a ``Kfuncs_to_tables*`` builder and normalize its return value."""
    result = builder(
        k=k,
        pk=pk,
        pk_now=pk_now,
        return_kernel_constants=return_kernel_constants,
        **fkpt_params,
    )
    if return_kernel_constants:
        table_w, table_now, kernel_constants = result
    else:
        table_w, table_now = result
        kernel_constants = None
    return make_table_state(table_w, table_now, kernel_constants=kernel_constants)


def poles_from_tables(
    state: FKPTTableState,
    *,
    jac: Any,
    kap: Any,
    muap: Any,
    pars: Any,
    mu: Any,
    wmu: Any,
    ells: Sequence[int] = (0, 2, 4),
    bias_scheme: str = "folps",
    IR_resummation: bool = True,
    damping: Any = None,
    A_full: bool = False,
    use_TNS_model: bool = False,
    to_poles: Any = None,
) -> Any:
    """Project an already-built FKPT table state to RSD multipoles.

    If ``to_poles`` is provided, use the caller's native multipole projector.
    This is the legacy downstream-wrapper behaviour and is the safest path when
    the caller owns the exact quadrature convention.  If ``to_poles`` is not provided, fall back to the standalone
    fkptjax projector through ``tables_to_poles``.
    """
    if to_poles is not None:
        pkmu = tables_to_pkmu(
            kap,
            muap,
            pars,
            state.table_all,
            ncols=state.ncols,
            bias_scheme=bias_scheme,
            IR_resummation=IR_resummation,
            damping=damping,
            A_full=A_full,
            use_TNS_model=use_TNS_model,
        )
        return to_poles(jac * pkmu)

    return tables_to_poles(
        jac,
        kap,
        muap,
        pars,
        state.table_all,
        mu,
        wmu,
        ncols=state.ncols,
        bias_scheme=bias_scheme,
        IR_resummation=IR_resummation,
        damping=damping,
        ells=tuple(ells),
        A_full=A_full,
        use_TNS_model=use_TNS_model,
    )

def _pipeline_poles(
    builder: Callable[..., Any],
    *,
    k: Any,
    pk: Any,
    pk_now: Any,
    jac: Any,
    kap: Any,
    muap: Any,
    pars: Any,
    mu: Any,
    wmu: Any,
    ells: Sequence[int] = (0, 2, 4),
    bias_scheme: str = "folps",
    IR_resummation: bool = True,
    damping: Any = None,
    A_full: bool = False,
    use_TNS_model: bool = False,
    return_kernel_constants: bool = True,
    to_poles: Any = None,
    **fkpt_params: Any,
) -> tuple[Any, FKPTTableState]:
    """Shared implementation for the public pipeline routes."""
    state = _call_table_builder(
        builder,
        k=k,
        pk=pk,
        pk_now=pk_now,
        return_kernel_constants=return_kernel_constants,
        **fkpt_params,
    )
    poles = poles_from_tables(
        state,
        jac=jac,
        kap=kap,
        muap=muap,
        pars=pars,
        mu=mu,
        wmu=wmu,
        ells=ells,
        bias_scheme=bias_scheme,
        IR_resummation=IR_resummation,
        damping=damping,
        A_full=A_full,
        use_TNS_model=use_TNS_model,
        to_poles=to_poles,
    )
    return poles, state


def full_method_poles(
    *,
    k: Any,
    pk: Any,
    pk_now: Any,
    jac: Any,
    kap: Any,
    muap: Any,
    pars: Any,
    mu: Any,
    wmu: Any,
    ells: Sequence[int] = (0, 2, 4),
    bias_scheme: str = "folps",
    IR_resummation: bool = True,
    damping: Any = None,
    A_full: bool = False,
    use_TNS_model: bool = False,
    return_kernel_constants: bool = True,
    to_poles: Any = None,
    **fkpt_params: Any,
) -> tuple[Any, FKPTTableState]:
    """Run the legacy/eager FKPT table builder and return RSD multipoles.

    ``fkpt_params`` are forwarded directly to :func:`Kfuncs_to_tables`; they must
    include the usual FKPT options such as ``z`` and ``Om``.
    """
    return _pipeline_poles(
        Kfuncs_to_tables,
        k=k,
        pk=pk,
        pk_now=pk_now,
        jac=jac,
        kap=kap,
        muap=muap,
        pars=pars,
        mu=mu,
        wmu=wmu,
        ells=ells,
        bias_scheme=bias_scheme,
        IR_resummation=IR_resummation,
        damping=damping,
        A_full=A_full,
        use_TNS_model=use_TNS_model,
        return_kernel_constants=return_kernel_constants,
        to_poles=to_poles,
        **fkpt_params,
    )


def binning_jax_poles(
    *,
    k: Any,
    pk: Any,
    pk_now: Any,
    jac: Any,
    kap: Any,
    muap: Any,
    pars: Any,
    mu: Any,
    wmu: Any,
    ells: Sequence[int] = (0, 2, 4),
    bias_scheme: str = "folps",
    IR_resummation: bool = True,
    damping: Any = None,
    A_full: bool = False,
    use_TNS_model: bool = False,
    return_kernel_constants: bool = True,
    to_poles: Any = None,
    static_ctx: Any = None,
    **fkpt_params: Any,
) -> tuple[Any, FKPTTableState]:
    """Run the fully JAX PHENOM/binning route and return RSD multipoles.

    This is the route to use for the jaxmapse/binning model.  For repeated calls,
    pass a ``static_ctx`` created by :func:`build_jax_static_ctx` so the kernel
    grids and BAO Bessel arrays are not rebuilt at every likelihood evaluation.
    """
    if static_ctx is not None:
        fkpt_params["static_ctx"] = static_ctx
    return _pipeline_poles(
        Kfuncs_to_tables_jax,
        k=k,
        pk=pk,
        pk_now=pk_now,
        jac=jac,
        kap=kap,
        muap=muap,
        pars=pars,
        mu=mu,
        wmu=wmu,
        ells=ells,
        bias_scheme=bias_scheme,
        IR_resummation=IR_resummation,
        damping=damping,
        A_full=A_full,
        use_TNS_model=use_TNS_model,
        return_kernel_constants=return_kernel_constants,
        to_poles=to_poles,
        **fkpt_params,
    )


def rescale_branch_poles(
    *,
    k: Any,
    pk: Any,
    pk_now: Any,
    jac: Any,
    kap: Any,
    muap: Any,
    pars: Any,
    mu: Any,
    wmu: Any,
    ells: Sequence[int] = (0, 2, 4),
    bias_scheme: str = "folps",
    IR_resummation: bool = True,
    damping: Any = None,
    A_full: bool = False,
    use_TNS_model: bool = False,
    return_kernel_constants: bool = True,
    to_poles: Any = None,
    static_ctx: Any = None,
    kernel_constants: Any = None,
    **fkpt_params: Any,
) -> tuple[Any, FKPTTableState]:
    """Run the JAX rescaling branch and return RSD multipoles.

    The input ``pk`` and ``pk_now`` should be the GR/LCDM linear spectra.  FKPT
    applies the MG growth rescaling internally through
    :func:`Kfuncs_to_tables_rescale_jax`.

    Parameters
    ----------
    static_ctx
        Optional result of :func:`build_jax_static_ctx`.  Reusing it is important
        for speed and for stable ``jit``/``vmap`` behaviour.
    kernel_constants
        Optional beyond-EdS constants object/dict/tuple.  This is required when
        ``beyond_eds=True`` for the rescaling branch.
    """
    if static_ctx is not None:
        fkpt_params["static_ctx"] = static_ctx
    if kernel_constants is not None:
        fkpt_params["kernel_constants"] = kernel_constants
    return _pipeline_poles(
        Kfuncs_to_tables_rescale_jax,
        k=k,
        pk=pk,
        pk_now=pk_now,
        jac=jac,
        kap=kap,
        muap=muap,
        pars=pars,
        mu=mu,
        wmu=wmu,
        ells=ells,
        bias_scheme=bias_scheme,
        IR_resummation=IR_resummation,
        damping=damping,
        A_full=A_full,
        use_TNS_model=use_TNS_model,
        return_kernel_constants=return_kernel_constants,
        to_poles=to_poles,
        **fkpt_params,
    )
