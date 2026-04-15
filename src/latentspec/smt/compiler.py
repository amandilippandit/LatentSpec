"""Compile invariant params into Z3 expressions.

The trace model:

    n              : Int   — length of the trace
    tool(i)        : Int   — tool identifier at step i (0 if non-tool step)
    latency(i)     : Int   — latency_ms at step i (0 if not a tool call)
    success(i)     : Bool  — true if step i is a successful tool call
    keyword(k)     : Bool  — true if user_input contains keyword k
    has_step_type(t,i) : Bool — true if step i has step_type t

Tool names are interned to small integers via a per-formula symbol table
so Z3 stays inside its decidable fragment (LIA + uninterpreted functions).

Each compiler returns a `Z3Compilation` containing:
  - `formula`   — the Z3 expression encoding the invariant ("forall traces
                  consistent with the symbol table, the rule holds")
  - `symbols`   — the trace-model variables for the verifier to instantiate
  - `signature` — a stable hash so the streaming detector can dedup compiled
                  formulae across processes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import z3

from latentspec.models.invariant import InvariantType


class Z3CompilerError(ValueError):
    """Raised when params can't be compiled (missing keys, unsupported metric, etc.)."""


@dataclass(frozen=True)
class Z3Symbols:
    """Reusable Z3 declarations the verifier instantiates per trace."""

    n: z3.ArithRef
    tool: z3.FuncDeclRef
    latency: z3.FuncDeclRef
    success: z3.FuncDeclRef
    has_step_type: z3.FuncDeclRef
    keyword: z3.FuncDeclRef
    tool_ids: dict[str, int]
    keyword_ids: dict[str, int]
    step_type_ids: dict[str, int]


def _new_symbols() -> Z3Symbols:
    return Z3Symbols(
        n=z3.Int("n"),
        tool=z3.Function("tool", z3.IntSort(), z3.IntSort()),
        latency=z3.Function("latency", z3.IntSort(), z3.IntSort()),
        success=z3.Function("success", z3.IntSort(), z3.BoolSort()),
        has_step_type=z3.Function(
            "has_step_type", z3.IntSort(), z3.IntSort(), z3.BoolSort()
        ),
        keyword=z3.Function("keyword", z3.IntSort(), z3.BoolSort()),
        tool_ids={},
        keyword_ids={},
        step_type_ids={
            "user_input": 0,
            "tool_call": 1,
            "agent_response": 2,
            "agent_thought": 3,
            "system": 4,
        },
    )


def _intern_tool(symbols: Z3Symbols, name: str) -> int:
    if name not in symbols.tool_ids:
        symbols.tool_ids[name] = len(symbols.tool_ids) + 1  # 0 reserved for "no tool"
    return symbols.tool_ids[name]


def _intern_keyword(symbols: Z3Symbols, kw: str) -> int:
    if kw not in symbols.keyword_ids:
        symbols.keyword_ids[kw] = len(symbols.keyword_ids) + 1
    return symbols.keyword_ids[kw]


@dataclass
class Z3Compilation:
    """Output of compiling one invariant — used by both the verifier and the
    enterprise certificate generator."""

    invariant_type: InvariantType
    params: dict[str, Any]
    symbols: Z3Symbols
    formula: z3.BoolRef
    description: str
    signature: str = field(default="")

    def __post_init__(self) -> None:
        if not self.signature:
            payload = json.dumps(
                {"type": self.invariant_type.value, "params": self.params},
                sort_keys=True,
                default=str,
            )
            self.signature = hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------- Per-type compilers --------------------------------------


def _compile_ordering(params: dict[str, Any]) -> Z3Compilation:
    tool_a = params.get("tool_a")
    tool_b = params.get("tool_b")
    if not tool_a or not tool_b:
        raise Z3CompilerError("ordering needs tool_a and tool_b")

    s = _new_symbols()
    a_id = _intern_tool(s, tool_a)
    b_id = _intern_tool(s, tool_b)
    i, j = z3.Ints("i j")

    # forall i in [0, n). tool(i) = B -> exists j in [0, i). tool(j) = A
    formula = z3.ForAll(
        [i],
        z3.Implies(
            z3.And(0 <= i, i < s.n, s.tool(i) == b_id),
            z3.Exists(
                [j],
                z3.And(0 <= j, j < i, s.tool(j) == a_id),
            ),
        ),
    )
    return Z3Compilation(
        invariant_type=InvariantType.ORDERING,
        params=params,
        symbols=s,
        formula=formula,
        description=f"`{tool_a}` must precede `{tool_b}`",
    )


def _compile_conditional(params: dict[str, Any]) -> Z3Compilation:
    keyword = params.get("keyword")
    tool = params.get("tool")
    if not keyword or not tool:
        raise Z3CompilerError("conditional needs keyword and tool")

    s = _new_symbols()
    kw_id = _intern_keyword(s, keyword)
    t_id = _intern_tool(s, tool)
    i = z3.Int("i")

    # keyword(K) -> exists i. tool(i) = T
    formula = z3.Implies(
        s.keyword(kw_id),
        z3.Exists(
            [i], z3.And(0 <= i, i < s.n, s.tool(i) == t_id)
        ),
    )
    return Z3Compilation(
        invariant_type=InvariantType.CONDITIONAL,
        params=params,
        symbols=s,
        formula=formula,
        description=f"keyword '{keyword}' implies tool `{tool}`",
    )


def _compile_negative(params: dict[str, Any]) -> Z3Compilation:
    patterns = params.get("forbidden_patterns") or []
    if not patterns:
        raise Z3CompilerError("negative needs forbidden_patterns")

    s = _new_symbols()
    forbidden_ids = [_intern_tool(s, p) for p in patterns]
    i = z3.Int("i")

    # forall i in [0, n). tool(i) not in forbidden_ids
    formula = z3.ForAll(
        [i],
        z3.Implies(
            z3.And(0 <= i, i < s.n),
            z3.And(*(s.tool(i) != fid for fid in forbidden_ids)),
        ),
    )
    return Z3Compilation(
        invariant_type=InvariantType.NEGATIVE,
        params=params,
        symbols=s,
        formula=formula,
        description=f"agent never invokes any of {patterns}",
    )
