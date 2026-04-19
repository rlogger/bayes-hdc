# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rajdeep Singh

"""JAX version compatibility helpers."""

import dataclasses

import jax

try:
    _test_cls = type("_Test", (), {"__annotations__": {"x": int}})
    jax.tree_util.register_dataclass(_test_cls)
    del _test_cls

    def register_dataclass(cls: type) -> type:
        return jax.tree_util.register_dataclass(cls)

except TypeError:

    def register_dataclass(cls: type) -> type:  # type: ignore[misc]
        data_fields = [
            f.name for f in dataclasses.fields(cls) if not f.metadata.get("static", False)
        ]
        meta_fields = [f.name for f in dataclasses.fields(cls) if f.metadata.get("static", False)]
        return jax.tree_util.register_dataclass(cls, data_fields, meta_fields)
