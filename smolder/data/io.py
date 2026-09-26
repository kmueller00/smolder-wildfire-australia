"""Locating and opening the SMOLDER data cubes.

Relative cube paths (e.g. ``cube_daily_smgrid_2020.zarr``) are resolved
against the current directory first, then against ``$SMOLDER_DATA``. The
channel-statistics JSON ships inside the package and is found there if it is
not present in the data directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import zarr
from zarr.errors import GroupNotFoundError

PACKAGE_DATA = Path(__file__).resolve().parent
CHANNEL_STATS = PACKAGE_DATA / "channel_stats_2015_2018.json"


def data_dir() -> Path:
    return Path(os.environ.get("SMOLDER_DATA", ".")).expanduser()


def resolve(path: str | os.PathLike) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return data_dir() / p


def resolve_stats(path: str | os.PathLike) -> Path:
    p = resolve(path)
    if not p.exists() and (PACKAGE_DATA / Path(path).name).exists():
        return PACKAGE_DATA / Path(path).name
    return p


def open_zarr_root(path: str | os.PathLike):
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Zarr store not found: '{path}' (looked in cwd and SMOLDER_DATA='{data_dir()}')")
    try:
        return zarr.open_group(str(p), mode="r")
    except GroupNotFoundError:
        pass
    root = zarr.open(str(p), mode="r")
    if isinstance(root, zarr.hierarchy.Group):
        return root
    raise GroupNotFoundError(f"Zarr root is not a group at path '{p}'.")


def daily_cube(year: int | str) -> str:
    """Name of the daily cube for `year`: the full 7-channel build
    (cube_daily_smgrid_YYYY.zarr) if present, else the NDVI-free Zenodo
    archive (cube_YYYY_zenodo.zarr). Both are read identically."""
    full = f"cube_daily_smgrid_{year}.zarr"
    archived = f"cube_{year}_zenodo.zarr"
    if not resolve(full).exists() and resolve(archived).exists():
        return archived
    return full
