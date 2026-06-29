"""Pipeline de suppression de filigrane : extraction, masque, inpainting, réassemblage."""

from . import utils, extract, mask, inpaint, assemble  # noqa: F401

__all__ = ["utils", "extract", "mask", "inpaint", "assemble"]
