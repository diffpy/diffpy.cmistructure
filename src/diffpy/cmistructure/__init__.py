#!/usr/bin/env python
##############################################################################
#
# (c) 2026 Contributors to diffpy.cmistructure.
# All rights reserved.
#
# File coded by: Members of the diffpy community.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.cmistructure/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Modules and classes that adapt structure representations to the
ParameterSet interface and automatic structure constraint generation
from space group information."""

from diffpy.cmistructure.sgconstraints import constrain_as_space_group

# package version
from diffpy.cmistructure.version import __version__  # noqa

__all__ = ["constrain_as_space_group", "structure_to_parameter_set"]


def structure_to_parameter_set(name, structure):
    """Create a ParameterSet adapted to a structure object.

    The adapter is chosen from the type of `structure`. Supported types are
    diffpy.structure.Structure, pyobjcryst.crystal.Crystal,
    pyobjcryst.molecule.Molecule and cctbx.crystal.special_position_settings.

    Parameters
    ----------
    name : str
        The name to give the structure.
    structure : object
        The structure object to adapt.

    Returns
    -------
    BaseStructureParSet
        The ParameterSet adapting `structure`.

    Raises
    ------
    TypeError
        If `structure` is not one of the supported structure types.
    """
    from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet

    if DiffpyStructureParSet.can_adapt(structure):
        return DiffpyStructureParSet(name, structure)

    from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet

    if ObjCrystCrystalParSet.can_adapt(structure):
        return ObjCrystCrystalParSet(name, structure)

    from diffpy.cmistructure.objcrystparset import ObjCrystMoleculeParSet

    if ObjCrystMoleculeParSet.can_adapt(structure):
        return ObjCrystMoleculeParSet(name, structure)

    from diffpy.cmistructure.cctbxparset import CCTBXCrystalParSet

    if CCTBXCrystalParSet.can_adapt(structure):
        return CCTBXCrystalParSet(name, structure)

    raise TypeError("Unadaptable structure format")


# silence the pyflakes syntax checker
assert __version__ or True

# End of file
