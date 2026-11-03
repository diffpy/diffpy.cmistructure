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
"""Converters between structure representations."""

import numpy
from pyobjcryst.atom import Atom
from pyobjcryst.crystal import Crystal
from pyobjcryst.scatteringpower import ScatteringPowerAtom

from diffpy.structure.symmetryutilities import epsilon, equal_positions

__all__ = ["create_crystal_from_structure"]

_ANISOTROPIC_ADP_NAMES = ("B11", "B22", "B33", "B12", "B13", "B23")


def create_crystal_from_structure(structure, space_group):
    """Create a pyobjcryst Crystal from a diffpy.structure Structure.

    A Structure holds the full unit cell and does not record its space
    group, so the space group must be given. The atoms are reduced to the
    asymmetric unit of that space group, keeping the first atom of each set
    of symmetry-equivalent atoms. The lattice, elements, positions,
    occupancies and displacement factors are copied, with B = 8 pi^2 U.
    Each atom is named by its label, or, if it has none, by its element and
    a count, such as La1 or O2.

    The space group is read by ObjCryst, which takes Hermann-Mauguin
    symbols with or without spaces, such as "P b n m" or "Pbnm", and space
    group numbers, such as 62. A number gives the standard setting, so 62
    is "P n m a", not "P b n m". For a space group with two origin choices,
    ObjCryst uses origin choice 2 unless another is given after a colon,
    such as "F d -3 m:1". Give "P1" to keep every atom of the cell.

    Parameters
    ----------
    structure : diffpy.structure.Structure
        The structure to convert, holding the full unit cell.
    space_group : str or int
        The space group of the structure, as a Hermann-Mauguin symbol or a
        space group number.

    Returns
    -------
    pyobjcryst.crystal.Crystal
        The crystal holding the asymmetric unit of the structure.

    Raises
    ------
    ValueError
        If ObjCryst does not recognize `space_group`, or if the lattice or
        the atoms of `structure` do not fit it.
    """
    lattice = structure.lattice
    cell_parameters = [
        lattice.a,
        lattice.b,
        lattice.c,
        lattice.alpha,
        lattice.beta,
        lattice.gamma,
    ]
    try:
        crystal = Crystal(
            lattice.a,
            lattice.b,
            lattice.c,
            numpy.radians(lattice.alpha),
            numpy.radians(lattice.beta),
            numpy.radians(lattice.gamma),
            str(space_group),
        )
    except ValueError:
        raise ValueError(
            f"{space_group!r} is not a space group symbol or number. "
            "Please give a Hermann-Mauguin symbol, such as 'P b n m', or a "
            "number from 1 to 230."
        ) from None
    # ObjCryst silently changes a cell that doesn't fit the space group
    crystal_cell_parameters = [
        crystal.a,
        crystal.b,
        crystal.c,
        numpy.degrees(crystal.alpha),
        numpy.degrees(crystal.beta),
        numpy.degrees(crystal.gamma),
    ]
    if not numpy.allclose(crystal_cell_parameters, cell_parameters):
        raise ValueError(
            _get_not_fitting_message(crystal.GetSpaceGroup(), "lattice")
        )
    element_counts = {}
    for atom in _get_asymmetric_unit(structure, crystal.GetSpaceGroup()):
        element_counts[atom.element] = element_counts.get(atom.element, 0) + 1
        name = atom.label or f"{atom.element}{element_counts[atom.element]}"
        scattering_power = ScatteringPowerAtom(name, atom.element)
        scattering_power.SetBiso(atom.Bisoequiv)
        if atom.anisotropy:
            for adp_name in _ANISOTROPIC_ADP_NAMES:
                setattr(scattering_power, adp_name, getattr(atom, adp_name))
        x, y, z = atom.xyz
        crystal.AddScatteringPower(scattering_power)
        crystal.AddScatterer(
            Atom(x, y, z, name, scattering_power, atom.occupancy)
        )
    return crystal


def _get_asymmetric_unit(structure, space_group):
    # Keep the first atom of each set of symmetry-equivalent atoms, and
    # check that the structure holds every atom of each set
    remaining_atoms = list(structure)
    asymmetric_unit = []
    while remaining_atoms:
        atom = remaining_atoms.pop(0)
        asymmetric_unit.append(atom)
        equivalent_atoms = [atom]
        x, y, z = atom.xyz
        positions = space_group.GetAllSymmetrics(x, y, z, False, False, True)
        for position in positions:
            matched_index = _find_atom_index(
                equivalent_atoms, atom.element, position
            )
            if matched_index is not None:
                continue
            index = _find_atom_index(remaining_atoms, atom.element, position)
            if index is None:
                raise ValueError(
                    _get_not_fitting_message(space_group, "atoms")
                )
            equivalent_atoms.append(remaining_atoms.pop(index))
    return asymmetric_unit


def _find_atom_index(atoms, element, xyz):
    for index, atom in enumerate(atoms):
        if atom.element == element and equal_positions(atom.xyz, xyz, epsilon):
            return index
    return None


def _get_not_fitting_message(space_group, structure_part):
    return (
        f"Space group {space_group.GetName()} does not fit the "
        f"{structure_part} of the structure. Please give the full unit "
        "cell of the structure and its space group, with the setting and "
        "origin choice, or 'P1' to keep every atom."
    )


# End of file
