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
"""Tests for diffpy.cmistructure package."""

import re

import numpy as np
import pytest

from diffpy.cmistructure.converters import create_crystal_from_structure
from diffpy.structure import Atom, Lattice, Structure


@pytest.mark.parametrize(
    "space_group",
    [
        # Test converting the full LaMnO3 cell to its asymmetric unit
        # C1: A user gives the space group with spaces, "P b n m".
        # Expected: The crystal holds the four atoms of the asymmetric unit.
        "P b n m",
        # C2: A user gives the space group without spaces, "Pbnm".
        # Expected: The crystal is the same as in C1.
        "Pbnm",
    ],
)
def test_create_crystal_from_structure(lamno3_p1_structure, space_group):
    crystal = create_crystal_from_structure(lamno3_p1_structure, space_group)
    # The lattice and space group are those of the structure
    actual_lattice = [
        crystal.a,
        crystal.b,
        crystal.c,
        np.degrees(crystal.alpha),
        np.degrees(crystal.beta),
        np.degrees(crystal.gamma),
    ]
    expected_lattice = [5.486341, 5.619215, 7.628206, 90.0, 90.0, 90.0]
    assert np.allclose(actual_lattice, expected_lattice)
    actual_space_group = crystal.GetSpaceGroup().GetName()
    expected_space_group = "P b n m"
    assert actual_space_group == expected_space_group
    # The first atom of each set of symmetry-equivalent atoms is kept
    atoms = [crystal.GetScatterer(i) for i in range(crystal.GetNbScatterer())]
    actual_names = [atom.GetName() for atom in atoms]
    expected_names = ["La1", "Mn1", "O1", "O2"]
    assert actual_names == expected_names
    actual_elements = [atom.GetScatteringPower().GetSymbol() for atom in atoms]
    expected_elements = ["La", "Mn", "O", "O"]
    assert actual_elements == expected_elements
    actual_xyz = [[atom.X, atom.Y, atom.Z] for atom in atoms]
    expected_xyz = [
        [0.99609631, 0.03214940, 0.25],
        [0.0, 0.5, 0.0],
        [0.05957463, 0.49616399, 0.25],
        [0.72005206, 0.28938726, 0.03111255],
    ]
    assert np.allclose(actual_xyz, expected_xyz)
    actual_occupancies = [atom.Occupancy for atom in atoms]
    expected_occupancies = [1.0, 1.0, 1.0, 1.0]
    assert np.allclose(actual_occupancies, expected_occupancies)
    actual_biso = [atom.GetScatteringPower().GetBiso() for atom in atoms]
    expected_uiso = np.array([0.00253993, 0.00065337, 0.00082010, 0.00512371])
    expected_biso = 8 * np.pi**2 * expected_uiso
    assert np.allclose(actual_biso, expected_biso)


def test_create_crystal_from_structure_in_p1():
    # Test converting a structure to a crystal in P1
    # C1: A user keeps their whole cell in P1. The copper atom has a label,
    # a partial occupancy and anisotropic displacements, and the oxygen
    # atom has none of these.
    # Expected: Both atoms are kept, the copper atom under its label and the
    # oxygen atom as O1, each with its occupancy and displacement factors.
    copper = Atom(
        "Cu",
        xyz=[0.1, 0.2, 0.3],
        label="Cu_surface",
        occupancy=0.8,
        anisotropy=True,
        U=[[0.004, 0.001, 0.0], [0.001, 0.006, 0.0], [0.0, 0.0, 0.008]],
    )
    oxygen = Atom("O", xyz=[0.4, 0.5, 0.6], Uisoequiv=0.005)
    lattice = Lattice(3.0, 4.0, 5.0, 90, 95, 90)
    structure = Structure([copper, oxygen], lattice=lattice)
    crystal = create_crystal_from_structure(structure, "P1")
    actual_lattice = [
        crystal.a,
        crystal.b,
        crystal.c,
        np.degrees(crystal.alpha),
        np.degrees(crystal.beta),
        np.degrees(crystal.gamma),
    ]
    expected_lattice = [3.0, 4.0, 5.0, 90.0, 95.0, 90.0]
    assert np.allclose(actual_lattice, expected_lattice)
    atoms = [crystal.GetScatterer(i) for i in range(crystal.GetNbScatterer())]
    actual_names = [atom.GetName() for atom in atoms]
    expected_names = ["Cu_surface", "O1"]
    assert actual_names == expected_names
    actual_xyz = [[atom.X, atom.Y, atom.Z] for atom in atoms]
    expected_xyz = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert np.allclose(actual_xyz, expected_xyz)
    actual_occupancies = [atom.Occupancy for atom in atoms]
    expected_occupancies = [0.8, 1.0]
    assert np.allclose(actual_occupancies, expected_occupancies)
    # The copper atom is anisotropic and the oxygen atom isotropic
    copper_power = atoms[0].GetScatteringPower()
    oxygen_power = atoms[1].GetScatteringPower()
    actual_isotropic = [copper_power.IsIsotropic(), oxygen_power.IsIsotropic()]
    expected_isotropic = [False, True]
    assert actual_isotropic == expected_isotropic
    actual_copper_bij = [
        copper_power.B11,
        copper_power.B22,
        copper_power.B33,
        copper_power.B12,
        copper_power.B13,
        copper_power.B23,
    ]
    expected_copper_uij = np.array([0.004, 0.006, 0.008, 0.001, 0.0, 0.0])
    expected_copper_bij = 8 * np.pi**2 * expected_copper_uij
    assert np.allclose(actual_copper_bij, expected_copper_bij)
    actual_oxygen_biso = oxygen_power.GetBiso()
    expected_oxygen_biso = 8 * np.pi**2 * 0.005
    assert np.isclose(actual_oxygen_biso, expected_oxygen_biso)


@pytest.mark.parametrize(
    "structure_fixture_name, space_group, expected_space_group, expected_xyz",
    [
        # Test converting silicon in F d -3 m, which has two origin choices
        # C1: A user's silicon cell has an atom at the origin, and they ask
        # for origin choice 1, "F d -3 m:1".
        # Expected: The crystal holds one silicon atom, at the origin.
        (
            "silicon_origin_1_structure",
            "F d -3 m:1",
            "F d -3 m:1",
            [0.0, 0.0, 0.0],
        ),
        # C2: A user's silicon cell has its origin midway between two atoms,
        # and they give "Fd-3m" without an origin choice.
        # Expected: ObjCryst's default, origin choice 2, is used, and the
        # crystal holds one silicon atom, at (1/8, 1/8, 1/8).
        (
            "silicon_origin_2_structure",
            "Fd-3m",
            "F d -3 m:2",
            [0.125, 0.125, 0.125],
        ),
    ],
)
def test_create_crystal_from_structure_with_origin_choice(
    request,
    structure_fixture_name,
    space_group,
    expected_space_group,
    expected_xyz,
):
    silicon_structure = request.getfixturevalue(structure_fixture_name)
    crystal = create_crystal_from_structure(silicon_structure, space_group)
    actual_space_group = crystal.GetSpaceGroup().GetName()
    assert actual_space_group == expected_space_group
    actual_atom_count = crystal.GetNbScatterer()
    expected_atom_count = 1
    assert actual_atom_count == expected_atom_count
    atom = crystal.GetScatterer(0)
    actual_xyz = [atom.X, atom.Y, atom.Z]
    assert np.allclose(actual_xyz, expected_xyz)


@pytest.mark.parametrize(
    "structure_fixture_name, space_group, objcryst_space_group, "
    "structure_part",
    [
        # Test giving a space group that does not fit the structure
        # C1: A user gives a cubic space group for the orthorhombic LaMnO3
        # cell.
        # Expected: A ValueError says the space group doesn't fit the
        # lattice.
        ("lamno3_p1_structure", "Pm-3m", "P m -3 m", "lattice"),
        # C2: A user refined LaMnO3 in Pbnm but gives Pnma, another setting
        # of the same space group.
        # Expected: A ValueError says the space group doesn't fit the atoms.
        ("lamno3_p1_structure", "Pnma", "P n m a", "atoms"),
        # C3: A user gives Fd-3m for a silicon cell with an atom at the
        # origin, which needs origin choice 1, not ObjCryst's default of 2.
        # Expected: A ValueError says the space group doesn't fit the atoms,
        # naming origin choice 2.
        ("silicon_origin_1_structure", "Fd-3m", "F d -3 m:2", "atoms"),
        # C4: A user gives only the atoms of the LaMnO3 asymmetric unit,
        # copied from a paper, instead of the full cell.
        # Expected: A ValueError says the space group doesn't fit the atoms.
        ("lamno3_asymmetric_unit_structure", "Pbnm", "P b n m", "atoms"),
    ],
)
def test_create_crystal_from_structure_with_space_group_not_fitting(
    request,
    structure_fixture_name,
    space_group,
    objcryst_space_group,
    structure_part,
):
    structure = request.getfixturevalue(structure_fixture_name)
    expected_msg = (
        f"Space group {objcryst_space_group} does not fit the "
        f"{structure_part} of the structure. Please give the full unit "
        "cell of the structure and its space group, with the setting and "
        "origin choice, or 'P1' to keep every atom."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        create_crystal_from_structure(structure, space_group)


def test_create_crystal_from_structure_with_unknown_space_group(
    lamno3_p1_structure,
):
    # Test giving a space group symbol that ObjCryst does not recognize
    # C1: A user mistypes the space group as "Pbnm2".
    # Expected: A ValueError names the symbol and says how to give a space
    # group.
    expected_msg = (
        "'Pbnm2' is not a space group symbol. Please give a "
        "Hermann-Mauguin symbol, such as 'P b n m' or 'Pbnm'."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        create_crystal_from_structure(lamno3_p1_structure, "Pbnm2")
