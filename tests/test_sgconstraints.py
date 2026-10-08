#!/usr/bin/env python
##############################################################################
#
# (c) 2010 The Trustees of Columbia University in the City of New York.
# (c) 2026 Contributors to diffpy.cmistructure.
# All rights reserved.
#
# File coded by: Pavol Juhas and members of the diffpy community.
#
# Originally developed in diffpy.srfit by the DANSE Diffraction group and
# Simon J. L. Billinge.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.cmistructure/graphs/contributors
#
# See LICENSE.rst and LICENSE_DANSE.rst for license information.
#
##############################################################################
"""Tests space group constraints."""

import unittest

import numpy
import pytest

# ----------------------------------------------------------------------------


def test_ObjCryst_constrain_space_group():
    """Make sure that all Parameters are constrained properly.

    This tests constrainSpaceGroup from
    diffpy.cmistructure.sgconstraints, which is performed automatically
    when an ObjCrystCrystalParSet is created.
    """
    from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet

    pi = numpy.pi

    occryst = makeLaMnO3()
    structure = ObjCrystCrystalParSet(occryst.GetName(), occryst)
    # Make sure we actually create the constraints
    structure._constrain_space_group()
    # Make the space group parameters individually
    structure.space_group_parameters.lattice_parameters
    structure.space_group_parameters.xyz_parameters
    structure.space_group_parameters.adp_parameters

    # C1: The orthorhombic lattice of LaMnO3 in P b n m.
    # Expected: The angles are fixed at pi / 2, the lengths are free, and
    # no constraint equations are needed.
    lattice = structure.get_lattice()
    lattice_names = ["a", "b", "c", "alpha", "beta", "gamma"]
    actual_lattice_const = {
        name: getattr(lattice, name).const for name in lattice_names
    }
    expected_lattice_const = {
        "a": False,
        "b": False,
        "c": False,
        "alpha": True,
        "beta": True,
        "gamma": True,
    }
    assert actual_lattice_const == expected_lattice_const
    actual_angles = [
        lattice.alpha.get_value(),
        lattice.beta.get_value(),
        lattice.gamma.get_value(),
    ]
    expected_angles = [pi / 2, pi / 2, pi / 2]
    assert actual_angles == expected_angles
    actual_lattice_constraint_count = len(lattice._constraints)
    expected_lattice_constraint_count = 0
    assert actual_lattice_constraint_count == expected_lattice_constraint_count

    # C2: The scatterers of LaMnO3 on their P b n m sites.
    # Expected: Coordinates on special positions are fixed, the rest are
    # free, and no constraint equations are needed.
    scatterers = structure.get_scatterers()
    la, mn, o1, o2 = scatterers
    actual_xyz_const = {
        "La1": [la.x.const, la.y.const, la.z.const],
        "Mn1": [mn.x.const, mn.y.const, mn.z.const],
        "O1": [o1.x.const, o1.y.const, o1.z.const],
        "O2": [o2.x.const, o2.y.const, o2.z.const],
    }
    expected_xyz_const = {
        "La1": [False, False, True],
        "Mn1": [True, True, True],
        "O1": [False, False, True],
        "O2": [False, False, False],
    }
    assert actual_xyz_const == expected_xyz_const
    actual_constraint_counts = [len(s._constraints) for s in scatterers]
    expected_constraint_counts = [0, 0, 0, 0]
    assert actual_constraint_counts == expected_constraint_counts

    # C3: Fixed coordinates are constrained or made into variables.
    # Expected: A ValueError is raised.
    with pytest.raises(ValueError):
        mn.add_constraint(mn.x, "y")

    with pytest.raises(ValueError):
        mn.add_constraint(mn.y, "z")

    with pytest.raises(ValueError):
        mn.add_constraint(mn.z, "x")

    # Nor can we make them into variables
    from diffpy.srfit.fitbase.fitrecipe import FitRecipe

    f = FitRecipe()
    with pytest.raises(ValueError):
        f.add_variable(mn.x)

    return


def test_DiffPy_constrain_as_space_group(datafile):
    """Test the constrain_as_space_group function."""
    from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet
    from diffpy.cmistructure.sgconstraints import constrain_as_space_group

    structure = makeLaMnO3_P1(datafile)
    parameter_set = DiffpyStructureParSet("LaMnO3", structure)

    space_group_parameters = constrain_as_space_group(
        parameter_set,
        "P b n m",
        scatterers=parameter_set.get_scatterers()[::2],
        constrainadps=True,
    )

    # C1: The space group Parameters are created.
    # Expected: Every Parameter exists and has a value.
    actual_unset_parameters = [
        parameter
        for parameter in space_group_parameters
        if parameter is None or parameter.get_value() is None
    ]
    expected_unset_parameters = []
    assert actual_unset_parameters == expected_unset_parameters

    # C2: Scatterers that were not passed to constrain_as_space_group.
    # Expected: Their positions and ADPs are free and unconstrained.
    unconstrained = parameter_set.get_scatterers()[1::2]
    actual_free_const = {
        scatterer.name: [
            scatterer.x.const,
            scatterer.y.const,
            scatterer.z.const,
            scatterer.U11.const,
            scatterer.U22.const,
            scatterer.U33.const,
            scatterer.U12.const,
            scatterer.U13.const,
            scatterer.U23.const,
        ]
        for scatterer in unconstrained
    }
    expected_free_const = {
        scatterer.name: [False] * 9 for scatterer in unconstrained
    }
    assert actual_free_const == expected_free_const
    actual_free_constraint_counts = {
        scatterer.name: len(scatterer._constraints)
        for scatterer in unconstrained
    }
    expected_free_constraint_counts = {
        scatterer.name: 0 for scatterer in unconstrained
    }
    assert actual_free_constraint_counts == expected_free_constraint_counts

    proxied = [p.par for p in space_group_parameters]

    def _consttest(parameter):
        return parameter.const

    def _constrainedtest(parameter):
        return parameter.constrained

    def _proxytest(parameter):
        return parameter in proxied

    def _alltests(parameter):
        return (
            _consttest(parameter)
            or _constrainedtest(parameter)
            or _proxytest(parameter)
        )

    # C3: Scatterers that were passed to constrain_as_space_group.
    # Expected: At least one position and one ADP Parameter of each is
    # fixed, constrained or proxied by a space group Parameter.
    constrained = parameter_set.get_scatterers()[::2]
    actual_restricted = {
        scatterer.name: [
            any(
                _alltests(parameter)
                for parameter in [scatterer.x, scatterer.y, scatterer.z]
            ),
            any(
                _alltests(parameter)
                for parameter in [
                    scatterer.U11,
                    scatterer.U22,
                    scatterer.U33,
                    scatterer.U12,
                    scatterer.U13,
                    scatterer.U23,
                ]
            ),
        ]
        for scatterer in constrained
    }
    expected_restricted = {
        scatterer.name: [True, True] for scatterer in constrained
    }
    assert actual_restricted == expected_restricted

    return


def test_constrain_as_space_group_args(datafile):
    """Test the arguments processing of constrain_as_space_group
    function."""
    from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet
    from diffpy.cmistructure.sgconstraints import constrain_as_space_group
    from diffpy.structure.spacegroups import get_space_group

    # C1: The space group is given as a symbol or as a SpaceGroup object.
    # Expected: Both create the same space group Parameters.
    structure = makeLaMnO3_P1(datafile)
    parameter_set = DiffpyStructureParSet("LaMnO3", structure)
    symbol_parameters = constrain_as_space_group(parameter_set, "P b n m")
    space_group = get_space_group("P b n m")
    object_parameter_set = DiffpyStructureParSet(
        "LMO", makeLaMnO3_P1(datafile)
    )
    object_parameters = constrain_as_space_group(
        object_parameter_set, space_group
    )
    list(symbol_parameters)
    list(object_parameters)
    actual_names = symbol_parameters.names
    expected_names = object_parameters.names
    assert actual_names == expected_names
    return


def makeLaMnO3_P1(datafile):
    from diffpy.structure import Structure

    structure = Structure()
    structure.read(datafile("LaMnO3.stru"))
    return structure


def makeLaMnO3():
    from pyobjcryst.atom import Atom
    from pyobjcryst.crystal import Crystal
    from pyobjcryst.scatteringpower import ScatteringPowerAtom

    pi = numpy.pi
    # It appears that ObjCryst only supports standard symbols
    crystal = Crystal(5.486341, 5.619215, 7.628206, "P b n m")
    crystal.SetName("LaMnO3")
    # La1
    sp = ScatteringPowerAtom("La1", "La")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.996096, 0.0321494, 0.25, "La1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # Mn1
    sp = ScatteringPowerAtom("Mn1", "Mn")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0, 0.5, 0, "Mn1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # O1
    sp = ScatteringPowerAtom("O1", "O")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.0595746, 0.496164, 0.25, "O1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # O2
    sp = ScatteringPowerAtom("O2", "O")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.720052, 0.289387, 0.0311126, "O2", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)

    return crystal


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
