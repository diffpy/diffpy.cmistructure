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
"""Tests for diffpy.cmistructure package."""

import pickle
import unittest

import numpy as np

from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet


def testDiffpyStructureParSet():
    """Test the structure conversion."""
    from diffpy.structure import Atom, Lattice, Structure

    a1 = Atom("Cu", xyz=np.array([0.0, 0.1, 0.2]), Uisoequiv=0.003)
    a2 = Atom("Ag", xyz=np.array([0.3, 0.4, 0.5]), Uisoequiv=0.002)
    lattice = Lattice(2.5, 2.5, 2.5, 90, 90, 90)

    dsstru = Structure([a1, a2], lattice)
    # Structure makes copies
    a1 = dsstru[0]
    a2 = dsstru[1]

    s = DiffpyStructureParSet("CuAg", dsstru)

    actual_name = s.name
    expected_name = "CuAg"
    assert actual_name == expected_name

    def _testAtoms():
        # Check the atoms thoroughly
        actual_atoms = {
            "Cu0": {
                "element": s.Cu0.element,
                "Uiso": s.Cu0.Uiso.get_value(),
                "Biso": s.Cu0.Biso.get_value(),
                "xyz": [
                    s.Cu0.x.get_value(),
                    s.Cu0.y.get_value(),
                    s.Cu0.z.get_value(),
                ],
            },
            "Ag0": {
                "element": s.Ag0.element,
                "Uiso": s.Ag0.Uiso.get_value(),
                "Biso": s.Ag0.Biso.get_value(),
            },
        }
        expected_atoms = {
            "Cu0": {
                "element": a1.element,
                "Uiso": a1.Uisoequiv,
                "Biso": a1.Bisoequiv,
                "xyz": [a1.xyz[0], a1.xyz[1], a1.xyz[2]],
            },
            "Ag0": {
                "element": a2.element,
                "Uiso": a2.Uisoequiv,
                "Biso": a2.Bisoequiv,
            },
        }
        assert actual_atoms == expected_atoms

        # The Uij and Uji (Bij and Bji) Parameters both read the
        # structure's Uij (Bij).
        actual_anisotropic = {}
        expected_anisotropic = {}
        for i in range(1, 4):
            for j in range(i, 4):
                for prefix in "UB":
                    ij = "%s%i%i" % (prefix, i, j)
                    ji = "%s%i%i" % (prefix, j, i)
                    actual_anisotropic[ij] = getattr(s.Cu0, ij).get_value()
                    actual_anisotropic[ji] = getattr(s.Cu0, ji).get_value()
                    expected_anisotropic[ij] = getattr(a1, ij)
                    expected_anisotropic[ji] = getattr(a1, ij)
        assert actual_anisotropic == expected_anisotropic
        return

    def _testLattice():
        # Test the lattice
        lattice_names = ["a", "b", "c", "alpha", "beta", "gamma"]
        actual_lattice = [
            getattr(s.lattice, name).get_value() for name in lattice_names
        ]
        expected_lattice = [
            getattr(dsstru.lattice, name) for name in lattice_names
        ]
        assert actual_lattice == expected_lattice

    # C1: The ParameterSet has just been created from the structure.
    # Expected: The Parameters match the atoms and lattice.
    _testAtoms()
    _testLattice()

    # C2: The diffpy Structure is changed directly.
    # Expected: The Parameters follow the changes.
    a1.xyz[1] = 0.123
    a1.U11 = 0.321
    a1.B32 = 0.111
    dsstru.lattice.set_latt_parms(a=3.0, gamma=121)
    _testAtoms()
    _testLattice()

    # C3: The Parameters of the DiffpyStructureParSet are changed.
    # Expected: The structure follows the changes, so the distance
    # between the atoms changes.
    s.Cu0.x.set_value(0.456)
    s.Cu0.U22.set_value(0.441)
    s.Cu0.B13.set_value(0.550)
    d = dsstru.lattice.dist(a1.xyz, a2.xyz)
    s.lattice.b.set_value(4.6)
    s.lattice.alpha.set_value(91.3)
    _testAtoms()
    _testLattice()
    actual_distance_changed = d != dsstru.lattice.dist(a1.xyz, a2.xyz)
    expected_distance_changed = True
    assert actual_distance_changed == expected_distance_changed
    return


def test___repr__():
    """Test representation of DiffpyStructureParSet objects."""
    from diffpy.structure import Atom, Lattice, Structure

    lat = Lattice(3, 3, 2, 90, 90, 90)
    atom = Atom("C", [0, 0.2, 0.5])
    structure = Structure([atom], lattice=lat)
    dsps = DiffpyStructureParSet("dsps", structure)
    # C1: The structure, lattice and atom ParameterSets are printed.
    # Expected: Each repr matches the repr of the adapted object.
    actual_reprs = [repr(dsps), repr(dsps.lattice), repr(dsps.atoms[0])]
    expected_reprs = [repr(structure), repr(lat), repr(atom)]
    assert actual_reprs == expected_reprs
    return


def test_pickling():
    """Test pickling of DiffpyStructureParSet."""
    from diffpy.structure import Atom, Structure

    structure = Structure([Atom("C", [0, 0.2, 0.5])])
    dsps = DiffpyStructureParSet("dsps", structure)
    data = pickle.dumps(dsps)
    dsps2 = pickle.loads(data)
    # C1: A DiffpyStructureParSet is pickled and unpickled.
    # Expected: The copy keeps its single atom and the atom's position.
    actual_atom_count = len(dsps2.atoms)
    expected_atom_count = 1
    assert actual_atom_count == expected_atom_count
    actual_y = dsps2.atoms[0].y.value
    expected_y = 0.2
    assert actual_y == expected_y
    return


# End of class TestParameterAdapter

if __name__ == "__main__":
    unittest.main()
