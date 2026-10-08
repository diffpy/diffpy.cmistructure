#!/usr/bin/env python
##############################################################################
#
# (c) 2009 The Trustees of Columbia University in the City of New York.
# (c) 2026 Contributors to diffpy.cmistructure.
# All rights reserved.
#
# File coded by: Chris Farrow and members of the diffpy community.
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
"""Adapters for interfacing a diffpy.structure.Structure with SrFit.

A diffpy.structure.Structure object is meant to be passed to a
DiffpyStructureParSet object from this module, which can then be used as a
ParameterSet. (It has other methods for interfacing with SrReal calculator
adapters.) Any change to the lattice or existing atoms will be registered with
the Structure. Changes in the number of atoms will not be recognized.  Thus,
the diffpy.structure.Structure object should be fully configured before passing
it to DiffpyStructureParSet.

The following classes are adapted:

- `DiffpyStructureParSet`: adapter for `diffpy.structure.Structure`.
- `DiffpyLatticeParSet`: adapter for `diffpy.structure.Lattice`.
- `DiffpyAtomParSet`: adapter for `diffpy.structure.Atom`.
"""

__all__ = ["DiffpyStructureParSet"]

from diffpy.cmistructure.srrealparset import SrRealParSet
from diffpy.srfit.fitbase.parameter import ParameterAdapter, ParameterProxy
from diffpy.srfit.fitbase.parameterset import ParameterSet
from diffpy.srfit.util.argbinders import bind2nd


# Accessor for xyz of atoms
class _xyzgetter(object):

    def __init__(self, i):
        self.i = i

    def __call__(self, atom):
        return atom.xyz[self.i]


class _xyzsetter(object):

    def __init__(self, i):
        self.i = i

    def __call__(self, atom, value):
        atom.xyz[self.i] = value


class DiffpyAtomParSet(ParameterSet):
    """Adapt a diffpy.structure.Atom to the ParameterSet interface.

    This class derives from diffpy.srfit.fitbase.parameterset.ParameterSet.
    See that class for base attributes.

    Attributes
    ----------
    atom : diffpy.structure.Atom
        The atom this is adapting.
    element : str
        The element name (property).
    x, y, z : ParameterAdapter
        The fractional coordinates of the atom.
    occupancy : ParameterAdapter
        The occupancy of the atom on its crystal location.
    occ : ParameterProxy
        The proxy for `occupancy`.
    Uij : ParameterAdapter or ParameterProxy
        The anisotropic displacement factors U11, U22, U33, U12, U21, U13,
        U31, U23 and U32 of the atom. The Uij and Uji parameters are the
        same.
    Uiso : ParameterAdapter
        The isotropic displacement factor of the atom.
    Bij : ParameterAdapter or ParameterProxy
        The anisotropic displacement factors B11, B22, B33, B12, B21, B13,
        B31, B23 and B32 of the atom, with Bij = 8*pi**2*Uij. The Bij and
        Bji parameters are the same.
    Biso : ParameterAdapter
        The isotropic displacement factor of the atom, as a B-factor.
    """

    def __init__(self, name, atom):
        """Initialize the atom ParameterSet.

        Parameters
        ----------
        name : str
            The name of this ParameterSet.
        atom : diffpy.structure.Atom
            The atom to adapt.
        """
        ParameterSet.__init__(self, name)
        self.atom = atom
        a = atom
        # x, y, z, occupancy
        self.add_parameter(
            ParameterAdapter("x", a, _xyzgetter(0), _xyzsetter(0))
        )
        self.add_parameter(
            ParameterAdapter("y", a, _xyzgetter(1), _xyzsetter(1))
        )
        self.add_parameter(
            ParameterAdapter("z", a, _xyzgetter(2), _xyzsetter(2))
        )
        occupancy = ParameterAdapter("occupancy", a, attr="occupancy")
        self.add_parameter(occupancy)
        self.add_parameter(ParameterProxy("occ", occupancy))
        # U
        self.add_parameter(ParameterAdapter("U11", a, attr="U11"))
        self.add_parameter(ParameterAdapter("U22", a, attr="U22"))
        self.add_parameter(ParameterAdapter("U33", a, attr="U33"))
        U12 = ParameterAdapter("U12", a, attr="U12")
        U21 = ParameterProxy("U21", U12)
        U13 = ParameterAdapter("U13", a, attr="U13")
        U31 = ParameterProxy("U31", U13)
        U23 = ParameterAdapter("U23", a, attr="U23")
        U32 = ParameterProxy("U32", U23)
        self.add_parameter(U12)
        self.add_parameter(U21)
        self.add_parameter(U13)
        self.add_parameter(U31)
        self.add_parameter(U23)
        self.add_parameter(U32)
        self.add_parameter(ParameterAdapter("Uiso", a, attr="Uisoequiv"))
        # B
        self.add_parameter(ParameterAdapter("B11", a, attr="B11"))
        self.add_parameter(ParameterAdapter("B22", a, attr="B22"))
        self.add_parameter(ParameterAdapter("B33", a, attr="B33"))
        B12 = ParameterAdapter("B12", a, attr="B12")
        B21 = ParameterProxy("B21", B12)
        B13 = ParameterAdapter("B13", a, attr="B13")
        B31 = ParameterProxy("B31", B13)
        B23 = ParameterAdapter("B23", a, attr="B23")
        B32 = ParameterProxy("B32", B23)
        self.add_parameter(B12)
        self.add_parameter(B21)
        self.add_parameter(B13)
        self.add_parameter(B31)
        self.add_parameter(B23)
        self.add_parameter(B32)
        self.add_parameter(ParameterAdapter("Biso", a, attr="Bisoequiv"))
        return

    def __repr__(self):
        return repr(self.atom)

    def _getelem(self):
        return self.atom.element

    def _setelem(self, el):
        self.atom.element = el

    element = property(_getelem, _setelem, "type of atom")


# End class DiffpyAtomParSet


def _latgetter(parameter):
    return bind2nd(getattr, parameter)


def _latsetter(parameter):
    return bind2nd(setattr, parameter)


class DiffpyLatticeParSet(ParameterSet):
    """Adapt a diffpy.structure.Lattice to the ParameterSet interface.

    This class derives from diffpy.srfit.fitbase.parameterset.ParameterSet.
    See that class for base attributes.

    Attributes
    ----------
    lattice : diffpy.structure.Lattice
        The lattice this is adapting.
    name : str
        The name of this ParameterSet, always "lattice".
    angle_units : str
        The units of the lattice angles, always "deg".
    a, b, c, alpha, beta, gamma : ParameterAdapter
        The unit cell parameters.
    """

    def __init__(self, lattice):
        """Initialize the lattice ParameterSet.

        Parameters
        ----------
        lattice : diffpy.structure.Lattice
            The lattice to adapt.
        """
        ParameterSet.__init__(self, "lattice")
        self.angle_units = "deg"
        self.lattice = lattice
        lat = lattice
        self.add_parameter(
            ParameterAdapter("a", lat, _latgetter("a"), _latsetter("a"))
        )
        self.add_parameter(
            ParameterAdapter("b", lat, _latgetter("b"), _latsetter("b"))
        )
        self.add_parameter(
            ParameterAdapter("c", lat, _latgetter("c"), _latsetter("c"))
        )
        self.add_parameter(
            ParameterAdapter(
                "alpha", lat, _latgetter("alpha"), _latsetter("alpha")
            )
        )
        self.add_parameter(
            ParameterAdapter(
                "beta", lat, _latgetter("beta"), _latsetter("beta")
            )
        )
        self.add_parameter(
            ParameterAdapter(
                "gamma", lat, _latgetter("gamma"), _latsetter("gamma")
            )
        )
        return

    def __repr__(self):
        return repr(self.lattice)


# End class DiffpyLatticeParSet


class DiffpyStructureParSet(SrRealParSet):
    """Adapt a diffpy.structure.Structure to the ParameterSet interface.

    This class derives from SrRealParSet. See that class for base
    attributes.

    Attributes
    ----------
    atoms : list of DiffpyAtomParSet
        The atom ParameterSets, provided for convenience.
    structure : diffpy.structure.Structure
        The structure this is adapting.
    lattice : DiffpyLatticeParSet
        The managed lattice ParameterSet.
    <el><idx> : DiffpyAtomParSet
        The managed atom ParameterSets. <el> is the atomic element and
        <idx> is the index of that element in the structure, starting
        from zero. For nickel in P1 symmetry, the managed
        DiffpyAtomParSets are named "Ni0", "Ni1", "Ni2" and "Ni3".
    """

    def __init__(self, name, structure):
        """Initialize the structure ParameterSet.

        Parameters
        ----------
        name : str
            The name of the structure.
        structure : diffpy.structure.Structure
            The structure to adapt.
        """
        SrRealParSet.__init__(self, name)
        self.structure = structure
        self.add_parameter_set(DiffpyLatticeParSet(structure.lattice))
        self.atoms = []

        cdict = {}
        for a in structure:
            el = a.element.title()
            # Try to sanitize the name.
            el = el.replace("+", "p")
            el = el.replace("-", "m")
            i = cdict.get(el, 0)
            aname = "%s%i" % (el, i)
            cdict[el] = i + 1
            atom = DiffpyAtomParSet(aname, a)
            self.add_parameter_set(atom)
            self.atoms.append(atom)

        return

    def __repr__(self):
        return repr(self.structure)

    def get_lattice(self):
        """Return the ParameterSet containing the lattice Parameters.

        Returns
        -------
        DiffpyLatticeParSet
            The lattice ParameterSet of the structure.
        """
        return self.lattice

    @classmethod
    def can_adapt(self, structure):
        """Return whether the structure can be adapted by this class.

        Parameters
        ----------
        structure : object
            The structure object to check.

        Returns
        -------
        bool
            The flag indicating if `structure` is a diffpy.structure.Structure.
        """
        from diffpy.structure import Structure

        return isinstance(structure, Structure)

    def get_scatterers(self):
        """Return the list of ParameterSets that represent the
        scatterers.

        Returns
        -------
        list of DiffpyAtomParSet
            The atom ParameterSets of the structure.
        """
        return self.atoms

    def _get_srreal_structure(self):
        """Get the structure object for use with SrReal calculators.

        If this is periodic, then return the structure, otherwise, pass
        it inside of a nosymmetry wrapper. This takes the extra step of
        wrapping the structure in a nometa wrapper.
        """
        from diffpy.srreal.structureadapter import nometa

        structure = SrRealParSet._get_srreal_structure(self)
        return nometa(structure)


# End class DiffpyStructureParSet
