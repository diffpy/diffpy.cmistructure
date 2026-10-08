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
"""Wrappers for interfacing cctbx crystal with SrFit.

This wraps a cctbx.crystal as a ParameterSet with a similar hierarchy, which
can then be used within a FitRecipe. Note that all manipulations to the
cctbx.crystal should be done before wrapping. Changes made to the cctbx.crystal
object after wrapping may not be reflected within the wrapper, which can have
unpredictable results during a structure refinement.

The following classes are adapted:

- `CCTBXCrystalParSet`: wrapper for `cctbx.crystal`.
- `CCTBXUnitCellParSet`: wrapper for the unit cell of `cctbx.crystal`.
- `CCTBXScattererParSet`: wrapper for `cctbx.xray.scatterer`.
"""

from diffpy.cmistructure.basestructureparset import BaseStructureParSet
from diffpy.srfit.fitbase.parameter import ParameterAdapter
from diffpy.srfit.fitbase.parameterset import ParameterSet

__all__ = ["CCTBXScattererParSet", "CCTBXUnitCellParSet", "CCTBXCrystalParSet"]


class CCTBXScattererParSet(ParameterSet):
    """Adapt a cctbx.xray.scatterer to the ParameterSet interface.

    This class derives from ParameterSet.

    Attributes
    ----------
    name : str
        The name of the scatterer. The name is always of the form
        "%s%i" % (element, number), where the number is the running
        index of that element type (starting at 0).
    x, y, z : ParameterAdapter
        The atom position in crystal coordinates.
    occupancy : ParameterAdapter
        The occupancy of the atom on its crystal location.
    Uiso : ParameterAdapter
        The isotropic displacement factor of the atom.
    """

    def __init__(self, name, structure_parameter_set, index):
        """Initialize the scatterer ParameterSet.

        Parameters
        ----------
        name : str
            The name of this scatterer.
        structure_parameter_set : CCTBXCrystalParSet
            The CCTBXCrystalParSet that contains the cctbx structure.
        index : int
            The index of the scatterer in the structure.
        """
        ParameterSet.__init__(self, name)
        self.structure_parameter_set = structure_parameter_set
        self.index = index

        # x, y, z, occupancy
        self.add_parameter(
            ParameterAdapter("x", None, self._xyzgetter(0), self._xyzsetter(0))
        )
        self.add_parameter(
            ParameterAdapter("y", None, self._xyzgetter(1), self._xyzsetter(1))
        )
        self.add_parameter(
            ParameterAdapter("z", None, self._xyzgetter(2), self._xyzsetter(2))
        )
        self.add_parameter(
            ParameterAdapter("occupancy", None, self._getocc, self._setocc)
        )
        self.add_parameter(
            ParameterAdapter("Uiso", None, self._getuiso, self._setuiso)
        )
        return

    # Getters and setters

    def _xyzgetter(self, i):

        def f(dummy):
            return self.structure_parameter_set.structure.scatterers()[
                self.index
            ].site[i]

        return f

    def _xyzsetter(self, i):

        def f(dummy, value):
            xyz = list(
                self.structure_parameter_set.structure.scatterers()[
                    self.index
                ].site
            )
            xyz[i] = value
            self.structure_parameter_set.structure.scatterers()[
                self.index
            ].site = tuple(xyz)
            return

        return f

    def _getocc(self, dummy):
        return self.structure_parameter_set.structure.scatterers()[
            self.index
        ].occupancy

    def _setocc(self, dummy, value):
        self.structure_parameter_set.structure.scatterers()[
            self.index
        ].occupancy = value
        return

    def _getuiso(self, dummy):
        return self.structure_parameter_set.structure.scatterers()[
            self.index
        ].u_iso

    def _setuiso(self, dummy, value):
        self.structure_parameter_set.structure.scatterers()[
            self.index
        ].u_iso = value
        return

    def _getelem(self):
        return self.structure.element_symbol()

    element = property(_getelem)


# End class CCTBXScattererParSet


class CCTBXUnitCellParSet(ParameterSet):
    """Adapt a cctbx unit_cell to the ParameterSet interface.

    Attributes
    ----------
    name : str
        The name of this ParameterSet, always "unitcell".
    a, b, c, alpha, beta, gamma : ParameterAdapter
        The unit cell parameters.
    """

    def __init__(self, structure_parameter_set):
        """Initialize the unit cell ParameterSet.

        Parameters
        ----------
        structure_parameter_set : CCTBXCrystalParSet
            The CCTBXCrystalParSet that contains the cctbx structure
            and the unit cell being wrapped.
        """
        ParameterSet.__init__(self, "unitcell")
        self.structure_parameter_set = structure_parameter_set
        self._lattice_parameters = list(
            self.structure_parameter_set.structure.unit_cell().parameters()
        )

        self.add_parameter(
            ParameterAdapter("a", None, self._latgetter(0), self._latsetter(0))
        )
        self.add_parameter(
            ParameterAdapter("b", None, self._latgetter(1), self._latsetter(1))
        )
        self.add_parameter(
            ParameterAdapter("c", None, self._latgetter(2), self._latsetter(2))
        )
        self.add_parameter(
            ParameterAdapter(
                "alpha", None, self._latgetter(3), self._latsetter(3)
            )
        )
        self.add_parameter(
            ParameterAdapter(
                "beta", None, self._latgetter(4), self._latsetter(4)
            )
        )
        self.add_parameter(
            ParameterAdapter(
                "gamma", None, self._latgetter(5), self._latsetter(5)
            )
        )

        return

    def _latgetter(self, i):

        def f(dummy):
            return self._lattice_parameters[i]

        return f

    def _latsetter(self, i):

        def f(dummy, value):
            self._lattice_parameters[i] = value
            self.structure_parameter_set._update = True
            return

        return f


# End class CCTBXUnitCellParSet

# FIXME - Special positions should be constant.


class CCTBXCrystalParSet(BaseStructureParSet):
    """Adapt a cctbx structure to the ParameterSet interface.

    Attributes
    ----------
    structure : cctbx.crystal.special_position_settings
        The adapted cctbx structure object.
    scatterers : list of CCTBXScattererParSet
        The scatterer ParameterSets.
    unitcell : CCTBXUnitCellParSet
        The unit cell ParameterSet for the structure.
    """

    def __init__(self, name, structure):
        """Initialize the crystal ParameterSet.

        Parameters
        ----------
        name : str
            The name of this ParameterSet.
        structure : cctbx.crystal.special_position_settings
            The cctbx structure to adapt.
        """
        ParameterSet.__init__(self, name)
        self.structure = structure
        self.add_parameter_set(CCTBXUnitCellParSet(self))
        self.scatterers = []

        self._update = False

        cdict = {}
        for s in structure.scatterers():
            el = s.element_symbol()
            i = cdict.get(el, 0)
            sname = "%s%i" % (el, i)
            cdict[el] = i + 1
            scatterer = CCTBXScattererParSet(sname, self, i)
            self.add_parameter_set(scatterer)
            self.scatterers.append(scatterer)

        # Constrain the lattice
        from diffpy.cmistructure.sgconstraints import _constrain_space_group

        symbol = self.get_space_group()
        _constrain_space_group(self, symbol)

        return

    def update(self):
        """Rebuild the unit cell after a change in lattice parameters.

        Call this function before using the CCTBXCrystalParSet. The unit
        cell is only remade if a lattice parameter has changed.
        """
        if not self._update:
            return

        self._update = False
        structure = self.structure
        sgn = structure.space_group().match_tabulated_settings().number()

        # Create the symmetry object
        from cctbx.crystal import symmetry

        symm = symmetry(
            unit_cell=self.unitcell._lattice_parameters, space_group_symbol=sgn
        )

        # Now the new structure
        newstru = structure.__class__(
            crystal_symmetry=symm, scatterers=structure.scatterers()
        )

        self.unitcell._lattice_parameters = list(
            newstru.unit_cell().parameters()
        )

        self.structure = newstru
        return

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
            The flag indicating if `structure` is a
            cctbx.crystal.special_position_settings. False if cctbx is
            not installed.
        """
        try:
            from cctbx.crystal import special_position_settings
        except ImportError:
            return False
        return isinstance(structure, special_position_settings)

    def get_lattice(self):
        """Return the ParameterSet containing the lattice Parameters.

        Returns
        -------
        CCTBXUnitCellParSet
            The unit cell ParameterSet of the structure.
        """
        return self.unitcell

    def get_scatterers(self):
        """Return the list of ParameterSets that represent the
        scatterers.

        Returns
        -------
        list of CCTBXScattererParSet
            The scatterer ParameterSets of the structure.
        """
        return self.scatterers

    def get_space_group(self):
        """Return the Hermann-Mauguin space group symbol of the
        structure.

        Returns
        -------
        str
            The Hermann-Mauguin space group symbol.
        """
        space_group = self.structure.space_group()
        t = space_group.type()
        return t.lookup_symbol()


# End class CCTBXCrystalParSet
