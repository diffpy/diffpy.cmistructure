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
"""Code to set space group constraints for a crystal structure."""

import re

import numpy

from diffpy.srfit.fitbase.parameter import ParameterProxy
from diffpy.srfit.fitbase.recipeorganizer import RecipeContainer

__all__ = ["constrain_as_space_group"]


def constrain_as_space_group(
    phase,
    spacegroup,
    scatterers=None,
    sgoffset=[0, 0, 0],
    constrainlat=True,
    constrainadps=True,
    adpsymbols=None,
    isosymbol="Uiso",
):
    """Constrain a P1 structure to a space group.

    This applies space group constraints to a structure ParameterSet with
    P1 symmetry. The passed scatterers are explicitly constrained to the
    specified space group, and the ADPs and lattice may be constrained as
    well. New Parameters used in the constraints are created within the
    returned SpaceGroupParameters object. Constraints are created in the
    ParameterSet that contains the constrained Parameter. This erases any
    constraints or constant flags on the scatterers, lattice or ADPs that
    are to be constrained.

    Parameters
    ----------
    phase : BaseStructureParSet
        The structure ParameterSet to constrain.
    spacegroup : int, str or diffpy.structure.spacegroups.SpaceGroup
        The space group number, symbol or SpaceGroup instance.
    scatterers : list of ParameterSet, optional
        The scatterer ParameterSets to constrain. If None (default), all
        scatterers returned by ``phase.get_scatterers()`` are constrained.
    sgoffset : list of float, optional
        The offset of the space group origin (default [0, 0, 0]).
    constrainlat : bool, optional
        The flag indicating whether to constrain the lattice (default
        True).
    constrainadps : bool, optional
        The flag indicating whether to constrain the ADPs (default True).
    adpsymbols : list of str, optional
        The ADP names. By default this is
        diffpy.structure.symmetryutilities.stdUsymbols (U11, U22, etc.).
        The names must be given in the same order as stdUsymbols.
    isosymbol : str, optional
        The name of the isotropic ADP (default "Uiso"). If None,
        isotropic ADPs are constrained via the anisotropic ADPs.

    Returns
    -------
    SpaceGroupParameters
        The free Parameters of the structure that remain after applying
        the space group constraints.

    Notes
    -----
    The lattice constraints are applied as follows.

    Triclinic
        No constraints.
    Monoclinic
        alpha and beta are fixed to 90 unless alpha != beta and
        alpha == gamma, in which case alpha and gamma are fixed to 90.
    Orthorhombic
        alpha, beta and gamma are fixed to 90.
    Tetragonal
        b is constrained to a and alpha, beta and gamma are fixed to 90.
    Trigonal
        If gamma == 120, then b is constrained to a, alpha and beta are
        fixed to 90 and gamma is fixed to 120. Otherwise, b and c are
        constrained to a, and beta and gamma are fixed to alpha.
    Hexagonal
        b is constrained to a, alpha and beta are fixed to 90 and gamma
        is fixed to 120.
    Cubic
        b and c are constrained to a, and alpha, beta and gamma are fixed
        to 90.
    """
    from diffpy.structure.spacegroups import SpaceGroup, get_space_group

    space_group = spacegroup
    if not isinstance(spacegroup, SpaceGroup):
        space_group = get_space_group(spacegroup)
    sgp = _constrain_as_space_group(
        phase,
        space_group,
        scatterers,
        sgoffset,
        constrainlat,
        constrainadps,
        adpsymbols,
        isosymbol,
    )

    return sgp


def _constrain_as_space_group(
    phase,
    space_group,
    scatterers=None,
    sgoffset=[0, 0, 0],
    constrainlat=True,
    constrainadps=True,
    adpsymbols=None,
    isosymbol="Uiso",
):
    """Restricted interface to constrain_as_space_group.

    Arguments: As constrain_as_space_group, except
    -----------------------------------------------
    sg
        diffpy.structure.spacegroups.SpaceGroup instance
    """
    from diffpy.structure.symmetryutilities import stdUsymbols

    if scatterers is None:
        scatterers = phase.get_scatterers()
    if adpsymbols is None:
        adpsymbols = stdUsymbols

    sgp = SpaceGroupParameters(
        phase,
        space_group,
        scatterers,
        sgoffset,
        constrainlat,
        constrainadps,
        adpsymbols,
        isosymbol,
    )

    return sgp


# End constrain_as_space_group


class BaseSpaceGroupParameters(RecipeContainer):
    """Base class for holding space group Parameters.

    This class stores the variable Parameters of a structure, leaving out
    those that are constrained or fixed by the space group. It has the same
    Parameter attribute access as a ParameterSet, which makes it easy to
    access the free variables of a structure when scripting.

    Attributes
    ----------
    name : str
        The name of this container (default "sgpars").
    """

    def __init__(self, name="sgpars"):
        """Initialize the space group Parameter container.

        Parameters
        ----------
        name : str, optional
            The name of this container (default "sgpars").
        """
        RecipeContainer.__init__(self, name)
        return

    def add_parameter(self, parameter, check=True):
        """Store a Parameter.

        Parameters
        ----------
        parameter : Parameter
            The Parameter to be stored.
        check : bool, optional
            The flag indicating whether to check for an existing Parameter
            of the same name (default True).

        Raises
        ------
        ValueError
            If the Parameter has no name, or if `check` is True and a
            Parameter of the same name has already been stored.
        """
        # Store the Parameter
        RecipeContainer._add_object(self, parameter, self._parameters, check)
        return


# End class BaseSpaceGroupParameters


class SpaceGroupParameters(BaseSpaceGroupParameters):
    """Create and hold the free Parameters of a space group constraint.

    This class stores the variable Parameters of a structure, leaving out
    those that are constrained or fixed by the space group, and does the
    work of constrain_as_space_group. It has the same Parameter attribute
    access as a ParameterSet.

    Attributes
    ----------
    name : str
        The name of this container, always "sgpars".
    phase : BaseStructureParSet
        The constrained structure ParameterSet.
    space_group : diffpy.structure.spacegroups.SpaceGroup
        The space group of the constraints.
    sgoffset : list of float
        The offset of the space group origin.
    scatterers : list of ParameterSet
        The constrained scatterer ParameterSets.
    constrainlat : bool
        The flag indicating whether the lattice is constrained.
    constrainadps : bool
        The flag indicating whether the ADPs are constrained.
    adpsymbols : list of str
        The ADP names.
    isosymbol : str or None
        The name of the isotropic ADP.
    xyz_parameters : BaseSpaceGroupParameters
        The free xyz Parameters, created on first access.
    lattice_parameters : BaseSpaceGroupParameters
        The free lattice Parameters, created on first access.
    adp_parameters : BaseSpaceGroupParameters
        The free ADP Parameters, created on first access.
    """

    def __init__(
        self,
        phase,
        space_group,
        scatterers,
        sgoffset,
        constrainlat,
        constrainadps,
        adpsymbols,
        isosymbol,
    ):
        """Initialize the space group Parameters.

        The constraints are not applied until the Parameters are first
        accessed.

        Parameters
        ----------
        phase : BaseStructureParSet
            The structure ParameterSet to be constrained.
        space_group : diffpy.structure.spacegroups.SpaceGroup
            The space group of the constraints.
        scatterers : list of ParameterSet
            The scatterer ParameterSets to constrain.
        sgoffset : list of float
            The offset of the space group origin.
        constrainlat : bool
            The flag indicating whether to constrain the lattice.
        constrainadps : bool
            The flag indicating whether to constrain the ADPs.
        adpsymbols : list of str
            The ADP names, in the same order as
            diffpy.structure.symmetryutilities.stdUsymbols.
        isosymbol : str or None
            The name of the isotropic ADP. If None, isotropic ADPs are
            constrained via the anisotropic ADPs.
        """
        BaseSpaceGroupParameters.__init__(self)
        self._lattice_parameters = None
        self._xyz_parameters = None
        self._adp_parameters = None

        self._parsets = {}
        self._manage(self._parsets)

        self.phase = phase
        self.space_group = space_group
        self.sgoffset = sgoffset
        self.scatterers = scatterers
        self.constrainlat = constrainlat
        self.constrainadps = constrainadps
        self.adpsymbols = adpsymbols
        self.isosymbol = isosymbol

        return

    def __iter__(self):
        """Iterate over top-level parameters."""
        if (
            self._lattice_parameters is None
            or self._xyz_parameters is None
            or self._adp_parameters is None
        ):
            self._make_constraints()
        return RecipeContainer.__iter__(self)

    lattice_parameters = property(lambda self: self._get_lat_pars())

    def _get_lat_pars(self):
        """Accessor for _lattice_parameters."""
        if self._lattice_parameters is None:
            self._constrain_lattice()
        return self._lattice_parameters

    xyz_parameters = property(lambda self: self._get_xyz_pars())

    def _get_xyz_pars(self):
        """Accessor for _xyz_parameters."""
        positions = []
        for scatterer in self.scatterers:
            xyz = [scatterer.x, scatterer.y, scatterer.z]
            positions.append([p.value for p in xyz])
        if self._xyz_parameters is None:
            self._constrain_xyzs(positions)
        return self._xyz_parameters

    adp_parameters = property(lambda self: self._get_adp_pars())

    def _get_adp_pars(self):
        """Accessor for _adp_parameters."""
        positions = []
        for scatterer in self.scatterers:
            xyz = [scatterer.x, scatterer.y, scatterer.z]
            positions.append([p.value for p in xyz])
        if self._adp_parameters is None:
            self._constrain_adps(positions)
        return self._adp_parameters

    def _make_constraints(self):
        """Constrain the structure to the space group.

        This works as described by the constrain_as_space_group method.
        """
        # Start by clearing the constraints
        self._clear_constraints()

        scatterers = self.scatterers

        # Prepare positions
        positions = []
        for scatterer in scatterers:
            xyz = [scatterer.x, scatterer.y, scatterer.z]
            positions.append([p.value for p in xyz])

        self._constrain_lattice()
        self._constrain_xyzs(positions)
        self._constrain_adps(positions)

        return

    def _clear_constraints(self):
        """Clear old constraints.

        This only clears constraints where new ones are going to be
        applied.
        """
        phase = self.phase
        scatterers = self.scatterers
        isosymbol = self.isosymbol
        adpsymbols = self.adpsymbols

        # Clear xyz
        for scatterer in scatterers:

            for parameter in [scatterer.x, scatterer.y, scatterer.z]:
                if scatterer.is_constrained(parameter):
                    scatterer.remove_constraint(parameter)
                parameter.set_constant(False)

        # Clear the lattice
        if self.constrainlat:

            lattice = phase.get_lattice()
            lattice_parameters = [
                lattice.a,
                lattice.b,
                lattice.c,
                lattice.alpha,
                lattice.beta,
                lattice.gamma,
            ]
            for parameter in lattice_parameters:
                if lattice.is_constrained(parameter):
                    lattice.remove_constraint(parameter)
                parameter.set_constant(False)

        # Clear ADPs
        if self.constrainadps:
            for scatterer in scatterers:
                if isosymbol:
                    parameter = scatterer.get(isosymbol)
                    if parameter is not None:
                        if scatterer.is_constrained(parameter):
                            scatterer.remove_constraint(parameter)
                        parameter.set_constant(False)

                for pname in adpsymbols:
                    parameter = scatterer.get(pname)
                    if parameter is not None:
                        if scatterer.is_constrained(parameter):
                            scatterer.remove_constraint(parameter)
                        parameter.set_constant(False)

        return

    def _constrain_lattice(self):
        """Constrain the lattice parameters."""
        if not self.constrainlat:
            return

        phase = self.phase
        space_group = self.space_group

        lattice = phase.get_lattice()
        system = space_group.crystal_system
        if not system:
            system = "Triclinic"
        system = system.title()
        # This makes the constraints
        f = _constraint_map[system]
        f(lattice)

        # Now get the unconstrained, non-constant lattice pars and store them.
        self._lattice_parameters = BaseSpaceGroupParameters(
            "lattice_parameters"
        )
        lattice_parameters = [
            lattice.a,
            lattice.b,
            lattice.c,
            lattice.alpha,
            lattice.beta,
            lattice.gamma,
        ]
        pars = [
            p for p in lattice_parameters if not p.const and not p.constrained
        ]
        for parameter in pars:
            # FIXME - the original parameter will still appear as
            # constrained.
            newpar = self.__add_par(parameter.name, parameter)
            self._lattice_parameters.add_parameter(newpar)

        return

    def _constrain_xyzs(self, positions):
        """Constrain the positions.

        Parameters
        ----------
        positions
            The coordinates of the scatterers.
        """
        from diffpy.structure.symmetryutilities import SymmetryConstraints

        space_group = self.space_group
        sgoffset = self.sgoffset

        # We do this without ADPs here so we can skip much complication. See
        # the _constrain_adps method for details.
        g = SymmetryConstraints(space_group, positions, sgoffset=sgoffset)

        scatterers = self.scatterers
        self._xyz_parameters = BaseSpaceGroupParameters("xyz_parameters")

        # Make proxies to the free xyz parameters
        xyznames = [name[:1] + "_" + name[1:] for name, value in g.pospars]
        for pname in xyznames:
            name, index = pname.rsplit("_", 1)
            index = int(index)
            parameter = scatterers[index].get(name)
            newpar = self.__add_par(pname, parameter)
            self._xyz_parameters.add_parameter(newpar)

        # Constrain non-free xyz parameters
        fpos = g.position_formulas(xyznames)
        for index, tmp in enumerate(zip(scatterers, fpos)):
            scatterer, fp = tmp

            # Extract the constraint equation from the formula
            for parname, formula in fp.items():
                _makeconstraint(
                    parname, formula, scatterer, index, self._parameters
                )

        return

    def _constrain_adps(self, positions):
        """Constrain the ADPs.

        Parameters
        ----------
        positions
            The coordinates of the scatterers.
        """
        from diffpy.structure.symmetryutilities import (
            SymmetryConstraints,
            stdUsymbols,
        )

        if not self.constrainadps:
            return

        space_group = self.space_group
        sgoffset = self.sgoffset
        scatterers = self.scatterers
        isosymbol = self.isosymbol
        adpsymbols = self.adpsymbols
        adpmap = dict(zip(stdUsymbols, adpsymbols))
        self._adp_parameters = BaseSpaceGroupParameters("adp_parameters")

        # Prepare ADPs. Note that not all scatterers have constrainable ADPs.
        # For example, MoleculeParSet from objcryststructure does not. We
        # discard those.
        nonadps = []
        Uijs = []
        for sidx, scatterer in enumerate(scatterers):

            pars = [scatterer.get(symb) for symb in adpsymbols]

            if None in pars:
                nonadps.append(sidx)
                continue

            Uij = numpy.zeros((3, 3), dtype=float)
            for index, parameter in enumerate(pars):
                i, j = _idxtoij[index]
                Uij[i, j] = Uij[j, i] = parameter.get_value()

            Uijs.append(Uij)

        # Discard any positions for the nonadps
        positions = list(positions)
        nonadps.reverse()
        [positions.pop(index) for index in nonadps]

        # Now we can create symmetry constraints without having to worry about
        # the nonadps
        g = SymmetryConstraints(
            space_group, positions, Uijs, sgoffset=sgoffset
        )

        adpnames = [
            adpmap[name[:3]] + "_" + name[3:] for name, value in g.Upars
        ]

        # Make proxies to the free adp parameters. We start by filtering out
        # the isotropic ones so we can use the isotropic parameter.
        isoidx = []
        isonames = []
        for pname in adpnames:
            name, index = pname.rsplit("_", 1)
            index = int(index)
            # Check for isotropic ADPs
            scatterer = scatterers[index]
            if isosymbol and g.Uisotropy[index] and index not in isoidx:
                isoidx.append(index)
                parameter = scatterer.get(isosymbol)
                if parameter is not None:
                    parname = "%s_%i" % (isosymbol, index)
                    newpar = self.__add_par(parname, parameter)
                    self._adp_parameters.add_parameter(newpar)
                    isonames.append(newpar.name)
            else:
                parameter = scatterer.get(name)
                if parameter is not None:
                    newpar = self.__add_par(pname, parameter)
                    self._adp_parameters.add_parameter(newpar)

        # Constrain dependent isotropics
        for index, isoname in zip(isoidx[:], isonames):
            for j in g.coremap[index]:
                if j == index:
                    continue
                isoidx.append(j)
                scatterer = scatterers[j]
                scatterer.add_constraint(
                    isosymbol, isoname, params=self._parameters
                )

        fadp = g.u_formulas(adpnames)

        # Constrain dependent anisotropics. We use the fact that an
        # anisotropic cannot be dependent on an isotropic.
        for index, tmp in enumerate(zip(scatterers, fadp)):
            if index in isoidx:
                continue
            scatterer, fa = tmp
            # Extract the constraint equation from the formula
            for stdparname, formula in fa.items():
                pname = adpmap[stdparname]
                _makeconstraint(
                    pname, formula, scatterer, index, self._parameters
                )

    def __add_par(self, parname, parameter):
        """Constrain a parameter via proxy with a specified name.

        Parameters
        ----------
        par
            Parameter to constrain
        idx
            Index to identify scatterer from which par comes
        """
        newpar = ParameterProxy(parname, parameter)
        self.add_parameter(newpar)
        return newpar


# End class SpaceGroupParameters

# crystal system rules
# ref: Benjamin, W. A., Introduction to crystallography,
# New York (1969), p.60


def _constrain_triclinic(lattice):
    """Make constraints for Triclinic systems."""
    return


def _constrain_monoclinic(lattice):
    """Make constraints for Monoclinic systems.

    alpha and beta are fixed to 90 unless alpha != beta and alpha ==
    gamma, in which case alpha and gamma are constrained to 90.
    """
    afactor = 1
    if lattice.angle_units == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    lattice.alpha.set_constant(True, ang90)
    beta = lattice.beta.get_value()
    gamma = lattice.gamma.get_value()

    if ang90 != beta and ang90 == gamma:
        lattice.gamma.set_constant(True, ang90)
    else:
        lattice.beta.set_constant(True, ang90)
    return


def _constrain_orthorhombic(lattice):
    """Make constraints for Orthorhombic systems.

    alpha, beta and gamma are constrained to 90
    """
    afactor = 1
    if lattice.angle_units == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    lattice.alpha.set_constant(True, ang90)
    lattice.beta.set_constant(True, ang90)
    lattice.gamma.set_constant(True, ang90)
    return


def _constrain_tetragonal(lattice):
    """Make constraints for Tetragonal systems.

    b is constrained to a and alpha, beta and gamma are constrained to
    90.
    """
    afactor = 1
    if lattice.angle_units == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    lattice.alpha.set_constant(True, ang90)
    lattice.beta.set_constant(True, ang90)
    lattice.gamma.set_constant(True, ang90)
    lattice.add_constraint(lattice.b, lattice.a)
    return


def _constrain_trigonal(lattice):
    """Make constraints for Trigonal systems.

    If gamma == 120, then b is constrained to a, alpha and beta are
    constrained to 90 and gamma is constrained to 120. Otherwise, b and
    c are constrained to a, beta and gamma are constrained to alpha.
    """
    afactor = 1
    if lattice.angle_units == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    ang120 = 120.0 * afactor
    if lattice.gamma.get_value() == ang120:
        lattice.add_constraint(lattice.b, lattice.a)
        lattice.alpha.set_constant(True, ang90)
        lattice.beta.set_constant(True, ang90)
        lattice.gamma.set_constant(True, ang120)
    else:
        lattice.add_constraint(lattice.b, lattice.a)
        lattice.add_constraint(lattice.c, lattice.a)
        lattice.add_constraint(lattice.beta, lattice.alpha)
        lattice.add_constraint(lattice.gamma, lattice.alpha)
    return


def _constrain_hexagonal(lattice):
    """Make constraints for Hexagonal systems.

    b is constrained to a, alpha and beta are constrained to 90 and
    gamma is constrained to 120.
    """
    afactor = 1
    if lattice.angle_units == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    ang120 = 120.0 * afactor
    lattice.add_constraint(lattice.b, lattice.a)
    lattice.alpha.set_constant(True, ang90)
    lattice.beta.set_constant(True, ang90)
    lattice.gamma.set_constant(True, ang120)
    return


def _constrain_cubic(lattice):
    """Make constraints for Cubic systems.

    b and c are constrained to a, alpha, beta and gamma are constrained
    to 90.
    """
    afactor = 1
    if lattice.angle_units == "rad":
        afactor = deg2rad
    ang90 = 90.0 * afactor
    lattice.add_constraint(lattice.b, lattice.a)
    lattice.add_constraint(lattice.c, lattice.a)
    lattice.alpha.set_constant(True, ang90)
    lattice.beta.set_constant(True, ang90)
    lattice.gamma.set_constant(True, ang90)
    return


# This is used to map the correct crystal system to the proper constraint
# function.
_constraint_map = {
    "Triclinic": _constrain_triclinic,
    "Monoclinic": _constrain_monoclinic,
    "Orthorhombic": _constrain_orthorhombic,
    "Tetragonal": _constrain_tetragonal,
    "Trigonal": _constrain_trigonal,
    "Hexagonal": _constrain_hexagonal,
    "Cubic": _constrain_cubic,
}


def _makeconstraint(parname, formula, scatterer, index, ns={}):
    """Constrain a parameter according to a formula.

    Parameters
    ----------
    parname
        Name of parameter
    formula
        Constraint formula
    scatterer
        scatterer containing par of parname
    idx
        Index to identify scatterer from which par comes
    ns
        namespace to draw extra names from (default {})

    Returns
    -------
    par
        Returns the parameter if it is free.
    """
    parameter = scatterer.get(parname)

    if parameter is None:
        return

    compname = "%s_%i" % (parname, index)

    # Check to see if this parameter is free
    pat = r"%s *([+-] *\d+)?$" % compname
    if re.match(pat, formula):
        return parameter

    # Check to see if it is a constant
    fval = _get_float(formula)
    if fval is not None:
        parameter.set_constant()
        return

    # If we got here, then we have a constraint equation
    # Fix any division issues
    formula = formula.replace("/", "*1.0/")
    scatterer.add_constraint(parameter, formula, params=ns)
    return


def _get_float(formula):
    """Get a float from a formula string, or None if this is not
    possible."""
    try:
        return eval(formula)
    except NameError:
        return None


# Constants needed above
_idxtoij = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
deg2rad = numpy.pi / 180
rad2deg = 1.0 / deg2rad


# End of file
