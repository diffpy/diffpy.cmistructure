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
"""Wrappers for adapting pyobjcryst.crystal.Crystal to a srfit
ParameterSet.

This will adapt a Crystal or Molecule object from pyobjcryst into the
ParameterSet interface. The following classes are adapted:

- `ObjCrystCrystalParSet`: adapter for `pyobjcryst.crystal.Crystal`.
- `ObjCrystAtomParSet`: adapter for `pyobjcryst.atom.Atom`.
- `ObjCrystMoleculeParSet`: adapter for `pyobjcryst.molecule.Molecule`.
- `ObjCrystMolAtomParSet`: adapter for `pyobjcryst.molecule.MolAtom`.

Related to the adaptation of Molecule and MolAtom, there are adaptors
for specifying molecule restraints:

- `ObjCrystBondLengthRestraint`
- `ObjCrystBondAngleRestraint`
- `ObjCrystDihedralAngleRestraint`

There are also Parameters for encapsulating and modifying atoms via
their relative positions. These Parameters can also act like
constraints, and can modify the positions of multiple MolAtoms:

- `ObjCrystBondLengthParameter`
- `ObjCrystBondAngleParameter`
- `ObjCrystDihedralAngleParameter`
"""

__all__ = ["ObjCrystMoleculeParSet", "ObjCrystCrystalParSet"]

import numpy
from pyobjcryst.molecule import (
    GetBondAngle,
    GetBondLength,
    GetDihedralAngle,
    StretchModeBondAngle,
    StretchModeBondLength,
    StretchModeTorsion,
)

from diffpy.cmistructure.srrealparset import SrRealParSet
from diffpy.srfit.fitbase.parameter import (
    Parameter,
    ParameterAdapter,
    ParameterProxy,
)
from diffpy.srfit.fitbase.parameterset import ParameterSet


class ObjCrystScattererParSet(ParameterSet):
    """Base adapter for a pyobjcryst scatterer.

    This class derives from diffpy.srfit.fitbase.parameterset.ParameterSet
    and adapts pyobjcryst.scatterer.Scatterer derivatives (Molecule, Atom)
    and objects with a similar interface (MolAtom). See the ParameterSet
    class for base attributes.

    Attributes
    ----------
    scatterer : pyobjcryst.scatterer.Scatterer
        The adapted pyobjcryst object.
    parent : ParameterSet or None
        The ParameterSet this belongs to.
    x, y, z : ParameterAdapter
        The position of the scatterer in crystal coordinates.
    occ : ParameterAdapter
        The occupancy of the scatterer on its crystal site.
    """

    def __init__(self, name, scatterer, parent):
        """Initialize the scatterer ParameterSet.

        Parameters
        ----------
        name : str
            The name of the scatterer.
        scatterer : pyobjcryst.scatterer.Scatterer
            The pyobjcryst scatterer to adapt.
        parent : ParameterSet or None
            The ParameterSet this belongs to.
        """
        ParameterSet.__init__(self, name)
        self.scatterer = scatterer
        self.parent = parent

        # x, y, z, occ
        self.add_parameter(ParameterAdapter("x", self.scatterer, attr="X"))
        self.add_parameter(ParameterAdapter("y", self.scatterer, attr="Y"))
        self.add_parameter(ParameterAdapter("z", self.scatterer, attr="Z"))
        self.add_parameter(
            ParameterAdapter("occ", self.scatterer, attr="Occupancy")
        )
        return

    def is_dummy(self):
        """Return whether this scatterer is a dummy atom.

        Returns
        -------
        bool
            The flag indicating if this is a dummy atom. Always False for
            this class.
        """
        return False

    def has_scatterers(self):
        """Return whether this scatterer has its own scatterers.

        Returns
        -------
        bool
            The flag indicating if this scatterer has a ``get_scatterers``
            method.
        """
        return hasattr(self, "get_scatterers")


# End class ObjCrystScattererParSet


class ObjCrystAtomParSet(ObjCrystScattererParSet):
    """Adapt a pyobjcryst.atom.Atom to the ParameterSet interface.

    This class derives from ObjCrystScattererParSet.

    Attributes
    ----------
    scatterer : pyobjcryst.atom.Atom
        The adapted atom.
    element : str
        The non-refinable name of the element (property).
    parent : ObjCrystCrystalParSet
        The crystal ParameterSet this belongs to.
    occ : ParameterAdapter
        The occupancy of the atom on its crystal location.
    Biso : ParameterAdapter
        The isotropic displacement factor of the atom.
    Bij : ParameterAdapter or ParameterProxy
        The anisotropic displacement factors B11, B22, B33, B12, B21, B13,
        B31, B23 and B32 of the atom. The Bij and Bji parameters are the
        same.
    """

    def __init__(self, name, atom, parent):
        """Initialize the atom ParameterSet.

        Parameters
        ----------
        name : str
            The name of the atom.
        atom : pyobjcryst.atom.Atom
            The atom to adapt.
        parent : ObjCrystCrystalParSet
            The crystal ParameterSet this belongs to.
        """
        ObjCrystScattererParSet.__init__(self, name, atom, parent)
        sp = atom.GetScatteringPower()

        # The B-parameters
        self.add_parameter(ParameterAdapter("Biso", sp, attr="Biso"))
        self.add_parameter(ParameterAdapter("B11", sp, attr="B11"))
        self.add_parameter(ParameterAdapter("B22", sp, attr="B22"))
        self.add_parameter(ParameterAdapter("B33", sp, attr="B33"))
        B12 = ParameterAdapter("B12", sp, attr="B12")
        B21 = ParameterProxy("B21", B12)
        B13 = ParameterAdapter("B13", sp, attr="B13")
        B31 = ParameterProxy("B31", B13)
        B23 = ParameterAdapter("B23", sp, attr="B23")
        B32 = ParameterProxy("B32", B23)
        self.add_parameter(B12)
        self.add_parameter(B21)
        self.add_parameter(B13)
        self.add_parameter(B31)
        self.add_parameter(B23)
        self.add_parameter(B32)

        # Give a value to Biso if it doesn't have one, and this is isotropic
        if sp.IsIsotropic() and self.Biso.value == 0:
            self.Biso.value = 0.5
        return

    def _getelem(self):
        """Getter for the element type."""
        return self.scatterer.GetScatteringPower().GetSymbol()

    element = property(_getelem)


# End class ObjCrystAtomParSet


class ObjCrystMoleculeParSet(ObjCrystScattererParSet):
    """Adapt a pyobjcryst.molecule.Molecule to the ParameterSet
    interface.

    This class derives from ObjCrystScattererParSet. Other attributes are
    inherited from diffpy.srfit.fitbase.parameterset.ParameterSet.

    Attributes
    ----------
    scatterer : pyobjcryst.molecule.Molecule
        The adapted molecule.
    structure : pyobjcryst.molecule.Molecule
        The adapted molecule.
    parent : ObjCrystCrystalParSet or None
        The crystal ParameterSet this belongs to. This is None when the
        ObjCrystMoleculeParSet is used on its own.
    atoms : list of ObjCrystMolAtomParSet
        The ParameterSets of the atoms in the molecule.
    occ : ParameterAdapter
        The occupancy of the molecule on its crystal location.
    q0, q1, q2, q3 : ParameterAdapter
        The orientational quaternion of the molecule.
    """

    def __init__(self, name, molecule, parent=None):
        """Initialize the molecule ParameterSet.

        Parameters
        ----------
        name : str
            The name of the molecule.
        molecule : pyobjcryst.molecule.Molecule
            The molecule to adapt.
        parent : ObjCrystCrystalParSet, optional
            The crystal ParameterSet this belongs to (default None).

        Raises
        ------
        AttributeError
            If a MolAtom in the molecule has no name, or if two MolAtoms
            share a name. Give every MolAtom a unique name before
            wrapping the molecule.
        """
        ObjCrystScattererParSet.__init__(self, name, molecule, parent)
        self.structure = molecule

        # Add orientation quaternion
        self.add_parameter(ParameterAdapter("q0", self.scatterer, attr="Q0"))
        self.add_parameter(ParameterAdapter("q1", self.scatterer, attr="Q1"))
        self.add_parameter(ParameterAdapter("q2", self.scatterer, attr="Q2"))
        self.add_parameter(ParameterAdapter("q3", self.scatterer, attr="Q3"))

        # Wrap the MolAtoms within the molecule
        self.atoms = []
        anames = []

        for a in molecule:

            name = a.GetName()
            if not name:
                raise AttributeError("Each MolAtom must have a name")
            if name in anames:
                raise AttributeError("MolAtom name '%s' is duplicated" % name)

            atom = ObjCrystMolAtomParSet(name, a, self)
            atom.molecule = self
            self.add_parameter_set(atom)
            self.atoms.append(atom)
            anames.append(name)

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
            The flag indicating if `structure` is a pyobjcryst Molecule.
        """
        from pyobjcryst.molecule import Molecule

        return isinstance(structure, Molecule)

    # Part of SrRealParSet interface
    def use_symmetry(self, use=True):
        """Set whether this structure uses symmetry.

        This structure object does not support symmetry, so this does
        nothing.

        Parameters
        ----------
        use : bool, optional
            The flag indicating if symmetry is used (default True).
        """
        return

    # Part of SrRealParSet interface
    def using_symmetry(self):
        """Return whether symmetry is being used.

        Returns
        -------
        bool
            The flag indicating if symmetry is used. Always False, since
            this structure object does not support symmetry.
        """
        return False

    # Part of SrRealParSet interface
    def _get_srreal_structure(self):
        """Get the structure object for use with SrReal calculators.

        Molecule objects are never periodic. Return the object and let
        the SrReal adapters do the proper thing.
        """
        return self.structure

    def get_lattice(self):
        """Return a ParameterSet holding a unit cubic lattice.

        A molecule is not periodic, so this returns a new ParameterSet
        with a = b = c = 1 and alpha = beta = gamma = 90 degrees.

        Returns
        -------
        ParameterSet
            The ParameterSet holding the placeholder lattice Parameters.
        """
        lattice = ParameterSet("lattice")
        lattice.new_parameter("a", 1.0)
        lattice.new_parameter("b", 1.0)
        lattice.new_parameter("c", 1.0)
        lattice.new_parameter("alpha", 90)
        lattice.new_parameter("beta", 90)
        lattice.new_parameter("gamma", 90)
        lattice.angle_units = "deg"
        return lattice

    def get_scatterers(self):
        """Return the list of ParameterSets that represent the
        scatterers.

        Returns
        -------
        list of ObjCrystMolAtomParSet
            The atom ParameterSets of the molecule.
        """
        return self.atoms

    def wrap_restraints(self):
        """Wrap the restraints implicit to the molecule.

        This wraps the MolBonds, MolBondAngles and MolDihedralAngles of
        the Molecule as ObjCrystMoleculeRestraint objects. Restraints
        wrapped this way cannot be modified from within this class.
        """
        # Wrap restraints. Restraints wrapped in this way cannot be modified
        # from within this class.
        for b in self.scatterer.GetBondList():
            restraint = ObjCrystMoleculeRestraint(b)
            self._restraints.add(restraint)

        for ba in self.scatterer.GetBondAngleList():
            restraint = ObjCrystMoleculeRestraint(ba)
            self._restraints.add(restraint)

        for da in self.scatterer.GetDihedralAngleList():
            restraint = ObjCrystMoleculeRestraint(da)
            self._restraints.add(restraint)

        return

    def wrap_stretch_mode_parameters(self):
        """Wrap the stretch modes implicit to the Molecule as
        Parameters.

        This wraps the StretchModeBondLengths and StretchModeBondAngles
        of the Molecule as Parameters. The MolBondAtoms in the Molecule
        must have unique names. Torsion angles are not wrapped, as there
        is not enough information to determine each MolAtom in the
        angle.

        Each Parameter is named after its constituent atoms, as
        "bl_aname1_aname2" for bond lengths and
        "ba_aname1_aname2_aname3" for bond angles.
        """
        for mode in self.scatterer.GetStretchModeBondLengthList():
            name1 = mode.mpAtom0.GetName()
            name2 = mode.mpAtom1.GetName()

            name = "bl_" + "_".join((name1, name2))

            atom1 = getattr(self, name1)
            atom2 = getattr(self, name2)

            parameter = ObjCrystBondLengthParameter(
                name, atom1, atom2, mode=mode
            )

            atoms = []
            for a in mode.GetAtoms():
                name = a.GetName()
                atoms.append(getattr(self, name))

            parameter.AddAtoms(atoms)

            self.add_parameter(parameter)

        for mode in self.scatterer.GetStretchModeBondAngleList():
            name1 = mode.mpAtom0.GetName()
            name2 = mode.mpAtom1.GetName()
            name3 = mode.mpAtom2.GetName()

            name = "ba_" + "_".join((name1, name2, name3))

            atom1 = getattr(self, name1)
            atom2 = getattr(self, name2)
            atom3 = getattr(self, name3)

            parameter = ObjCrystBondAngleParameter(
                name, atom1, atom2, atom3, mode=mode
            )

            atoms = []
            for a in mode.GetAtoms():
                name = a.GetName()
                atoms.append(getattr(self, name))
            parameter.AddAtoms(atoms)

            self.add_parameter(parameter)

        return

    def restrain_bond_length(
        self, atom1, atom2, length, sigma, delta, scaled=False
    ):
        """Add a bond length restraint between two atoms.

        This creates an ObjCrystBondLengthRestraint and adds it to the
        ObjCrystMoleculeParSet.

        Parameters
        ----------
        atom1 : ObjCrystMolAtomParSet
            The first atom in the bond.
        atom2 : ObjCrystMolAtomParSet
            The second atom in the bond.
        length : float
            The length of the bond in Angstroms.
        sigma : float
            The uncertainty of the bond length in Angstroms.
        delta : float
            The width of the bond in Angstroms.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).

        Returns
        -------
        ObjCrystBondLengthRestraint
            The restraint, for use with the ``unrestrain`` method.
        """
        restraint = ObjCrystBondLengthRestraint(
            atom1, atom2, length, sigma, delta, scaled
        )
        self._restraints.add(restraint)

        return restraint

    def restrain_bond_length_parameter(
        self, parameter, length, sigma, delta, scaled=False
    ):
        """Add a bond length restraint on a bond length Parameter.

        This creates an ObjCrystBondLengthRestraint between the atoms of
        `parameter` and adds it to the ObjCrystMoleculeParSet.

        Parameters
        ----------
        parameter : ObjCrystBondLengthParameter
            The bond length Parameter to restrain (see
            add_bond_length_parameter).
        length : float
            The length of the bond in Angstroms.
        sigma : float
            The uncertainty of the bond length in Angstroms.
        delta : float
            The width of the bond in Angstroms.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).

        Returns
        -------
        ObjCrystBondLengthRestraint
            The restraint, for use with the ``unrestrain`` method.
        """
        return self.restrain_bond_length(
            parameter.atom1, parameter.atom2, length, sigma, delta, scaled
        )

    def restrain_bond_angle(
        self, atom1, atom2, atom3, angle, sigma, delta, scaled=False
    ):
        """Add a bond angle restraint between three atoms.

        This creates an ObjCrystBondAngleRestraint and adds it to the
        ObjCrystMoleculeParSet.

        Parameters
        ----------
        atom1 : ObjCrystMolAtomParSet
            The first atom in the bond angle.
        atom2 : ObjCrystMolAtomParSet
            The second (central) atom in the bond angle.
        atom3 : ObjCrystMolAtomParSet
            The third atom in the bond angle.
        angle : float
            The bond angle in radians.
        sigma : float
            The uncertainty of the bond angle in radians.
        delta : float
            The width of the bond angle in radians.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).

        Returns
        -------
        ObjCrystBondAngleRestraint
            The restraint, for use with the ``unrestrain`` method.
        """
        restraint = ObjCrystBondAngleRestraint(
            atom1, atom2, atom3, angle, sigma, delta, scaled
        )
        self._restraints.add(restraint)

        return restraint

    def restrain_bond_angle_parameter(
        self, parameter, angle, sigma, delta, scaled=False
    ):
        """Add a bond angle restraint on a bond angle Parameter.

        This creates an ObjCrystBondAngleRestraint between the atoms of
        `parameter` and adds it to the ObjCrystMoleculeParSet.

        Parameters
        ----------
        parameter : ObjCrystBondAngleParameter
            The bond angle Parameter to restrain (see
            add_bond_angle_parameter).
        angle : float
            The bond angle in radians.
        sigma : float
            The uncertainty of the bond angle in radians.
        delta : float
            The width of the bond angle in radians.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).

        Returns
        -------
        ObjCrystBondAngleRestraint
            The restraint, for use with the ``unrestrain`` method.
        """
        return self.restrain_bond_angle(
            parameter.atom1,
            parameter.atom2,
            parameter.atom3,
            angle,
            sigma,
            delta,
            scaled,
        )

    def restrain_dihedral_angle(
        self, atom1, atom2, atom3, atom4, angle, sigma, delta, scaled=False
    ):
        """Add a dihedral angle restraint between four atoms.

        This creates an ObjCrystDihedralAngleRestraint and adds it to the
        ObjCrystMoleculeParSet.

        Parameters
        ----------
        atom1 : ObjCrystMolAtomParSet
            The first atom in the angle.
        atom2 : ObjCrystMolAtomParSet
            The second (central) atom in the angle.
        atom3 : ObjCrystMolAtomParSet
            The third (central) atom in the angle.
        atom4 : ObjCrystMolAtomParSet
            The fourth atom in the angle.
        angle : float
            The dihedral angle in radians.
        sigma : float
            The uncertainty of the dihedral angle in radians.
        delta : float
            The width of the dihedral angle in radians.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).

        Returns
        -------
        ObjCrystDihedralAngleRestraint
            The restraint, for use with the ``unrestrain`` method.
        """
        restraint = ObjCrystDihedralAngleRestraint(
            atom1, atom2, atom3, atom4, angle, sigma, delta, scaled
        )
        self._restraints.add(restraint)

        return restraint

    def restrain_dihedral_angle_parameter(
        self, parameter, angle, sigma, delta, scaled=False
    ):
        """Add a dihedral angle restraint on a dihedral angle Parameter.

        This creates an ObjCrystDihedralAngleRestraint between the atoms of
        `parameter` and adds it to the ObjCrystMoleculeParSet.

        Parameters
        ----------
        parameter : ObjCrystDihedralAngleParameter
            The dihedral angle Parameter to restrain (see
            add_dihedral_angle_parameter).
        angle : float
            The dihedral angle in radians.
        sigma : float
            The uncertainty of the dihedral angle in radians.
        delta : float
            The width of the dihedral angle in radians.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).

        Returns
        -------
        ObjCrystDihedralAngleRestraint
            The restraint, for use with the ``unrestrain`` method.
        """
        return self.restrain_dihedral_angle(
            parameter.atom1,
            parameter.atom2,
            parameter.atom3,
            parameter.atom4,
            angle,
            sigma,
            delta,
            scaled,
        )

    def add_bond_length_parameter(
        self, name, atom1, atom2, value=None, const=False
    ):
        """Add a refinable bond length to the Molecule.

        This adds an ObjCrystBondLengthParameter to the
        ObjCrystMoleculeParSet that can be adjusted during the fit.

        Parameters
        ----------
        name : str
            The name of the new Parameter.
        atom1 : ObjCrystMolAtomParSet
            The first atom in the bond.
        atom2 : ObjCrystMolAtomParSet
            The second (mutated) atom in the bond.
        value : float, optional
            The initial bond length. If None (default), the current
            distance between the atoms is used.
        const : bool, optional
            The flag indicating whether the Parameter is constant
            (default False).

        Returns
        -------
        ObjCrystBondLengthParameter
            The new bond length Parameter.
        """
        parameter = ObjCrystBondLengthParameter(
            name, atom1, atom2, value, const
        )
        self.add_parameter(parameter)

        return parameter

    def add_bond_angle_parameter(
        self, name, atom1, atom2, atom3, value=None, const=False
    ):
        """Add a refinable bond angle to the Molecule.

        This adds an ObjCrystBondAngleParameter to the
        ObjCrystMoleculeParSet that can be adjusted during the fit.

        Parameters
        ----------
        name : str
            The name of the new Parameter.
        atom1 : ObjCrystMolAtomParSet
            The first atom in the bond angle.
        atom2 : ObjCrystMolAtomParSet
            The second (central) atom in the bond angle.
        atom3 : ObjCrystMolAtomParSet
            The third (mutated) atom in the bond angle.
        value : float, optional
            The initial bond angle in radians. If None (default), the
            current bond angle between the atoms is used.
        const : bool, optional
            The flag indicating whether the Parameter is constant
            (default False).

        Returns
        -------
        ObjCrystBondAngleParameter
            The new bond angle Parameter.
        """
        parameter = ObjCrystBondAngleParameter(
            name, atom1, atom2, atom3, value, const
        )
        self.add_parameter(parameter)

        return parameter

    def add_dihedral_angle_parameter(
        self, name, atom1, atom2, atom3, atom4, value=None, const=False
    ):
        """Add a refinable dihedral angle to the Molecule.

        This adds an ObjCrystDihedralAngleParameter to the
        ObjCrystMoleculeParSet that can be adjusted during the fit.

        Parameters
        ----------
        name : str
            The name of the new Parameter.
        atom1 : ObjCrystMolAtomParSet
            The first atom in the dihedral angle.
        atom2 : ObjCrystMolAtomParSet
            The second (central) atom in the dihedral angle.
        atom3 : ObjCrystMolAtomParSet
            The third (central) atom in the dihedral angle.
        atom4 : ObjCrystMolAtomParSet
            The fourth (mutated) atom in the dihedral angle.
        value : float, optional
            The initial dihedral angle in radians. If None (default), the
            current dihedral angle between the atoms is used.
        const : bool, optional
            The flag indicating whether the Parameter is constant
            (default False).

        Returns
        -------
        ObjCrystDihedralAngleParameter
            The new dihedral angle Parameter.
        """
        parameter = ObjCrystDihedralAngleParameter(
            name, atom1, atom2, atom3, atom4, value, const
        )
        self.add_parameter(parameter)

        return parameter


# End class ObjCrystMoleculeParSet


class ObjCrystMolAtomParSet(ObjCrystScattererParSet):
    """Adapt a pyobjcryst.molecule.MolAtom to the ParameterSet
    interface.

    This class derives from ObjCrystScattererParSet. MolAtom does not
    derive from Scatterer, but the relevant interface is the same within
    pyobjcryst. See the ParameterSet class for base attributes.

    Attributes
    ----------
    scatterer : pyobjcryst.molecule.MolAtom
        The adapted MolAtom.
    parent : ObjCrystMoleculeParSet
        The molecule ParameterSet this belongs to.
    element : str
        The non-refinable name of the element, or "dummy" for a dummy
        atom (property).
    occ : ParameterAdapter
        The occupancy of the atom on its crystal location.
    Biso : ParameterAdapter
        The isotropic displacement factor of the atom. This does not
        exist for dummy atoms; see the ``is_dummy`` method.
    Bij : ParameterAdapter or ParameterProxy
        The anisotropic displacement factors B11, B22, B33, B12, B21, B13,
        B31, B23 and B32 of the atom. The Bij and Bji parameters are the
        same. These do not exist for dummy atoms.
    """

    def __init__(self, name, scatterer, parent):
        """Initialize the MolAtom ParameterSet.

        Parameters
        ----------
        name : str
            The name of the atom.
        scatterer : pyobjcryst.molecule.MolAtom
            The MolAtom to adapt.
        parent : ObjCrystMoleculeParSet
            The molecule ParameterSet this belongs to.
        """
        ObjCrystScattererParSet.__init__(self, name, scatterer, parent)
        sp = scatterer.GetScatteringPower()

        # Only wrap this if there is a scattering power
        if sp is not None:
            self.add_parameter(ParameterAdapter("Biso", sp, attr="Biso"))
            self.add_parameter(ParameterAdapter("B11", sp, attr="B11"))
            self.add_parameter(ParameterAdapter("B22", sp, attr="B22"))
            self.add_parameter(ParameterAdapter("B33", sp, attr="B33"))
            B12 = ParameterAdapter("B12", sp, attr="B12")
            B21 = ParameterProxy("B21", B12)
            B13 = ParameterAdapter("B13", sp, attr="B13")
            B31 = ParameterProxy("B31", B13)
            B23 = ParameterAdapter("B23", sp, attr="B23")
            B32 = ParameterProxy("B32", B23)
            self.add_parameter(B12)
            self.add_parameter(B21)
            self.add_parameter(B13)
            self.add_parameter(B31)
            self.add_parameter(B23)
            self.add_parameter(B32)

        return

    def _getelem(self):
        """Getter for the element type."""
        sp = self.scatterer.GetScatteringPower()
        if sp:
            return sp.GetSymbol()
        else:
            return "dummy"

    element = property(_getelem)

    def is_dummy(self):
        """Return whether this atom is a dummy atom.

        Returns
        -------
        bool
            The flag indicating if this is a dummy atom.
        """
        return self.scatterer.IsDummy()


# End class ObjCrystMolAtomParSet


class ObjCrystMoleculeRestraint(object):
    """Base class for adapting pyobjcryst Molecule restraints to srfit.

    This implements the ``penalty`` method of
    diffpy.srfit.fitbase.restraint.Restraint by calling
    ``GetLogLikelihood`` of the pyobjcryst restraint. The ``restrain``
    method is not needed or implemented.

    Attributes
    ----------
    restraint : object
        The pyobjcryst Molecule restraint.
    scaled : bool
        The flag indicating if the restraint is scaled (multiplied) by
        the unrestrained point-average chi^2 (chi^2/numpoints) (default
        False).
    """

    def __init__(self, restraint, scaled=False):
        """Wrap a pyobjcryst Molecule restraint as a Restraint.

        Parameters
        ----------
        restraint : object
            The pyobjcryst Molecule restraint.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).
        """
        self.restraint = restraint
        self.scaled = scaled
        return

    def penalty(self, w=1.0):
        """Calculate the penalty of the restraint.

        Parameters
        ----------
        w : float, optional
            The point-average chi^2 which is optionally used to scale the
            penalty (default 1.0).

        Returns
        -------
        float
            The log-likelihood of the pyobjcryst restraint, optionally
            scaled by `w`.
        """
        penalty = self.restraint.GetLogLikelihood()
        if self.scaled:
            penalty *= w
        return penalty


# End class ObjCrystMoleculeRestraint


class ObjCrystBondLengthRestraint(ObjCrystMoleculeRestraint):
    """Restrain the distance between two atoms.

    Attributes
    ----------
    atom1 : ObjCrystMolAtomParSet
        The first atom in the bond.
    atom2 : ObjCrystMolAtomParSet
        The second atom in the bond.
    length : float
        The length of the bond in Angstroms.
    sigma : float
        The uncertainty of the bond length in Angstroms.
    delta : float
        The width of the bond in Angstroms.
    restraint : pyobjcryst.molecule.MolBond
        The pyobjcryst bond length restraint.
    scaled : bool
        The flag indicating if the restraint is scaled (multiplied) by
        the unrestrained point-average chi^2 (chi^2/numpoints) (default
        False).
    """

    def __init__(self, atom1, atom2, length, sigma, delta, scaled=False):
        """Initialize the bond length restraint.

        Parameters
        ----------
        atom1 : ObjCrystMolAtomParSet
            The first atom in the bond.
        atom2 : ObjCrystMolAtomParSet
            The second atom in the bond.
        length : float
            The length of the bond in Angstroms.
        sigma : float
            The uncertainty of the bond length in Angstroms.
        delta : float
            The width of the bond in Angstroms.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).
        """
        self.atom1 = atom1
        self.atom2 = atom2

        m = self.atom1.scatterer.GetMolecule()
        restraint = m.AddBond(
            atom1.scatterer, atom2.scatterer, length, sigma, delta
        )

        ObjCrystMoleculeRestraint.__init__(self, restraint, scaled)
        return

    # Give access to the parameters of the restraint
    length = property(
        lambda self: self.restraint.GetLength0(),
        lambda self, value: self.restraint.SetLength0(value),
    )
    sigma = property(
        lambda self: self.restraint.GetLengthSigma(),
        lambda self, value: self.restraint.SetLengthSigma(value),
    )
    delta = property(
        lambda self: self.restraint.GetLengthDelta(),
        lambda self, value: self.restraint.SetLengthDelta(value),
    )


# End class ObjCrystBondLengthRestraint


class ObjCrystBondAngleRestraint(ObjCrystMoleculeRestraint):
    """Restrain the angle defined by three atoms.

    Attributes
    ----------
    atom1 : ObjCrystMolAtomParSet
        The first atom in the angle.
    atom2 : ObjCrystMolAtomParSet
        The second (central) atom in the angle.
    atom3 : ObjCrystMolAtomParSet
        The third atom in the angle.
    angle : float
        The bond angle in radians.
    sigma : float
        The uncertainty of the bond angle in radians.
    delta : float
        The width of the bond angle in radians.
    restraint : pyobjcryst.molecule.MolBondAngle
        The pyobjcryst bond angle restraint.
    scaled : bool
        The flag indicating if the restraint is scaled (multiplied) by
        the unrestrained point-average chi^2 (chi^2/numpoints) (default
        False).
    """

    def __init__(self, atom1, atom2, atom3, angle, sigma, delta, scaled=False):
        """Initialize the bond angle restraint.

        Parameters
        ----------
        atom1 : ObjCrystMolAtomParSet
            The first atom in the bond angle.
        atom2 : ObjCrystMolAtomParSet
            The second (central) atom in the bond angle.
        atom3 : ObjCrystMolAtomParSet
            The third atom in the bond angle.
        angle : float
            The bond angle in radians.
        sigma : float
            The uncertainty of the bond angle in radians.
        delta : float
            The width of the bond angle in radians.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).
        """
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3

        m = self.atom1.scatterer.GetMolecule()
        restraint = m.AddBondAngle(
            atom1.scatterer,
            atom2.scatterer,
            atom3.scatterer,
            angle,
            sigma,
            delta,
        )

        ObjCrystMoleculeRestraint.__init__(self, restraint, scaled)
        return

    # Give access to the parameters of the restraint
    angle = property(
        lambda self: self.restraint.GetAngle0(),
        lambda self, value: self.restraint.SetAngle0(value),
    )
    sigma = property(
        lambda self: self.restraint.GetAngleSigma(),
        lambda self, value: self.restraint.SetAngleSigma(value),
    )
    delta = property(
        lambda self: self.restraint.GetAngleDelta(),
        lambda self, value: self.restraint.SetAngleDelta(value),
    )


# End class ObjCrystBondAngleRestraint


class ObjCrystDihedralAngleRestraint(ObjCrystMoleculeRestraint):
    """Restrain the dihedral (torsion) angle defined by four atoms.

    Attributes
    ----------
    atom1 : ObjCrystMolAtomParSet
        The first atom in the angle.
    atom2 : ObjCrystMolAtomParSet
        The second (central) atom in the angle.
    atom3 : ObjCrystMolAtomParSet
        The third (central) atom in the angle.
    atom4 : ObjCrystMolAtomParSet
        The fourth atom in the angle.
    angle : float
        The dihedral angle in radians.
    sigma : float
        The uncertainty of the dihedral angle in radians.
    delta : float
        The width of the dihedral angle in radians.
    restraint : pyobjcryst.molecule.MolDihedralAngle
        The pyobjcryst dihedral angle restraint.
    scaled : bool
        The flag indicating if the restraint is scaled (multiplied) by
        the unrestrained point-average chi^2 (chi^2/numpoints) (default
        False).
    """

    def __init__(
        self, atom1, atom2, atom3, atom4, angle, sigma, delta, scaled=False
    ):
        """Initialize the dihedral angle restraint.

        Parameters
        ----------
        atom1 : ObjCrystMolAtomParSet
            The first atom in the angle.
        atom2 : ObjCrystMolAtomParSet
            The second (central) atom in the angle.
        atom3 : ObjCrystMolAtomParSet
            The third (central) atom in the angle.
        atom4 : ObjCrystMolAtomParSet
            The fourth atom in the angle.
        angle : float
            The dihedral angle in radians.
        sigma : float
            The uncertainty of the dihedral angle in radians.
        delta : float
            The width of the dihedral angle in radians.
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).
        """
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4

        m = self.atom1.scatterer.GetMolecule()
        restraint = m.AddDihedralAngle(
            atom1.scatterer,
            atom2.scatterer,
            atom3.scatterer,
            atom4.scatterer,
            angle,
            sigma,
            delta,
        )

        ObjCrystMoleculeRestraint.__init__(self, restraint, scaled)
        return

    # Give access to the parameters of the restraint
    angle = property(
        lambda self: self.restraint.GetAngle0(),
        lambda self, value: self.restraint.SetAngle0(value),
    )
    sigma = property(
        lambda self: self.restraint.GetAngleSigma(),
        lambda self, value: self.restraint.SetAngleSigma(value),
    )
    delta = property(
        lambda self: self.restraint.GetAngleDelta(),
        lambda self, value: self.restraint.SetAngleDelta(value),
    )


# End class ObjCrystDihedralAngleRestraint


class StretchModeParameter(Parameter):
    """Partial Parameter class encapsulating pyobjcryst stretch modes.

    This class relies upon attributes that subclasses must set before
    calling ``StretchModeParameter.__init__``. Do not instantiate this
    class directly.

    Attributes
    ----------
    mutated_atoms : set of ObjCrystMolAtomParSet
        The set of all mutated atoms. Set by the subclass.
    molecule : ObjCrystMoleculeParSet
        The molecule the atoms belong to. Set by the subclass.
    mode : pyobjcryst.molecule.StretchMode
        The stretch mode used to change atomic positions. Set by the
        subclass.
    keepcenter : bool
        The flag indicating whether to keep the center of mass of the
        molecule stationary within the crystal when changing the value
        of the parameter (default True).
    """

    def __init__(self, name, value=None, const=False):
        """Initialize the stretch mode Parameter.

        Parameters
        ----------
        name : str
            The name of this Parameter. It must be a valid attribute
            identifier.
        value : float, optional
            The initial value of this Parameter (default None).
        const : bool, optional
            The flag indicating whether the Parameter is a constant
            (default False).

        Raises
        ------
        ValueError
            If `name` is not a valid attribute identifier.
        """
        Parameter.__init__(self, name, value, const)
        self.keepcenter = True

    def set_value(self, value):
        """Set the value of the Parameter by stretching the molecule.

        The stretch mode moves the mutated atoms by the change in value.

        Parameters
        ----------
        value : float
            The new value of the Parameter.

        Returns
        -------
        StretchModeParameter
            Return self so that mutators can be chained.
        """
        curval = self.get_value()
        value = float(value)

        if value == curval:
            return self

        # The StretchMode expects the change in mutated value.
        delta = value - curval
        self.mode.Stretch(delta, self.keepcenter)

        # Let Parameter take care of the general details
        Parameter.set_value(self, value)

        return self

    def add_atoms(self, atomlist):
        """Associate additional atoms with the Parameter.

        The added atoms are mutated in exactly the same way as the
        primary mutated atom. This is useful when a group of atoms should
        move rigidly in response to a change in a bond property.

        Parameters
        ----------
        atomlist : ObjCrystMolAtomParSet or list of ObjCrystMolAtomParSet
            The atom or atoms to associate with the Parameter.

        Returns
        -------
        StretchModeParameter
            Return self so that mutators can be chained.
        """
        if not hasattr(atomlist, "__iter__"):
            atomlist = [atomlist]
        # Record the added atoms in the Parameter
        self.mutated_atoms.update(atomlist)
        # Make sure we're observing these atoms
        for a in atomlist:
            a.x.addObserver(self._flush)
            a.y.addObserver(self._flush)
            a.z.addObserver(self._flush)

        # Record the added atoms in the StretchMode
        scatlist = [a.scatterer for a in atomlist]
        self.mode.AddAtoms(scatlist)
        return self

    def notify(self, other=()):
        """Notify all mutated Parameters and observers.

        Some of the mutated Parameters observe this Parameter while this
        Parameter also observes them. Observable does not allow both, so
        the mutated Parameters are notified directly.

        Parameters
        ----------
        other : tuple, optional
            The objects that have already been notified (default empty).
        """
        noneother = ()
        # Notify the atoms that have moved
        for a in self.mutated_atoms:
            a.x._flush(noneother)
            a.y._flush(noneother)
            a.z._flush(noneother)
        # Notify the molecule position
        self.molecule.x._flush(noneother)
        self.molecule.y._flush(noneother)
        self.molecule.z._flush(noneother)

        # Notify observers
        Parameter.notify(self, other)
        return


# End class StretchModeParameter


class ObjCrystBondLengthParameter(StretchModeParameter):
    """Represent a bond length in a Molecule as a Parameter.

    This wraps up a pyobjcryst.molecule.StretchModeBondLength object so that
    the distance between two MolAtoms in a Molecule can be used as an
    adjustable Parameter. When a bond length is adjusted, the second MolAtom is
    moved, and the absolute position of the Molecule is altered to preserve the
    location of the center of mass within the Crystal. Thus, the x, y and z
    Parameters of the MolAtom and its parent Molecule are altered. This can be
    changed by setting the 'keepcenter' attribute of the parameter to False.

    This Parameter makes it possible to mutate a MolAtom multiple times in a
    single refinement step. If these mutations are not orthogonal, then this
    could lead to nonconvergence of a fit, depending on the optimizer. Consider
    mutating atom2 of a bond directly, and via a ObjCrystBondLengthParameter.
    The two mutations of atom2 may be determined independently by the
    optimizer, in which case the composed mutation will have an unexpected
    effect on the residual. It is best practice to either modify MolAtom
    positions directly, or thorough BondLengthParameters, BondAngleParameters
    and DihedralAngleParameters (which are mutually orthogonal).

    Note that by making a ObjCrystBondLengthParameter constant it also makes
    the underlying ObjCrystMolAtomParSets constant. When setting it as
    nonconstant, each ObjCrystMolAtomParSet is set nonconstant.  Changing the
    bond length changes the position of the second MolAtom and Molecule, even
    if either is set as constant.

    Attributes
    ----------
    atom1 : ObjCrystMolAtomParSet
        The first atom in the bond.
    atom2 : ObjCrystMolAtomParSet
        The second (mutated) atom in the bond.
    mutated_atoms : set of ObjCrystMolAtomParSet
        The set of all mutated atoms.
    molecule : ObjCrystMoleculeParSet
        The molecule the atoms belong to.
    mode : pyobjcryst.molecule.StretchModeBondLength
        The stretch mode for the bond.
    name : str
        The name of this Parameter (inherited).
    const : bool
        The flag indicating whether this is considered a constant
        (inherited).
    value : float
        The property for ``get_value`` and ``set_value`` (inherited).
    constraint : callable or None
        The callable that calculates the value of this Parameter. If
        None, the Parameter is responsible for its own value
        (inherited).
    bounds : list of float
        The lower and upper bounds on the Parameter, which some
        optimizers use when the Parameter is varied (inherited).
    """

    def __init__(self, name, atom1, atom2, value=None, const=False, mode=None):
        """Initialize the bond length Parameter.

        Parameters
        ----------
        name : str
            The name of the Parameter.
        atom1 : ObjCrystMolAtomParSet
            The first atom in the bond.
        atom2 : ObjCrystMolAtomParSet
            The second (mutated) atom in the bond.
        value : float, optional
            The initial bond length. If None (default), the current
            distance between the atoms is used.
        const : bool, optional
            The flag indicating whether the Parameter is constant
            (default False).
        mode : pyobjcryst.molecule.StretchModeBondLength, optional
            The existing stretch mode to use. If None (default), a new
            StretchModeBondLength is built.
        """
        # Create the mode
        self.mode = mode
        if mode is None:
            self.mode = StretchModeBondLength(
                atom1.scatterer, atom2.scatterer, None
            )
        # We only add the last atom. This is the one that will move
        self.mode.AddAtom(atom2.scatterer)
        self.mutated_atoms = set([atom2])

        # Observe the atom positions
        for a in [atom1, atom2]:
            a.x.addObserver(self._flush)
            a.y.addObserver(self._flush)
            a.z.addObserver(self._flush)

        self.atom1 = atom1
        self.atom2 = atom2
        self.molecule = atom1.parent

        # We do this last so the atoms are defined before we set any values.
        if value is None:
            value = GetBondLength(atom1.scatterer, atom2.scatterer)
        StretchModeParameter.__init__(self, name, value, const)
        self.set_constant(const)

        return

    def set_constant(self, is_constant=True, value=None):
        """Toggle the Parameter as constant.

        This sets the underlying ObjCrystMolAtomParSet positions
        constant as well.

        Parameters
        ----------
        is_constant : bool, optional
            The flag indicating if the Parameter is constant (default
            True).
        value : float, optional
            The value to set the Parameter to (default None). If this is
            not None, the Parameter gets a new value, constant or
            otherwise.

        Returns
        -------
        StretchModeParameter
            Return self so that mutators can be chained.
        """
        StretchModeParameter.set_constant(self, is_constant, value)

        for a in [self.atom1, self.atom2]:
            a.x.set_constant(is_constant)
            a.y.set_constant(is_constant)
            a.z.set_constant(is_constant)
        return self

    def get_value(self):
        """Return the bond length, recalculating it if needed.

        The atoms underlying the bond may have moved, so the bond length
        is recalculated whenever the cached value has been cleared.

        Returns
        -------
        float
            The bond length in Angstroms.
        """
        if self._value is None:
            value = GetBondLength(self.atom1.scatterer, self.atom2.scatterer)
            Parameter.set_value(self, value)

        return self._value


# End class ObjCrystBondLengthParameter


class ObjCrystBondAngleParameter(StretchModeParameter):
    """Represent a bond angle in a Molecule as a Parameter.

    This wraps up a pyobjcryst.molecule.StretchModeBondAngle object so that the
    angle defined by three MolAtoms in a Molecule can be used as an adjustable
    Parameter. When a bond angle is adjusted, the third MolAtom is moved, and
    the absolute position of the Molecule is altered to preserve the location
    of the center of mass within the crystal. This can be changed by setting
    the 'keepcenter' attribute of the parameter to False.

    See precautions in the ObjCrystBondLengthParameter class.

    Attributes
    ----------
    atom1 : ObjCrystMolAtomParSet
        The first atom in the bond angle.
    atom2 : ObjCrystMolAtomParSet
        The second (central) atom in the bond angle.
    atom3 : ObjCrystMolAtomParSet
        The third (mutated) atom in the bond angle.
    mutated_atoms : set of ObjCrystMolAtomParSet
        The set of all mutated atoms.
    molecule : ObjCrystMoleculeParSet
        The molecule the atoms belong to.
    mode : pyobjcryst.molecule.StretchModeBondAngle
        The stretch mode for the bond angle.
    name : str
        The name of this Parameter (inherited).
    const : bool
        The flag indicating whether this is considered a constant
        (inherited).
    value : float
        The property for ``get_value`` and ``set_value`` (inherited).
    constraint : callable or None
        The callable that calculates the value of this Parameter. If
        None, the Parameter is responsible for its own value
        (inherited).
    bounds : list of float
        The lower and upper bounds on the Parameter, which some
        optimizers use when the Parameter is varied (inherited).
    """

    def __init__(
        self, name, atom1, atom2, atom3, value=None, const=False, mode=None
    ):
        """Initialize the bond angle Parameter.

        Parameters
        ----------
        name : str
            The name of the Parameter.
        atom1 : ObjCrystMolAtomParSet
            The first atom in the bond angle.
        atom2 : ObjCrystMolAtomParSet
            The second (central) atom in the bond angle.
        atom3 : ObjCrystMolAtomParSet
            The third (mutated) atom in the bond angle.
        value : float, optional
            The initial bond angle in radians. If None (default), the
            current bond angle between the atoms is used.
        const : bool, optional
            The flag indicating whether the Parameter is constant
            (default False).
        mode : pyobjcryst.molecule.StretchModeBondAngle, optional
            The existing stretch mode to use. If None (default), a new
            StretchModeBondAngle is built.
        """
        # Create the stretch mode
        self.mode = mode
        if mode is None:
            self.mode = StretchModeBondAngle(
                atom1.scatterer, atom2.scatterer, atom3.scatterer, None
            )
        # We only add the last atom. This is the one that will move
        self.mode.AddAtom(atom3.scatterer)
        self.mutated_atoms = set([atom3])

        # Observe the atom positions
        for a in [atom1, atom2, atom3]:
            a.x.addObserver(self._flush)
            a.y.addObserver(self._flush)
            a.z.addObserver(self._flush)

        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.molecule = atom1.parent

        # We do this last so the atoms are defined before we set any values.
        if value is None:
            value = GetBondAngle(
                atom1.scatterer, atom2.scatterer, atom3.scatterer
            )
        StretchModeParameter.__init__(self, name, value, const)
        self.set_constant(const)

        return

    def set_constant(self, is_constant=True, value=None):
        """Toggle the Parameter as constant.

        This sets the underlying ObjCrystMolAtomParSet positions
        constant as well.

        Parameters
        ----------
        is_constant : bool, optional
            The flag indicating if the Parameter is constant (default
            True).
        value : float, optional
            The value to set the Parameter to (default None). If this is
            not None, the Parameter gets a new value, constant or
            otherwise.

        Returns
        -------
        StretchModeParameter
            Return self so that mutators can be chained.
        """
        StretchModeParameter.set_constant(self, is_constant, value)
        for a in [self.atom1, self.atom2, self.atom3]:
            a.x.set_constant(is_constant)
            a.y.set_constant(is_constant)
            a.z.set_constant(is_constant)
        return self

    def get_value(self):
        """Return the bond angle, recalculating it if needed.

        The atoms underlying the bond angle may have moved, so the angle
        is recalculated whenever the cached value has been cleared.

        Returns
        -------
        float
            The bond angle in radians.
        """
        if self._value is None:
            value = GetBondAngle(
                self.atom1.scatterer,
                self.atom2.scatterer,
                self.atom3.scatterer,
            )
            Parameter.set_value(self, value)

        return self._value


# End class ObjCrystBondAngleParameter


class ObjCrystDihedralAngleParameter(StretchModeParameter):
    """Represent a dihedral angle in a Molecule as a Parameter.

    This wraps up a pyobjcryst.molecule.StretchModeTorsion object so that the
    angle defined by four MolAtoms ([a1-a2].[a3-a4]) in a Molecule can be used
    as an adjustable parameter. When a dihedral angle is adjusted, the fourth
    MolAtom is moved, and the absolute position of the Molecule is altered to
    preserve the location of the center of mass within the crystal.  This can
    be changed by setting the 'keepcenter' attribute of the parameter to False.

    See precautions in the ObjCrystBondLengthParameter class.

    Attributes
    ----------
    atom1 : ObjCrystMolAtomParSet
        The first atom in the dihedral angle.
    atom2 : ObjCrystMolAtomParSet
        The second (central) atom in the dihedral angle.
    atom3 : ObjCrystMolAtomParSet
        The third (central) atom in the dihedral angle.
    atom4 : ObjCrystMolAtomParSet
        The fourth (mutated) atom in the dihedral angle.
    mutated_atoms : set of ObjCrystMolAtomParSet
        The set of all mutated atoms.
    molecule : ObjCrystMoleculeParSet
        The molecule the atoms belong to.
    mode : pyobjcryst.molecule.StretchModeTorsion
        The stretch mode for the dihedral angle.
    name : str
        The name of this Parameter (inherited).
    const : bool
        The flag indicating whether this is considered a constant
        (inherited).
    value : float
        The property for ``get_value`` and ``set_value`` (inherited).
    constraint : callable or None
        The callable that calculates the value of this Parameter. If
        None, the Parameter is responsible for its own value
        (inherited).
    bounds : list of float
        The lower and upper bounds on the Parameter, which some
        optimizers use when the Parameter is varied (inherited).
    """

    def __init__(
        self,
        name,
        atom1,
        atom2,
        atom3,
        atom4,
        value=None,
        const=False,
        mode=None,
    ):
        """Initialize the dihedral angle Parameter.

        Parameters
        ----------
        name : str
            The name of the Parameter.
        atom1 : ObjCrystMolAtomParSet
            The first atom in the dihedral angle.
        atom2 : ObjCrystMolAtomParSet
            The second (central) atom in the dihedral angle.
        atom3 : ObjCrystMolAtomParSet
            The third (central) atom in the dihedral angle.
        atom4 : ObjCrystMolAtomParSet
            The fourth (mutated) atom in the dihedral angle.
        value : float, optional
            The initial dihedral angle in radians. If None (default), the
            current dihedral angle between the atoms is used.
        const : bool, optional
            The flag indicating whether the Parameter is constant
            (default False).
        mode : pyobjcryst.molecule.StretchModeTorsion, optional
            The existing stretch mode to use. If None (default), a new
            StretchModeTorsion is built.
        """
        # Create the stretch mode
        self.mode = mode
        if mode is None:
            self.mode = StretchModeTorsion(
                atom2.scatterer, atom3.scatterer, None
            )
        # We only add the last atom. This is the one that will move
        self.mode.AddAtom(atom4.scatterer)
        self.mutated_atoms = set([atom4])

        # Observe the atom positions
        for a in [atom1, atom2, atom3, atom4]:
            a.x.addObserver(self._flush)
            a.y.addObserver(self._flush)
            a.z.addObserver(self._flush)

        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.atom4 = atom4
        self.molecule = atom1.parent

        # We do this last so the atoms are defined before we set any values.
        if value is None:
            value = GetDihedralAngle(
                atom1.scatterer,
                atom2.scatterer,
                atom3.scatterer,
                atom4.scatterer,
            )
        StretchModeParameter.__init__(self, name, value, const)
        self.set_constant(const)

        return

    def set_constant(self, is_constant=True, value=None):
        """Toggle the Parameter as constant.

        This sets the underlying ObjCrystMolAtomParSet positions
        constant as well.

        Parameters
        ----------
        is_constant : bool, optional
            The flag indicating if the Parameter is constant (default
            True).
        value : float, optional
            The value to set the Parameter to (default None). If this is
            not None, the Parameter gets a new value, constant or
            otherwise.

        Returns
        -------
        StretchModeParameter
            Return self so that mutators can be chained.
        """
        StretchModeParameter.set_constant(self, is_constant, value)
        for a in [self.atom1, self.atom2, self.atom3, self.atom4]:
            a.x.set_constant(is_constant)
            a.y.set_constant(is_constant)
            a.z.set_constant(is_constant)
        return self

    def get_value(self):
        """Return the dihedral angle, recalculating it if needed.

        The atoms underlying the dihedral angle may have been moved by
        another Parameter, so the angle is recalculated whenever the
        cached value has been cleared.

        Returns
        -------
        float
            The dihedral angle in radians.
        """
        if self._value is None:
            value = GetDihedralAngle(
                self.atom1.scatterer,
                self.atom2.scatterer,
                self.atom3.scatterer,
                self.atom4.scatterer,
            )
            Parameter.set_value(self, value)

        return self._value


# End class ObjCrystDihedralAngleParameter


class ObjCrystCrystalParSet(SrRealParSet):
    """Adapt a pyobjcryst.crystal.Crystal to the ParameterSet interface.

    This class derives from SrRealParSet. See that class for base
    attributes.

    Attributes
    ----------
    structure : pyobjcryst.crystal.Crystal
        The adapted crystal.
    scatterers : list of ObjCrystAtomParSet or ObjCrystMoleculeParSet
        The scatterer ParameterSets, provided for convenience.
    space_group_parameters : SpaceGroupParameters
        The free structure Parameters after applying the crystal's space
        group constraints, created when first accessed. See the
        diffpy.cmistructure.sgconstraints module.
    angle_units : str
        The units of the lattice angles, always "rad".
    a, b, c, alpha, beta, gamma : ParameterAdapter
        The unit cell parameters.
    """

    def __init__(self, name, crystal):
        """Initialize the crystal ParameterSet.

        Parameters
        ----------
        name : str
            The name of this ParameterSet.
        crystal : pyobjcryst.crystal.Crystal
            The crystal to adapt.

        Raises
        ------
        ValueError
            If a scatterer in the crystal has no name, or if two
            scatterers share a name. Give every scatterer a unique name
            before wrapping the crystal.
        TypeError
            If the crystal contains a scatterer that is neither an Atom
            nor a Molecule.
        """
        SrRealParSet.__init__(self, name)
        self.angle_units = "rad"
        self.structure = crystal
        self._space_group_parameters = None

        self.add_parameter(ParameterAdapter("a", self.structure, attr="a"))
        self.add_parameter(ParameterAdapter("b", self.structure, attr="b"))
        self.add_parameter(ParameterAdapter("c", self.structure, attr="c"))
        self.add_parameter(
            ParameterAdapter("alpha", self.structure, attr="alpha")
        )
        self.add_parameter(
            ParameterAdapter("beta", self.structure, attr="beta")
        )
        self.add_parameter(
            ParameterAdapter("gamma", self.structure, attr="gamma")
        )

        # Now we must loop over the scatterers and create parameter sets from
        # them.
        self.scatterers = []
        snames = []

        for j in range(self.structure.GetNbScatterer()):
            s = self.structure.GetScatt(j)
            name = s.GetName()
            if not name:
                raise ValueError("Each Scatterer must have a name")
            if name in snames:
                raise ValueError("Scatterer name '%s' is duplicated" % name)

            # Now create the proper object
            cname = s.GetClassName()
            if cname == "Atom":
                parameter_set = ObjCrystAtomParSet(name, s, self)
            elif cname == "Molecule":
                parameter_set = ObjCrystMoleculeParSet(name, s, self)
            else:
                raise TypeError("Unrecognized scatterer '%s'" % cname)

            self.add_parameter_set(parameter_set)
            self.scatterers.append(parameter_set)
            snames.append(name)

        return

    def _constrain_space_group(self):
        """Constrain the space group."""
        if self._space_group_parameters is not None:
            return self._space_group_parameters
        space_group = self._create_space_group(self.structure.GetSpaceGroup())
        from diffpy.cmistructure.sgconstraints import (
            _constrain_as_space_group,
        )

        adpsymbols = ["B11", "B22", "B33", "B12", "B13", "B23"]
        isosymbol = "Biso"
        sgoffset = [0, 0, 0]
        self._space_group_parameters = _constrain_as_space_group(
            self,
            space_group,
            self.scatterers,
            sgoffset,
            adpsymbols=adpsymbols,
            isosymbol=isosymbol,
        )
        return self._space_group_parameters

    space_group_parameters = property(_constrain_space_group)

    @staticmethod
    def _create_space_group(sgobjcryst):
        """Create a diffpy.structure SpaceGroup object from pyobjcryst.

        Parameters
        ----------
        sgobjcryst
            A pyobjcryst.spacegroup.SpaceGroup instance.

        This uses the actual space group operations from the
        pyobjcryst.spacegroup.SpaceGroup instance so there is no ambiguity
        about the actual space group.
        """
        import copy

        from diffpy.structure.spacegroups import SymOp, get_space_group

        name = sgobjcryst.GetName()
        extnstr = ":%s" % sgobjcryst.GetExtension()
        if name.endswith(extnstr):
            name = name[: -len(extnstr)]

        # Get whatever spacegroup we can get by name. This will set the proper
        # crystal system. Creating a copy of the singleton from
        # get_space_group, as this function messes with symop_list.
        space_group = copy.copy(get_space_group(name))

        # Replace the symmetry operations to guarantee that we get it right.
        symops = sgobjcryst.GetSymmetryOperations()
        tranops = sgobjcryst.GetTranslationVectors()
        space_group.symop_list = []

        for trans in tranops:
            for shift, rot in symops:
                tv = trans + shift
                tv -= numpy.floor(tv)
                space_group.symop_list.append(SymOp(rot, tv))

        if sgobjcryst.IsCentrosymmetric():
            center = sgobjcryst.GetInversionCenter()
            for trans in tranops:
                for shift, rot in symops:
                    tv = center - trans - shift
                    tv -= numpy.floor(tv)
                    space_group.symop_list.append(SymOp(-rot, tv))

        return space_group

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
            The flag indicating if `structure` is a pyobjcryst Crystal.
        """
        from pyobjcryst.crystal import Crystal

        return isinstance(structure, Crystal)

    def get_lattice(self):
        """Return the ParameterSet containing the lattice Parameters.

        Returns
        -------
        ObjCrystCrystalParSet
            This ParameterSet, which holds the lattice Parameters
            directly.
        """
        return self

    def get_scatterers(self):
        """Return the list of ParameterSets that represent the
        scatterers.

        Returns
        -------
        list of ObjCrystAtomParSet or ObjCrystMoleculeParSet
            The scatterer ParameterSets of the crystal.
        """
        return self.scatterers


# End class ObjCrystCrystalParSet
