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
"""Base class for adapting structures to a ParameterSet interface.

The BaseStructureParSet is a ParameterSet with functionality required by
all structure adapters.
"""

__all__ = ["BaseStructureParSet"]

from diffpy.srfit.fitbase.parameterset import ParameterSet


class BaseStructureParSet(ParameterSet):
    """Base class for structure adapters.

    BaseStructureParSet derives from ParameterSet and provides methods that
    help interface the ParameterSet with the space group constraint methods in
    the sgconstraints module and to ProfileGenerators.

    Attributes
    ----------
    structure : object
        The adapted structure object.
    """

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
            The flag indicating if `structure` can be adapted. The base class
            always returns False.
        """
        return False

    def get_lattice(self):
        """Return the ParameterSet containing the lattice Parameters.

        The returned ParameterSet may contain other Parameters than the
        lattice Parameters. It is assumed that the lattice parameters
        are named "a", "b", "c", "alpha", "beta", "gamma".

        The lattice must also have the "angle_units" attribute, which is
        either "deg" or "rad", to signify degrees or radians.

        Returns
        -------
        ParameterSet
            The ParameterSet holding the lattice Parameters.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("The must be overloaded")

    def get_scatterers(self):
        """Return the list of ParameterSets that represent the
        scatterers.

        The site positions must be accessible from the list entries via
        the names "x", "y", and "z". The ADPs must be accessible as
        well, but the name and nature of the ADPs (U-factors, B-factors,
        isotropic, anisotropic) depends on the adapted structure.

        Returns
        -------
        list of ParameterSet
            The ParameterSets of the scatterers in the structure.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("The must be overloaded")
