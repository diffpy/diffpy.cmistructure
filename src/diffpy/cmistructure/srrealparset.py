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
"""Structure wrapper class for structures compatible with SrReal."""

__all__ = ["SrRealParSet"]

from diffpy.cmistructure.basestructureparset import BaseStructureParSet
from diffpy.cmistructure.bvsrestraint import BVSRestraint


class SrRealParSet(BaseStructureParSet):
    """Base class for SrReal-compatible structure adapters.

    This derives from BaseStructureParSet and provides some extended
    functionality provided by SrReal.

    Attributes
    ----------
    structure : object
        The adapted structure object.
    _usesymmetry : bool
        The flag indicating if SrReal calculators that operate on
        this object should use symmetry (default True).
    """

    def __init__(self, *args, **kw):
        BaseStructureParSet.__init__(self, *args, **kw)
        self._usesymmetry = True
        self.structure = None
        return

    def restrain_bvs(self, sig=1, scaled=False):
        """Restrain the bond-valence sum to zero.

        This adds a penalty to the cost function equal to
        ``bvmsdiff / sig**2``, where ``bvmsdiff`` is the mean-squared
        difference between the calculated and expected bond valence sums
        for the structure. If `scaled` is True, this is also scaled by the
        current point-averaged chi^2 value so the restraint is roughly
        equally weighted in the fit.

        Parameters
        ----------
        sig : float, optional
            The uncertainty on the BVS (default 1).
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).

        Returns
        -------
        BVSRestraint
            The restraint object, for use with the ``unrestrain`` method.
        """
        # Create the Restraint object
        restraint = BVSRestraint(self, sig, scaled)
        # Add it to the _restraints set
        self._restraints.add(restraint)
        # Our configuration changed. Notify observers.
        self._update_configuration()
        # Return the Restraint object
        return restraint

    def use_symmetry(self, use=True):
        """Set whether this structure uses symmetry.

        This determines how the structure is treated by SrReal
        calculators.

        Parameters
        ----------
        use : bool, optional
            The flag indicating if symmetry is used (default True).
        """
        self._usesymmetry = bool(use)
        return

    def using_symmetry(self):
        """Return whether symmetry is being used.

        Returns
        -------
        bool
            The flag indicating if symmetry is used.
        """
        return self._usesymmetry

    def _get_srreal_structure(self):
        """Get the structure object for use with SrReal calculators.

        If this is periodic, then return the structure, otherwise, pass
        it inside of a nosymmetry wrapper.
        """
        from diffpy.srreal.structureadapter import nosymmetry

        if self._usesymmetry:
            return self.structure
        return nosymmetry(self.structure)
