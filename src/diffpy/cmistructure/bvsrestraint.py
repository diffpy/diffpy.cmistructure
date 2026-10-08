#!/usr/bin/env python
##############################################################################
#
# (c) 2010 The Trustees of Columbia University in the City of New York.
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
"""Bond-valence sum calculator from SrReal wrapped as a Restraint.

This can be used as an addition to a cost function during a structure
refinement to keep the bond-valence sum within tolerable limits.
"""

__all__ = ["BVSRestraint"]

from diffpy.srfit.exceptions import SrFitError
from diffpy.srfit.fitbase.restraint import Restraint


class BVSRestraint(Restraint):
    """Wrapping of BVSCalculator.bvmsdiff as a Restraint.

    The restraint penalty is the root-mean-square deviation of the theoretical
    and calculated bond-valence sum of a structure.

    Attributes
    ----------
    _calc : BVSCalculator
        The SrReal BVSCalculator instance.
    _parameter_set : SrRealParSet
        The SrRealParSet that created this BVSRestraint.
    sig : float
        The uncertainty on the BVS (default 1).
    scaled : bool
        The flag indicating if the restraint is scaled (multiplied)
        by the unrestrained point-average chi^2 (chi^2/numpoints)
        (default False).
    """

    def __init__(self, parameter_set, sig=1, scaled=False):
        """Initialize the Restraint.

        Parameters
        ----------
        parameter_set : SrRealParSet
            The SrRealParSet that creates this BVSRestraint.
        sig : float, optional
            The uncertainty on the BVS (default 1).
        scaled : bool, optional
            The flag indicating if the restraint is scaled (multiplied)
            by the unrestrained point-average chi^2 (chi^2/numpoints)
            (default False).
        """
        from diffpy.srreal.bvscalculator import BVSCalculator

        self._calc = BVSCalculator()
        self._parameter_set = parameter_set
        self.sig = float(sig)
        self.scaled = bool(scaled)
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
            The bond-valence penalty.
        """
        # Get the bvms from the BVSCalculator
        structure = self._parameter_set._get_srreal_structure()
        self._calc.eval(structure)
        penalty = self._calc.bvmsdiff

        # Scale by the prefactor
        penalty /= self.sig**2

        # Optionally scale by w
        if self.scaled:
            penalty *= w

        return penalty

    def _validate(self):
        """This evaluates the calculator.

        Raises SrFitError if validation fails.
        """
        from numpy import nan

        p = self.penalty()
        if p is None or p is nan:
            raise SrFitError("Cannot evaluate penalty")
        v = self._calc.value
        if len(v) > 1 and not v.any():
            emsg = (
                "Bond valence sums are all zero.  Check atom symbols in "
                "the structure or define custom bond-valence parameters."
            )
            raise SrFitError(emsg)
        return

    # End of class BVSRestraint


# End of file
