:tocdepth: -1

.. index:: making-a-structure-adapter

.. _making-a-structure-adapter:

==========================
Making a structure adapter
==========================

A structure adapter is a ``ParameterSet`` that wraps a structure object, so
that the lattice parameters, atomic positions and displacement parameters of
that object become refinable srfit ``Parameter`` objects. When a ``Parameter``
changes during a refinement, the adapter writes the new value straight into
the wrapped structure object.

``diffpy.cmistructure`` ships adapters for these structure representations:

.. list-table::
   :header-rows: 1

   * - Structure object
     - Adapter
   * - ``diffpy.structure.Structure``
     - ``diffpy.cmistructure.diffpyparset.DiffpyStructureParSet``
   * - ``pyobjcryst.crystal.Crystal``
     - ``diffpy.cmistructure.objcrystparset.ObjCrystCrystalParSet``
   * - ``pyobjcryst.molecule.Molecule``
     - ``diffpy.cmistructure.objcrystparset.ObjCrystMoleculeParSet``
   * - ``cctbx.crystal.special_position_settings``
     - ``diffpy.cmistructure.cctbxparset.CCTBXCrystalParSet``

If your structures live in some other representation, for example a class
from another package or one written for your own project, write an adapter
for it. The adapter then works with ``constrain_as_space_group`` and, if it
derives from ``SrRealParSet``, with the PDF generators in ``diffpy.cmipdf``.

What an adapter must provide
----------------------------

An adapter is a subclass of one of two base classes:

``BaseStructureParSet``
    The minimal base class. Use it when the adapter only needs to work with
    space group constraints.

``SrRealParSet``
    A subclass of ``BaseStructureParSet`` that adds what the
    ``diffpy.srreal`` calculators need: ``use_symmetry()``,
    ``using_symmetry()``, ``restrain_bvs()`` and ``_get_srreal_structure()``.
    Use it when the structure will be used to calculate a PDF or other
    srreal quantity. This is the usual choice.

The rest of the package relies on the following parts of an adapter:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Requirement
   * - ``structure``
     - The wrapped structure object.
   * - ``can_adapt(structure)``
     - A classmethod that returns True if ``structure`` is of the type this
       adapter wraps, and False otherwise. It must not raise if an optional
       package is missing; return False instead.
   * - ``get_lattice()``
     - Return a ``ParameterSet`` holding ``Parameter`` objects named ``a``,
       ``b``, ``c``, ``alpha``, ``beta`` and ``gamma``. That ``ParameterSet``
       must have an ``angle_units`` attribute set to ``"deg"`` or ``"rad"``.
   * - ``get_scatterers()``
     - Return a list of ``ParameterSet`` objects, one per atom or scatterer.
       Each holds the fractional coordinates as ``Parameter`` objects named
       ``x``, ``y`` and ``z``, plus its displacement parameters (see
       :ref:`adapter-adps`).
   * - ``_get_srreal_structure()``
     - ``SrRealParSet`` only. Return an object that the ``diffpy.srreal``
       calculators can read (see :ref:`adapter-srreal`).

Example: adapting a simple crystal class
----------------------------------------

The example below adapts a minimal crystal representation, ``SimpleCrystal``,
which stores the unit cell as a list of six numbers and the atoms as a list
of ``SimpleSite`` objects. Swap these two classes for the structure classes
you actually use.

.. code-block:: python

    from operator import getitem, setitem

    from diffpy.cmistructure.srrealparset import SrRealParSet
    from diffpy.srfit.fitbase.parameter import ParameterAdapter
    from diffpy.srfit.fitbase.parameterset import ParameterSet
    from diffpy.srfit.util.argbinders import bind2nd


    # The structure representation being adapted.
    class SimpleSite:
        def __init__(self, element, xyz, uiso):
            self.element = element
            self.xyz = list(xyz)
            self.uiso = uiso


    class SimpleCrystal:
        def __init__(self, cell, sites):
            self.cell = list(cell)
            self.sites = list(sites)


    # The adapter.
    def _add_list_item_parameters(parameter_set, names, values):
        for index, name in enumerate(names):
            parameter_set.add_parameter(
                ParameterAdapter(
                    name,
                    values,
                    bind2nd(getitem, index),
                    bind2nd(setitem, index),
                )
            )


    class SimpleCellParSet(ParameterSet):
        def __init__(self, cell):
            ParameterSet.__init__(self, "lattice")
            self.angle_units = "deg"
            _add_list_item_parameters(
                self, ("a", "b", "c", "alpha", "beta", "gamma"), cell
            )


    class SimpleSiteParSet(ParameterSet):
        def __init__(self, name, site):
            ParameterSet.__init__(self, name)
            self.site = site
            _add_list_item_parameters(self, ("x", "y", "z"), site.xyz)
            self.add_parameter(ParameterAdapter("Uiso", site, attr="uiso"))

        @property
        def element(self):
            return self.site.element


    class SimpleCrystalParSet(SrRealParSet):
        def __init__(self, name, structure):
            SrRealParSet.__init__(self, name)
            self.structure = structure
            self.add_parameter_set(SimpleCellParSet(structure.cell))
            self.sites = []
            element_counts = {}
            for site in structure.sites:
                index = element_counts.get(site.element, 0)
                element_counts[site.element] = index + 1
                site_parameter_set = SimpleSiteParSet(
                    f"{site.element}{index}", site
                )
                self.add_parameter_set(site_parameter_set)
                self.sites.append(site_parameter_set)

        @classmethod
        def can_adapt(cls, structure):
            return isinstance(structure, SimpleCrystal)

        def get_lattice(self):
            return self.lattice

        def get_scatterers(self):
            return self.sites

        def _get_srreal_structure(self):
            from diffpy.srreal.structureadapter import nosymmetry
            from diffpy.structure import Atom, Lattice, Structure

            atoms = [
                Atom(site.element, site.xyz, Uisoequiv=site.uiso)
                for site in self.structure.sites
            ]
            lattice = Lattice(*self.structure.cell)
            diffpy_structure = Structure(atoms, lattice=lattice)
            if self.using_symmetry():
                return diffpy_structure
            return nosymmetry(diffpy_structure)

The sections below explain each part.

Wrap each value with ``ParameterAdapter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``ParameterAdapter`` is a ``Parameter`` that does not store its value.
Instead it reads and writes the value in the wrapped object every time, so
the structure object and the ``Parameter`` always agree. There are two ways
to create one:

- If the value is an attribute of an object, pass the attribute name, as in
  ``ParameterAdapter("Uiso", site, attr="uiso")``.
- Otherwise, pass a getter and a setter. The getter is called as
  ``getter(obj)`` and returns the value. The setter is called as
  ``setter(obj, value)``. In the example, ``bind2nd(getitem, index)`` and
  ``bind2nd(setitem, index)`` read and write one item of a list.

Keep a reference to the live object, never a copy of its values. If you
wrap a copy, the refinement changes the copy and the structure object never
sees the new values.

The lattice
^^^^^^^^^^^

``get_lattice()`` returns a ``ParameterSet`` with ``Parameter`` objects
named ``a``, ``b``, ``c``, ``alpha``, ``beta`` and ``gamma``.
``constrain_as_space_group`` looks the lattice parameters up by these names
and fixes the angles to 90 or 120 for the higher-symmetry crystal systems.
It reads the ``angle_units`` attribute to know whether to use degrees
(``"deg"``) or radians (``"rad"``).

The ``ParameterSet`` in the example is named ``"lattice"``, so the adapter
can return it as ``self.lattice``.

The scatterers
^^^^^^^^^^^^^^

``get_scatterers()`` returns one ``ParameterSet`` per atom, in a fixed
order. ``constrain_as_space_group`` reads the positions from the ``x``,
``y`` and ``z`` parameters of each one, so these must be fractional
coordinates.

Each scatterer is added to the adapter with ``add_parameter_set``, which
makes it reachable as an attribute of the adapter, for example
``phase.Ni0``. The names must therefore be valid Python identifiers and
unique within the structure. The built-in adapters use the element symbol
followed by a running index for that element, starting from zero: ``Ni0``,
``Ni1``, ``O0`` and so on.

An ``element`` property is not required by ``constrain_as_space_group``,
but the built-in adapters provide one, and users expect it.

.. _adapter-adps:

Displacement parameters
^^^^^^^^^^^^^^^^^^^^^^^

``constrain_as_space_group`` constrains the atomic displacement parameters
(ADPs) only for scatterers that have all six anisotropic parameters. By
default it looks for parameters named ``U11``, ``U22``, ``U33``, ``U12``,
``U13`` and ``U23``. It then uses the isotropic parameter, by default
``Uiso``, for sites whose symmetry makes them isotropic. It leaves
scatterers that lack any of the six anisotropic parameters out of the ADP
constraints, without raising an error.

The ``SimpleSite`` class in the example has only an isotropic ``uiso``. The
space group constraints therefore leave the ADPs alone, and you constrain
``Uiso`` yourself in the fit recipe. If your structure stores anisotropic
ADPs, add all six as parameters. ``DiffpyAtomParSet`` in
``diffpy.cmistructure.diffpyparset`` is a complete example.

If your structure uses different names, for example B-factors, pass them
to ``constrain_as_space_group`` with the ``adpsymbols`` and ``isosymbol``
arguments instead of renaming the parameters:

.. code-block:: python

    constrain_as_space_group(
        phase,
        "Fm-3m",
        adpsymbols=["B11", "B22", "B33", "B12", "B13", "B23"],
        isosymbol="Biso",
    )

.. _adapter-srreal:

The structure seen by srreal calculators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PDF generators do not read the adapter's ``Parameter`` objects. Before every
calculation they call ``_get_srreal_structure()`` and pass the result to a
``diffpy.srreal`` calculator. The default in ``SrRealParSet`` returns
``self.structure``, which works only if srreal already understands that
type, as it does for ``diffpy.structure.Structure`` and the ``pyobjcryst``
classes.

For any other type you have two options:

- Convert the structure to one that srreal understands, as the example does
  by building a ``diffpy.structure.Structure`` from the current values. This
  is the simplest option. The conversion runs on every calculation, which is
  fast enough for small structures.
- Register a converter for your type with
  ``diffpy.srreal.structureadapter.RegisterStructureAdapter``, so that
  srreal can read the structure directly. See the ``diffpy.srreal``
  documentation for details.

In both cases, wrap the result with
``diffpy.srreal.structureadapter.nosymmetry`` when ``self.using_symmetry()``
is False, so that the generator's ``periodic`` setting is respected.

Use the adapter
---------------

Create the adapter from a structure object and use it like a built-in
adapter. Here the example adapter is used to constrain a nickel unit cell,
given in P1 with four atoms, to the space group Fm-3m:

.. code-block:: python

    from diffpy.cmistructure import constrain_as_space_group

    positions = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    crystal = SimpleCrystal(
        [3.52, 3.52, 3.52, 90, 90, 90],
        [SimpleSite("Ni", xyz, 0.005) for xyz in positions],
    )
    phase = SimpleCrystalParSet("nickel", crystal)

    space_group_parameters = constrain_as_space_group(phase, "Fm-3m")
    print([parameter.name for parameter in space_group_parameters.latpars])

which prints:

.. code-block:: text

    ['a']

For a cubic space group only ``a`` is left free. ``b`` and ``c`` are
constrained to it, and the angles are fixed to 90 degrees.

To calculate a PDF from the adapter, pass it to the generator's
``set_structure_from_parset`` method instead of passing a structure object
to ``set_structure``:

.. code-block:: python

    from diffpy.cmipdf.pdfgenerator import PDFGenerator

    generator = PDFGenerator("nickel_pdf")
    generator.set_structure_from_parset(phase)

Add the adapter to ``structure_to_parameter_set``
-------------------------------------------------

``diffpy.cmistructure.structure_to_parameter_set``, which is also what a
generator's ``set_structure`` method uses, only knows about the adapters
listed at the top of this page. For any other type it raises
``TypeError("Unadaptable structure format")``. Outside this package, create
your adapter directly as shown above.

If you are adding an adapter to ``diffpy.cmistructure`` itself:

#. Put the adapter in a new module, ``src/diffpy/cmistructure/<name>parset.py``,
   with the package header and NumPy-style docstrings used by the other
   adapter modules.
#. Add a ``can_adapt`` branch to ``structure_to_parameter_set`` in
   ``src/diffpy/cmistructure/__init__.py``. Import the adapter inside the
   function, as the existing branches do, so that the optional package it
   depends on is imported only when it is needed.
#. Add tests in ``tests/test_<name>parset.py``. At a minimum, cover
   ``can_adapt`` for supported and unsupported objects, that changing each
   ``Parameter`` changes the wrapped structure, and that
   ``constrain_as_space_group`` produces the expected free parameters.
#. Add the module to ``docs/source/api/diffpy.cmistructure.rst`` and a news
   item to ``news/``.

Checklist
---------

- The adapter subclasses ``SrRealParSet``, or ``BaseStructureParSet`` if it
  is never used with srreal calculators.
- ``structure`` holds the wrapped object, and every ``ParameterAdapter``
  reads and writes that object, not a copy.
- ``can_adapt`` returns False, without raising, for unsupported objects.
- ``get_lattice()`` returns parameters named ``a``, ``b``, ``c``,
  ``alpha``, ``beta`` and ``gamma``, and has ``angle_units`` set.
- ``get_scatterers()`` returns one ``ParameterSet`` per atom, with
  fractional ``x``, ``y`` and ``z`` and uniquely named.
- The ADPs use the default ``U11``…``U23`` and ``Uiso`` names, or you pass
  ``adpsymbols`` and ``isosymbol`` to ``constrain_as_space_group``.
- ``_get_srreal_structure()`` returns an object srreal can read and
  respects ``using_symmetry()``.
