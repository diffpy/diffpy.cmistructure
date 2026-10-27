:tocdepth: -1

.. index:: migrating-from-srfit

.. _migrating-from-srfit:

=========================================
Migrating from ``diffpy.srfit.structure``
=========================================

``diffpy.cmistructure`` holds the code that used to live in
``diffpy.srfit.structure``. The behavior is the same, but the names follow
the Python style: camelCase methods are now snake_case, and abbreviated
attribute and argument names are spelled out.

``diffpy.srfit`` 3.5.0 warns when you use an old method or attribute and names
its replacement. **Renamed function arguments can't be warned about**, so check
the :ref:`renamed-arguments` table if your script passes arguments by keyword.

Update your imports
-------------------

Replace ``diffpy.srfit.structure`` with ``diffpy.cmistructure``. The module
names are unchanged:

.. code-block:: python

    # Before
    from diffpy.srfit.structure import constrainAsSpaceGroup
    from diffpy.srfit.structure.objcrystparset import ObjCrystCrystalParSet

    # After
    from diffpy.cmistructure import constrain_as_space_group
    from diffpy.cmistructure.objcrystparset import ObjCrystCrystalParSet

``constrain_as_space_group`` accepts a generator
------------------------------------------------

``constrain_as_space_group`` now also accepts a generator, such as a
``PDFGenerator``, in place of its ``.phase``. Both of these calls work:

.. code-block:: python

    constrain_as_space_group(generator.phase, "Fm-3m")
    constrain_as_space_group(generator, "Fm-3m")

Anything that is neither a structure parameter set nor an object with one as
its ``.phase`` raises a ``TypeError``.

What has not changed
--------------------

- Class names, such as ``DiffpyStructureParSet`` and
  ``ObjCrystCrystalParSet``.
- The space group containers ``.sgpars``, ``.latpars``, ``.xyzpars`` and
  ``.adppars``. They also name Parameters in fit results, for example
  ``sgpars.latpars.a``.
- The ``constrainadps`` and ``periodic`` keyword arguments.

Renamed functions and methods
-----------------------------

.. list-table::
   :header-rows: 1

   * - diffpy.srfit.structure
     - diffpy.cmistructure
   * - ``struToParameterSet(name, stru)``
     - ``structure_to_parameter_set(name, structure)``
   * - ``constrainAsSpaceGroup``
     - ``constrain_as_space_group``
   * - ``canAdapt`` (all adapters)
     - ``can_adapt``
   * - ``getLattice`` (all adapters)
     - ``get_lattice``
   * - ``getScatterers`` (all adapters)
     - ``get_scatterers``
   * - ``CCTBXCrystalParSet.getSpaceGroup``
     - ``get_space_group``
   * - ``SrRealParSet.restrainBVS``
     - ``restrain_bvs``
   * - ``useSymmetry``
     - ``use_symmetry``
   * - ``usingSymmetry``
     - ``using_symmetry``
   * - ``isDummy``
     - ``is_dummy``
   * - ``ObjCrystScattererParSet.hasScatterers``
     - ``has_scatterers``
   * - ``ObjCrystMoleculeParSet.wrapRestraints``
     - ``wrap_restraints``
   * - ``ObjCrystMoleculeParSet.wrapStretchModeParameters``
     - ``wrap_stretch_mode_parameters``
   * - ``ObjCrystMoleculeParSet.restrainBondLength``
     - ``restrain_bond_length``
   * - ``ObjCrystMoleculeParSet.restrainBondLengthParameter``
     - ``restrain_bond_length_parameter``
   * - ``ObjCrystMoleculeParSet.restrainBondAngle``
     - ``restrain_bond_angle``
   * - ``ObjCrystMoleculeParSet.restrainBondAngleParameter``
     - ``restrain_bond_angle_parameter``
   * - ``ObjCrystMoleculeParSet.restrainDihedralAngle``
     - ``restrain_dihedral_angle``
   * - ``ObjCrystMoleculeParSet.restrainDihedralAngleParameter``
     - ``restrain_dihedral_angle_parameter``
   * - ``ObjCrystMoleculeParSet.addBondLengthParameter``
     - ``add_bond_length_parameter``
   * - ``ObjCrystMoleculeParSet.addBondAngleParameter``
     - ``add_bond_angle_parameter``
   * - ``ObjCrystMoleculeParSet.addDihedralAngleParameter``
     - ``add_dihedral_angle_parameter``
   * - ``StretchModeParameter.addAtoms``
     - ``add_atoms``
   * - ``setConst`` on the bond length, bond angle and dihedral angle
       Parameters
     - ``set_constant``
   * - ``BaseSpaceGroupParameters.addParameter``
     - ``add_parameter``

Renamed attributes
------------------

.. list-table::
   :header-rows: 1

   * - Class
     - diffpy.srfit.structure
     - diffpy.cmistructure
   * - ``SrRealParSet``, ``DiffpyStructureParSet``,
       ``ObjCrystCrystalParSet``, ``ObjCrystMoleculeParSet``,
       ``CCTBXCrystalParSet``
     - ``.stru``
     - ``.structure``
   * - ``DiffpyLatticeParSet``, ``ObjCrystCrystalParSet``
     - ``.angunits``
     - ``.angle_units``
   * - ``ObjCrystScattererParSet``
     - ``.scat``
     - ``.scatterer``
   * - ``ObjCrystMoleculeRestraint``
     - ``.res``
     - ``.restraint``
   * - Bond length, bond angle and dihedral angle Parameters
     - ``.matoms``
     - ``.mutated_atoms``
   * - ``SpaceGroupParameters``
     - ``.sg``
     - ``.space_group``
   * - ``CCTBXScattererParSet``, ``CCTBXUnitCellParSet``
     - ``.strups``
     - ``.structure_parameter_set``
   * - ``CCTBXScattererParSet``
     - ``.idx``
     - ``.index``

.. _renamed-arguments:

Renamed arguments
-----------------

These only matter if you pass the argument by keyword. Positional calls keep
working, because no argument has moved.

.. list-table::
   :header-rows: 1

   * - Function or class
     - diffpy.srfit.structure
     - diffpy.cmistructure
   * - ``structure_to_parameter_set``, ``can_adapt``,
       ``DiffpyStructureParSet``, ``CCTBXCrystalParSet``
     - ``stru``
     - ``structure``
   * - ``ObjCrystCrystalParSet``
     - ``cryst``
     - ``crystal``
   * - ``ObjCrystScattererParSet``, ``ObjCrystMolAtomParSet``
     - ``scat``
     - ``scatterer``
   * - ``ObjCrystMoleculeRestraint``
     - ``res``
     - ``restraint``
   * - ``BVSRestraint``
     - ``parset``
     - ``parameter_set``
   * - ``CCTBXScattererParSet``, ``CCTBXUnitCellParSet``
     - ``strups``
     - ``structure_parameter_set``
   * - ``CCTBXScattererParSet``
     - ``idx``
     - ``index``
   * - ``SpaceGroupParameters``
     - ``sg``
     - ``space_group``
   * - ``restrain_bond_length_parameter``,
       ``restrain_bond_angle_parameter``,
       ``restrain_dihedral_angle_parameter``,
       ``BaseSpaceGroupParameters.add_parameter``
     - ``par``
     - ``parameter``
   * - ``StretchModeParameter.set_value``
     - ``val``
     - ``value``
