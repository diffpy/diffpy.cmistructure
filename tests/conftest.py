import importlib.resources
import io
import json
import sys
from pathlib import Path

import numpy
import pytest
from pyobjcryst.atom import Atom
from pyobjcryst.crystal import Crystal
from pyobjcryst.molecule import Molecule
from pyobjcryst.scatteringpower import ScatteringPowerAtom

from diffpy.cmistructure.diffpyparset import DiffpyStructureParSet
from diffpy.structure import Lattice, Structure

C60_XYZ = """\
3.451266498   0.685000000   0.000000000
3.451266498  -0.685000000   0.000000000
-3.451266498   0.685000000   0.000000000
-3.451266498  -0.685000000   0.000000000
0.685000000   0.000000000   3.451266498
-0.685000000   0.000000000   3.451266498
0.685000000   0.000000000  -3.451266498
-0.685000000   0.000000000  -3.451266498
0.000000000   3.451266498   0.685000000
0.000000000   3.451266498  -0.685000000
0.000000000  -3.451266498   0.685000000
0.000000000  -3.451266498  -0.685000000
3.003809890   1.409000000   1.171456608
3.003809890   1.409000000  -1.171456608
3.003809890  -1.409000000   1.171456608
3.003809890  -1.409000000  -1.171456608
-3.003809890   1.409000000   1.171456608
-3.003809890   1.409000000  -1.171456608
-3.003809890  -1.409000000   1.171456608
-3.003809890  -1.409000000  -1.171456608
1.409000000   1.171456608   3.003809890
1.409000000  -1.171456608   3.003809890
-1.409000000   1.171456608   3.003809890
-1.409000000  -1.171456608   3.003809890
1.409000000   1.171456608  -3.003809890
1.409000000  -1.171456608  -3.003809890
-1.409000000   1.171456608  -3.003809890
-1.409000000  -1.171456608  -3.003809890
1.171456608   3.003809890   1.409000000
-1.171456608   3.003809890   1.409000000
1.171456608   3.003809890  -1.409000000
-1.171456608   3.003809890  -1.409000000
1.171456608  -3.003809890   1.409000000
-1.171456608  -3.003809890   1.409000000
1.171456608  -3.003809890  -1.409000000
-1.171456608  -3.003809890  -1.409000000
2.580456608   0.724000000   2.279809890
2.580456608   0.724000000  -2.279809890
2.580456608  -0.724000000   2.279809890
2.580456608  -0.724000000  -2.279809890
-2.580456608   0.724000000   2.279809890
-2.580456608   0.724000000  -2.279809890
-2.580456608  -0.724000000   2.279809890
-2.580456608  -0.724000000  -2.279809890
0.724000000   2.279809890   2.580456608
0.724000000  -2.279809890   2.580456608
-0.724000000   2.279809890   2.580456608
-0.724000000  -2.279809890   2.580456608
0.724000000   2.279809890  -2.580456608
0.724000000  -2.279809890  -2.580456608
-0.724000000   2.279809890  -2.580456608
-0.724000000  -2.279809890  -2.580456608
2.279809890   2.580456608   0.724000000
-2.279809890   2.580456608   0.724000000
2.279809890   2.580456608  -0.724000000
-2.279809890   2.580456608  -0.724000000
2.279809890  -2.580456608   0.724000000
-2.279809890  -2.580456608   0.724000000
2.279809890  -2.580456608  -0.724000000
-2.279809890  -2.580456608  -0.724000000
"""


@pytest.fixture
def user_filesystem(tmp_path):
    base_dir = Path(tmp_path)
    home_dir = base_dir / "home_dir"
    home_dir.mkdir(parents=True, exist_ok=True)
    cwd_dir = base_dir / "cwd_dir"
    cwd_dir.mkdir(parents=True, exist_ok=True)

    home_config_data = {"username": "home_username", "email": "home@email.com"}
    with open(home_dir / "diffpyconfig.json", "w") as f:
        json.dump(home_config_data, f)

    yield tmp_path


@pytest.fixture(scope="session")
def datafile():
    """Fixture to load a test data file from the testdata package
    directory."""

    def _datafile(filename):
        return importlib.resources.files("tests.testdata").joinpath(filename)

    return _datafile


@pytest.fixture(scope="session")
def capturestdout():
    def _capturestdout(f, *args, **kwargs):
        """Capture the standard output from a call of function f."""
        savestdout = sys.stdout
        fp = io.StringIO()
        try:
            sys.stdout = fp
            f(*args, **kwargs)
        finally:
            sys.stdout = savestdout
        return fp.getvalue()

    return _capturestdout


class GeneratorStandIn:
    """Stand in for a generator, such as a PDFGenerator, that holds a
    structure parameter set as ``.phase``."""

    def __init__(self, phase):
        self.phase = phase


@pytest.fixture
def lamno3_p1_structure(datafile):
    """Return the full LaMnO3 unit cell in P1 as a diffpy Structure."""
    structure = Structure()
    structure.read(datafile("LaMnO3.stru"))
    return structure


@pytest.fixture
def lamno3_asymmetric_unit_structure(lamno3_p1_structure):
    """Return the four atoms of the LaMnO3 asymmetric unit in Pbnm,
    without the rest of the cell, as a diffpy Structure."""
    structure = lamno3_p1_structure[[0, 4, 8, 12]]
    return structure


@pytest.fixture
def silicon_origin_1_structure():
    """Return the full silicon unit cell, with an atom at the origin, as
    a diffpy Structure."""
    structure = Structure(lattice=Lattice(5.431, 5.431, 5.431, 90, 90, 90))
    positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]
    for xyz in positions:
        structure.add_new_atom("Si", xyz=xyz)
    return structure


@pytest.fixture
def silicon_origin_2_structure():
    """Return the full silicon unit cell, with the origin midway between
    two bonded atoms, as a diffpy Structure."""
    structure = Structure(lattice=Lattice(5.431, 5.431, 5.431, 90, 90, 90))
    positions = [
        [0.125, 0.125, 0.125],
        [0.125, 0.625, 0.625],
        [0.625, 0.125, 0.625],
        [0.625, 0.625, 0.125],
        [0.875, 0.875, 0.875],
        [0.875, 0.375, 0.375],
        [0.375, 0.875, 0.375],
        [0.375, 0.375, 0.875],
    ]
    for xyz in positions:
        structure.add_new_atom("Si", xyz=xyz)
    return structure


@pytest.fixture
def lamno3_generator(lamno3_p1_structure):
    parameter_set = DiffpyStructureParSet("LaMnO3", lamno3_p1_structure)
    generator = GeneratorStandIn(parameter_set)
    return generator


@pytest.fixture
def lamno3_crystal():
    """Return the LaMnO3 asymmetric unit in P b n m as a pyobjcryst
    Crystal."""
    pi = numpy.pi
    # It appears that ObjCryst only supports standard symbols
    crystal = Crystal(5.486341, 5.619215, 7.628206, "P b n m")
    crystal.SetName("LaMnO3")
    # La1
    sp = ScatteringPowerAtom("La1", "La")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.996096, 0.0321494, 0.25, "La1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # Mn1
    sp = ScatteringPowerAtom("Mn1", "Mn")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0, 0.5, 0, "Mn1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # O1
    sp = ScatteringPowerAtom("O1", "O")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.0595746, 0.496164, 0.25, "O1", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    # O2
    sp = ScatteringPowerAtom("O2", "O")
    sp.SetBiso(8 * pi * pi * 0.003)
    atom = Atom(0.720052, 0.289387, 0.0311126, "O2", sp)
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(atom)
    return crystal


@pytest.fixture
def c60_crystal():
    """Return a P1 pyobjcryst Crystal holding the C60 molecule "c60"."""
    pi = numpy.pi
    c = Crystal(100, 100, 100, "P1")
    c.SetName("c60frame")
    m = Molecule(c, "c60")
    c.AddScatterer(m)
    sp = ScatteringPowerAtom("C", "C")
    sp.SetBiso(8 * pi * pi * 0.003)
    for i, l in enumerate(C60_XYZ.strip().splitlines()):
        x, y, z = map(float, l.split())
        m.AddAtom(x, y, z, sp, "C%i" % i)
    return c


@pytest.fixture
def c60_molecule(c60_crystal):
    """Return the C60 Molecule held by the c60_crystal fixture."""
    molecule = c60_crystal.GetScatterer("c60")
    return molecule
