"""
`rodigan` is a FEM software for modelling of special Cosserat rods.

The `common` subpackage contains the general parts of the code.
The module `common.solver` contains an abstract base class.
Therefore `common` cannot do anything useful on its own.

The `cantilever` subpackage extends and uses `common` to solve
a static elasticity problem.
For an example on how to use the package please have a look at the
Jupyter notebook `cantilever_example.ipynb`.
"""

try:
    __RODIGAN_SETUP__
except NameError:
    __RODIGAN_SETUP__ = False

if not __RODIGAN_SETUP__:
    from .common.geometry import Geometry
    from .common.material import Material
    from .cantilever.cantilever import Cantilever

    __all__ = ["Geometry", "Material", "Cantilever"]
