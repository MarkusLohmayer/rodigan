# Finite elements for special Cosserat rods

The `rodigan` package implements a FEM solver for special Cosserat rods.
The subpackage `rodigan.common` contains code that should be reusable for many particular types of problems,
whereas the subpackage `rodigan.cantilever` builds upon `rodigan.common` to solve the static elasticity equations
for a cantilever. A usage example is shown in the Jupyter notebook `cantilever_example.ipynb`.

The code is written in Python. Further it makes use of the packages `numpy`, `matplotlib` and `numba`.

For theory on special Cosserat rods, you may check out https://github.com/MarquitoForrest/Elasticity1D,
where you can find the lecture notes of the course 'Elasticity of one-dimensional continua' held by
Prof. Ajeet Kumar at FAU, Erlangen in summer 2017.
