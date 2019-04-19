from dolfin import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(32,32)

BDM = FiniteElement("BDM", mesh.ufl_cell(),1)
DG = FiniteElement("DG", mesh.ufl_cell(),0)
W = FunctionSpace(mesh, BDM * DG)
(sigma, u) = TrialFunctions(W)
(tau, v) = TestFunctions(W)

f = Expression("10*exp(-(pow(x[0] - 0.5,2) + pow(x[1]-0.5,2)) / 0.02)", degree=2)
a = (dot(sigma,tau) + div(tau)*u + div(sigma)*v)*dx
L = 
