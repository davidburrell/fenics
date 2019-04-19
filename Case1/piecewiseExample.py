from fenics import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(200,200)
p1 = Expression('x[0]',degree=1,domain=mesh)
p2 = Expression('1-x[0]',degree=1,domain=mesh)

g = Expression('x[0] < 0.25 + DOLFIN_EPS ? p1 : (x[0] < 0.5 + DOLFIN_EPS ? p2 : 0)', p1=p1, p2=p2, degree=1)

V = FunctionSpace(mesh, 'P', 1)
g = interpolate(g,V)
f = interpolate(Expression('x[0] + x[1]', degree=2),V)
#plot(f,title='f')
plot(g,title='g')
vtkfile = File('test/solution.pvd')
vtkfile << g
plt.show()
