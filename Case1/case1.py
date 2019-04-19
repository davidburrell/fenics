from fenics import *
import matplotlib.pyplot as plt
import numpy as np
a = -2.0;
b = 1.0;
n = 10.4;
m = 1-1/n;
alpha = 0.79;
tol = 1E-14;
num_steps = 10;

mesh = UnitSquareMesh(20,20)
V = FunctionSpace(mesh,"P",1)
u_k = interpolate(Expression("3*x[0]-2",degree=1),V)
u_X = Expression("b*x[0] - a*(x[0]-1)",degree=1,b=b,a=a)
u_Z = Expression("b*x[0] - a*(x[0]-1)",degree=1,b=b,a=a)

def boundary_X(x, on_boundary):
    return on_boundary and (near(x[0],0,tol) or near(x[0],1,tol))

bc_X = DirichletBC(V, u_X, boundary_X)


def boundary_Z(x, on_boundary):
    return on_boundary and (near(x[1],0,tol) or near(x[0],1,tol))

bc_Z = DirichletBC(V, u_Z, boundary_Z)

boundaryConditions = [bc_X, bc_Z]


def s(h):
    #parameterization of saturation van Genuchten
    if h < 0:
         return (1+(alpha*abs(h))**(n))**(-m)
    else:
        return 1 

def k_r(h):
    return (s(h)**(1/2))*((1-(1-s(h)**(1/m))**m)**2)

class NonLinearCoefficient(UserExpression):
    def eval(self, value, x):
        #print(x)
        if (u_k(x)) < 0:
           #print(u_k(x))
           #print(k_r(abs(u_k(x))))
           value[0]= k_r(u_k(x))
        else:
           value[0]= 1


# Define Variational Problem
u = TrialFunction(V)
v = TestFunction(V)
g = NonLinearCoefficient(degree=1,element = V.ufl_element())
#u = interpolate(u,V)
#u_X = interpolate(u_X,V)
#u_Z = interpolate(u_Z,V)
#print("u_X(0,0),u_X(0.5,0),u_X(0.5,1),u_X(1,1)",u_X(0,0),u_X(0.5,0),u_X(0.5,1),u_X(1,1))
#print("u_Z(0,0),u_Z(0,0.5), u_Z(1,0.5),u_Z(1,1)",u_Z(0,0),u_Z(0,0.5),u_Z(1,0.5),u_Z(1,1))
#print('u_k(0.25,0.25) and u(0.25,0.25)',u_k(0.25,0.25), u(0.25,0.25))
#g = interpolate(g,V)
a = g*(dot(grad(u),grad(v)))*dx
#reassign u
u = Function(V)
#L = Expression('a',a=0,degree=1)*dx(mesh)
L = Constant(0)*v*dx
#solve(a==L,u,boundaryConditions)

#vtkfile_evens = File('results/solution_even.pvd')
#vtkfile_odds = File('results/solution_odds.pvd')
#vtkfile_evens << (u_k,0)
#vtkfile_odds << (u_k,0)
#vtkfile = File('results/solution_u_k.pvd')
#vtkfile << u_k

#compute error in L2 norm

vtkfile = File('results/solution.pvd')
vtkfile << (u_k,0)

eps = 1.0
loopTol = 1.0E-5
iter = 0
maxiter = 20  
print(g(0.4,0.5))
print(u_k(0.4,0.5))
while eps > loopTol and iter < maxiter:
    iter += 1
    solve(a==L,u,boundaryConditions)
    vertex_values_u_k = u_k.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    #eps = errornorm(u_k,u, "L2")
    error_max = np.abs(vertex_values_u_k - vertex_values_u)
    eps = np.linalg.norm(error_max, ord=np.inf)
    print('iter=%d: norm=%g' % (iter,eps))
    #print('u(0.5,0.5)=',u(0.5,0.5))
    u_k.assign(u)
    #g = interpolate(g,V)
#    if iter % 2==0:
#        vtkfile_evens << (u_k,iter)
#    else:
#       vtkfile_odds << (u_k,iter)
    vtkfile << (u_k,iter)   

#compute maximum error at vertices

print(g(0.4,0.5))
print(u_k(0.4,0.5))

