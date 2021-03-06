from fenics import *
import matplotlib.pyplot as plt
import numpy as np
a = -2.0;
b = 1.0;
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
    return on_boundary and (near(x[1],0,tol) or near(x[1],1,tol))

bc_Z = DirichletBC(V, u_Z, boundary_Z)

boundaryConditions = [bc_X, bc_Z]


def boundary_W(x, on_boundary):
    return on_boundary

bcs = DirichletBC(V,u_X, boundary_W)
class NonLinearCoefficient(UserExpression):
    n = 3;
    m = 1-1/n;
    alpha = 1;

    def __init__(self,initialGuess,**kwargs):
        super().__init__(**kwargs)
        self.initialGuess = initialGuess
    
    @staticmethod
    def s(h):
    #parameterization of saturation van Genuchten
        if h < 0:
             return (1+(NonLinearCoefficient.alpha*abs(h))**(NonLinearCoefficient.n))**(-NonLinearCoefficient.m)
        else:
            return 1 
    
    @staticmethod
    def k_r(h):
        return (NonLinearCoefficient.s(h)**(1/2))*((1-(1-NonLinearCoefficient.s(h)**(1/NonLinearCoefficient.m))**NonLinearCoefficient.m)**2)

    def eval(self, value, x):
        #print(x)
        if (u_k(x)) < 0:
           #print(u_k(x))
           #print(NonLinearCoefficient.k_r((u_k(x))))
           value[0]= NonLinearCoefficient.k_r(u_k(x))
        else:
           value[0]= 1
    
    def value_shape(self):
        return(0,)


vtkfile = File('results/solution.pvd')
vtkfile << (u_k,0)
iter = 0;
maxIters = 22;
loopTol = 1.0E-1
eps=1
print(u_k(0.3,0.3))

while iter < maxIters and eps > loopTol:
    iter += 1
# Define Variational Problem
    u = TrialFunction(V)
    v = TestFunction(V)
    g = NonLinearCoefficient(degree=2,element = V.ufl_element(),initialGuess=2)

    a = dot(g*grad(u),grad(v))*dx
    u = Function(V)
    f = Constant(0.0)
    L = f*v*dx
    solve(a==L,u,bcs)
    
    vertex_values_u_k = u_k.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_max = np.abs(vertex_values_u_k - vertex_values_u)
    eps = np.linalg.norm(error_max, ord=2)
    print('iter=%d:norm=%g:u(0.3,.3)=%g' % (iter,eps,u(0.3,0.3)))
    u_k.assign(u)
    #print(vertex_values_u)

    vtkfile << (u_k,iter)




