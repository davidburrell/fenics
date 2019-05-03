from fenics import *
import matplotlib.pyplot as plt
import numpy as np
a = -2.0;
b = 1.0;
tol = 1E-14;
num_steps = 10;
alpha=2;
n=2.5;
mesh = UnitSquareMesh(30,30)
V = FunctionSpace(mesh,"P",1)
u_k = interpolate(Expression("3*x[0]-2",degree=1),V)
u_ln = interpolate(Expression("3*x[0]-2",degree=1),V)
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
    n = n;
    m = 1-1/n;
    alpha = alpha;

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
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
        if (u_k(x)) < 0:
           value[0]= NonLinearCoefficient.k_r(u_k(x))
        else:
           value[0]= 1
    
    def value_shape(self):
        return(0,)

class DerivativeNonLinearCoefficient(UserExpression):
    n = n;
    m = 1-1/n;
    alpha = alpha;

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def s(h):
        if h < 0:
             return (1+(DerivativeNonLinearCoefficient.alpha*abs(h))**(DerivativeNonLinearCoefficient.n))**(-DerivativeNonLinearCoefficient.m)
        else:
            return 1 
    
    @staticmethod
    def k_r(h):
        return (DerivativeNonLinearCoefficient.s(h)**(1/2))*((1-(1-DerivativeNonLinearCoefficient.s(h)**(1/DerivativeNonLinearCoefficient.m))**DerivativeNonLinearCoefficient.m)**2)
    @staticmethod
    def NumDiff(self,value,x,h):
        return (DerivativeNonLinearCoefficient.k_r(u_k(x)+h)-DerivativeNonLinearCoefficient.k_r(u_k(x)))/(h)
    def eval(self, value, x):
        if (u_k(x)) < 0:
           value[0]= DerivativeNonLinearCoefficient.NumDiff(self,value,x,0.00000001)
        else:
           value[0]= 0
    
    def value_shape(self):
        return(0,)
    

vtkfile = File('resultsNewton/solution.pvd')
vtkfile << (u_k,0)
iter = 0;
innerIter = 0;
maxIters = 200;
maxInnerIters = 20;
loopTol = 1.0E-4
eps=1
eps2 = 1
omega = 0.5;
time = 0;
while iter < maxIters and eps > loopTol:
    iter += 1

    du = TrialFunction(V)
    v = TestFunction(V)
    g = NonLinearCoefficient(degree=2,element=V.ufl_element())
    g_p = DerivativeNonLinearCoefficient(degree=2,element=V.ufl_element())
    a = dot((g+g_p)*grad(du),grad(v))*dx
    L = -dot((g+g_p)*grad(u_k),grad(v))*dx
    du = Function(V)
    solve(a==L,du,DirichletBC(V,Constant(0.0),boundary_W))
    eps = np.linalg.norm(du.vector(),ord=np.inf)        
    print('outer norm:%g' %eps)
    while innerIter < maxInnerIters:
        innerIter += 1
        du = TrialFunction(V)
        a = dot((g+g_p)*grad((omega**(-innerIter))*du),grad(v))*dx
        du = Function(V)
        solve(a==L, du, DirichletBC(V,Constant(0.0),boundary_W))
        eps2 = np.linalg.norm(du.vector(),ord=np.inf)
    #    print ('inner Norm:%g'% eps2)
        if eps2 < eps:
            u_k.assign(u_k+du)
            time +=1
            vtkfile << (u_k,time)
            break
        
            
    
    innerIter = 0
    print('iteration: %d'%iter)
    eps = eps2
    eps2 = 1
    #u_k.assign(u_k+du)
    #time += 1
    #print('outer norm:%g' % eps)
    #vtkfile << (u_k,time)
