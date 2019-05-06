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
V_o = FunctionSpace(mesh,"P",2)
V_p = VectorFunctionSpace(mesh,"Lagrange",2)
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
    #u_k = 0;
    def __init__(self, **kwargs):
        self.u_k = kwargs.pop('u_k')
        super().__init__(**kwargs)
    def s(h):
    #parameterization of saturation van Genuchten
        if h < 0:
             return (1+(NonLinearCoefficient.alpha*abs(h))**(NonLinearCoefficient.n))**(-NonLinearCoefficient.m)
        else:
            return 1 
    
    def k_r(h):
        return (NonLinearCoefficient.s(h)**(1/2))*((1-(1-NonLinearCoefficient.s(h)**(1/NonLinearCoefficient.m))**NonLinearCoefficient.m)**2)

    def eval(self, value, x):
        if (self.u_k(x)) < 0:
           value[0]= NonLinearCoefficient.k_r(self.u_k(x))
        else:
           value[0]= 1
    def update_u_k(self, u_k):
        """

        :type u_k: function        """
        self.u_k.assign(u_k)
    def value_shape(self):
        return(0,)

class DerivativeNonLinearCoefficient(UserExpression):
    n = n;
    m = 1-1/n;
    alpha = alpha;
    #u_k = 0;
    def __init__(self,**kwargs):
        self.u_k = kwargs.pop('u_k')
        super().__init__(**kwargs)
    def s(h):
        if h < 0:
             return (1+(DerivativeNonLinearCoefficient.alpha*abs(h))**(DerivativeNonLinearCoefficient.n))**(-DerivativeNonLinearCoefficient.m)
        else:
            return 1 
    
    def k_r(h):
        return (DerivativeNonLinearCoefficient.s(h)**(1/2))*((1-(1-DerivativeNonLinearCoefficient.s(h)**(1/DerivativeNonLinearCoefficient.m))**DerivativeNonLinearCoefficient.m)**2)
    def NumDiff(self,value,x,h):
        return (DerivativeNonLinearCoefficient.k_r(self.u_k(x)+h)-DerivativeNonLinearCoefficient.k_r(self.u_k(x)))/(h)
    def eval(self, value, x):
        if (self.u_k(x)) < 0:
           value[0]= DerivativeNonLinearCoefficient.NumDiff(self,value,x,0.00000001)
        else:
           value[0]= 0
    def update_u_k(self,u_k):
        self.u_k.assign(u_k)
    def value_shape(self):
        return(0,)
    

vtkfile = File('resultsNewton/solution2.pvd')
vtkfile << (u_k,0)
iter = 0;
innerIter = 0;
maxIters = 200;
maxInnerIters = 5;
loopTol = 1.0E-4
lineSeachTol = 1.0
eps=1
eps2 = 10
omega = 0.9;
time = 0;
while iter < maxIters and eps > loopTol:
    iter += 1

    du = TrialFunction(V)
    v = TestFunction(V)
    g = NonLinearCoefficient(degree=2,element=V.ufl_element(),u_k = u_k)
    g_p = DerivativeNonLinearCoefficient(degree=2,element=V.ufl_element(),u_k = u_k)
    a = dot((g+g_p)*grad(du),grad(v))*dx
    L = -dot((g+g_p)*grad(u_k),grad(v))*dx
    du = Function(V)
    solve(a==L,du,DirichletBC(V,Constant(0.0),boundary_W))
    eps = np.linalg.norm(du.vector(),ord=np.inf)
    print('outer norm:%g' %eps)
    residuals = np.zeros(maxInnerIters)

    while innerIter < maxInnerIters and eps2 > .001:
        u_k_t = Function(V)
        u_k_t.assign(u_k + omega**(innerIter)*du)
        g_t = NonLinearCoefficient(degree=2, element=V.ufl_element(),u_k = u_k_t)
        div_gu = project(-div(project(g_t*grad(u_k_t),V_p)),V_o)
        residuals[innerIter] = np.linalg.norm(div_gu.compute_vertex_values(mesh),ord=np.inf)
        if innerIter > 0:
            eps2 = abs(residuals[innerIter] - residuals[innerIter-1])
        innerIter += 1

            

    eps2 = 10
    u_k.assign(u_k_t)
    innerIter = 0
    time += 1
    vtkfile << (u_k,time)
