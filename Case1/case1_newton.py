from fenics import *
import matplotlib.pyplot as plt
import numpy as np
a = -2.0;
b = 1.0;
tol = 1E-14;
num_steps = 10;
alpha=2;
n=4;
mesh = UnitSquareMesh(20,20)
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
           #print(k_r(abs(u_k(x))))
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
    #parameterization of saturation van Genuchten
        if h < 0:
             return (1+(NonLinearCoefficient.alpha*abs(h))**(NonLinearCoefficient.n))**(-NonLinearCoefficient.m)
        else:
            return 1 
    
    @staticmethod
    def k_r(h):
        return (NonLinearCoefficient.s(h)**(1/2))*((1-(1-NonLinearCoefficient.s(h)**(1/NonLinearCoefficient.m))**NonLinearCoefficient.m)**2)
    @staticmethod
    def NumDiff(self, value, x, h):
        #print(x) 
        #print((NonLinearCoefficient.k_r(u_k(x))+NonLinearCoefficient.k_r(u_k(x)))/(2))
        return (NonLinearCoefficient.k_r(u_k(x)-h)-NonLinearCoefficient.k_r(u_k(x)+h))/(2*h)
    def eval(self, value, x):
        #print(x)
        if (u_k(x)) < 0:
           #print(u_k(x))
           #print(k_r(abs(u_k(x))))
           value[0]= DerivativeNonLinearCoefficient.NumDiff(self,value,x,0.000001)
        else:
           value[0]= 0
    
    def value_shape(self):
        return(0,)
    

vtkfile = File('resultsNewton/solution.pvd')
vtkfile << (u_k,0)
iter = 0;
innerIter = 0;
maxIters = 30;
maxInnerIters = 10;
loopTol = 1.0E-2
eps=1
eps2 = 1
omega = 0.5;

innerLoopTol = 150

while iter < maxIters and eps > loopTol:
    iter += 1
# Define Variational Problem
    u = Function(V)
    du = TrialFunction(V)
    v = TestFunction(V)
    g = NonLinearCoefficient(degree=2,element = V.ufl_element(),initialGuess=2)
    g_u = DerivativeNonLinearCoefficient(degree=2,element=V.ufl_element())
    #print(g_u(1/2,1/2))
#    g_u = diff(g,u)
#    g_u = g_u*dx(mesh)
    #grad_u = project(grad(u_k),VectorFunctionSpace(mesh,'P',2))
    #g_u = project(g_u,FunctionSpace(mesh,'P',1))
    a = dot(g*grad(du),grad(v))*dx + dot(g_u*du*grad(u_k),grad(v))*dx
    #a = dot(g*grad(du),grad(v))*dx + dot(g_u*du*grad_u,grad(v))*dx
    du = Function(V)
    f = Constant(0.0)
    L = -dot(g*grad(u_k),grad(v))*dx 
    #L = -dot(g*grad_u,grad(v))*dx 
    solve(a==L,du,DirichletBC(V,Constant(0.0),boundary_W)
)
    
    vertex_values_u_k = u_k.compute_vertex_values(mesh)
    #print(vertex_values_u_k)
    #print(vertex_values_u)
    #u.vector()[:] = u_k.vector() + du.vector()
    u.assign(u_k+du)
    vertex_values_u = u.compute_vertex_values(mesh)
    eps = np.linalg.norm(np.abs(vertex_values_u_k - vertex_values_u),ord=2)
    deps = np.linalg.norm(du.vector(), ord=np.Inf)
    print('norm=%g,u_k+1(0.3,0.3)=%g, u_k(0.3,0.3)=%g'%(eps,u(0.3,0.3),u_k(0.3,0.3)))
    u_k.assign(u)
    u_ln.assign(u)
   
    while eps2 > innerLoopTol and innerIter < maxInnerIters:
        innerIter +=1
    #print('iter=%d:norm=%g' % (iter,eps))
        #u.vector()[:] = u_k.vector() + (omega**innerIter)*du.vector()

    #print(vertex_values_u_k)
    #    u_ln.assign(u)
        #grad_u = project(grad(u_ln),VectorFunctionSpace(mesh,'P',2))
        g_u = project(g_u,FunctionSpace(mesh,'P',2))
        du=TrialFunction(V)
        a = dot(g*grad(du),grad(v))*dx + dot(g_u*du*grad(u_ln),grad(v))*dx
        L = -dot(g*grad(u_ln),grad(v))*dx
        du=Function(V)
        solve(a==L,du,DirichletBC(V,Constant(0.0),boundary_W))
        eps1 = np.linalg.norm(du.vector(),ord=2)
        
        error_max = np.abs(u_k.compute_vertex_values(mesh) - u_ln.compute_vertex_values(mesh)) 
        eps2 = np.linalg.norm(error_max, ord=2)
        #print(du.vector())
        print('inner eps=%g, inner eps2=%g' % (eps1,eps2))
        u_ln.assign(u_k + (omega**innerIter)*du)
    innerIter = 0;
    u_k.assign(u_ln)
    #print(u_k.compute_vertex_values(mesh))
    
    vtkfile << (u_k,iter)




