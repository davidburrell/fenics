from fenics import *
import numpy as np

mesh = UnitSquareMesh(20,20)
V = FunctionSpace(mesh,'Lagrange',1)

#define boundary conditions
u0 = Expression("x[0]*x[1]",degree=2)
def u0_boundary(x,on_boundary):
    return on_boundary

bc = DirichletBC(V,u0,u0_boundary)


#define Variational Probelm
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-0.0)
a = inner(grad(u),grad(v))*dx
L = f*v*dx

#compute solution
u = Function(V)
solve(a == L, u, bc)
u.rename("u", "solution field")
u_array = u.vector().get_local()

#comput Gradient
V_g = VectorFunctionSpace(mesh,"Lagrange",1)
v = TestFunction(V_g)
w = TrialFunction(V_g)
a = inner(w,v)*dx
L = inner(grad(u),v)*dx
grad_u = Function(V_g)
solve(a==L,grad_u)
grad_u.rename("grad(u)","continuous gradient field")
(grad_u_x, grad_u_y) = grad_u.split() #extract components
grad_u_x.rename("grad(u)_x", "x-component of grad(u)")
grad_u_y.rename("grad(u)_y", "y-component of grad(u)")

#alternative computation of grad(u)
grad_u2 = project(grad(u), VectorFunctionSpace(mesh,"Lagrange",1))
grad_u2_x, grad_u2_y = grad_u2.split(deepcopy=True)

#exact expressions for grad u
u1 = Expression("x[1]",degree=1)
u2 = Expression("x[0]",degree=1)
grad_u_e_x = interpolate(u1,V)
grad_u_e_y = interpolate(u2,V)

u_e = interpolate(u0,V)
u_e_array = u_e.vector().get_local()
u_error = (u-u_e)**2*dx
grad_error = (grad_u_x - grad_u_e_x)**2*dx + (grad_u_y - -grad_u_e_y)**2*dx
grad2_error = (grad_u2_x - grad_u_e_x)**2*dx + (grad_u2_y - grad_u_e_y)**2*dx
u_norm = u**2*dx
grad_norm = grad_u_e_x**2*dx + grad_u_e_y**2*dx
Err_u = sqrt(abs(assemble(u_error)))
Err_g = sqrt(abs(assemble(grad_error)))
Err_g2 = sqrt(abs(assemble(grad2_error)))
Nrm_u = sqrt(abs(assemble(u_norm)))
Nrm_g = sqrt(abs(assemble(grad_norm)))

#Verification
print("Max error at nodes:", np.abs(u_e_array - u_array).max())

print("L2 err_u=", Err_u/Nrm_u)
print("L2 err_g=", Err_g/Nrm_g)
print("L2 err_g2=", Err_g2/Nrm_g)

interactive()
