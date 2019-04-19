from dolfin import *

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)

V = FunctionSpace(mesh, "CG", 1)

# Define boundary condition
g = Constant(1.0)
bc = DirichletBC(V, g, DirichletBoundary())

# Define variational problem
u_k = Function(V)
du = TrialFunction(V)
v = TestFunction(V)
f = Expression("x[0]*sin(x[1])",degree=2)

# define F and J
method = 2
if method == 1: # as in the demo-code
    F = inner((1 + u_k**2)*grad(u_k), grad(v))*dx - f*v*dx
    J = derivative(F, u_k, du)

elif method == 2: # using custom expressions, doing the same thing
    class CustomNonlinearity(UserExpression):
        def eval(self, value, x):
            value[0] = 1 + u_k(x)**2
    q = CustomNonlinearity(degree=2)

    class DCustomNonlinearity(UserExpression):
        def eval(self, value, x):
            value[0] = 2 * u_k(x)
    Dq = DCustomNonlinearity(degree=2)

    F = inner(q*grad(u_k), grad(v))*dx - f*v*dx
    J = inner(q*grad(du), grad(v)) * dx + inner(Dq * du * grad(u_k), grad(v)) * dx

# Compute solution
problem = NonlinearVariationalProblem(F, u_k, bc, J)
solver = NonlinearVariationalSolver(problem)
solver.solve()

# Plot solution and solution gradient
plot(u_k, title="Solution")
plot(grad(u_k), title="Solution gradient")
