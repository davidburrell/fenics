from fenics import *

class NonLinearCoefficient(UserExpression):
    def eval(self, value, x):
        #value[0] = 2*self.u_k(x)
        value[0] = 2
    def value_shape(self):
        return (0,)
