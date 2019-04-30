from fenics import *

class NonLinearCoefficient(UserExpression):
    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
    def eval(self, value, x)

