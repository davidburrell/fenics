from dolfin import *

class SaturationParamaterization(Expression):
    def __init__(self, param, **kwargs):
        self.param = param

    def eval(self, values, x):
        values
