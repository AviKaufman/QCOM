from qcom import LatticeRegister


register = LatticeRegister([(0.0, 0.0, 0.0), (5.0e-6, 0.0, 0.0)])
print(register.index_map())
print(register.distances())
