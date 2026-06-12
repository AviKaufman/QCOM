from qcom import LatticeRegister, build_ising


register = LatticeRegister([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
hamiltonian = build_ising(register, J=1.0, hx=0.2, hz=0.0)
print(hamiltonian.parameters())
print(hamiltonian.to_sparse().shape)
