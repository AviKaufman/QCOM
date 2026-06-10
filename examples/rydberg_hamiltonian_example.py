from qcom import LatticeRegister, build_rydberg


register = LatticeRegister([(0.0, 0.0, 0.0), (5.0e-6, 0.0, 0.0)])
hamiltonian = build_rydberg(register, C6=1.0e-36, Omega=1.0, Delta=0.2)
print(hamiltonian)
print(hamiltonian.to_sparse().shape)
