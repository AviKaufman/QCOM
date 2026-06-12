from qcom import LatticeRegister, build_ising, ground_state


register = LatticeRegister([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
hamiltonian = build_ising(register, J=1.0, hx=0.1)
energy = ground_state(hamiltonian, return_vector=False)
print(f"ground energy = {energy:.6f}")
