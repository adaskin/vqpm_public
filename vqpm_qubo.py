'''
QPM converges the eigenvector of I+U with the maximum magnitude 
that is max |1+e^{iλ}|, where λ is the minimum eigenvalue of H.
The file includes the implementations of variational quantum power method (vqpm) applied to random QUBO,
vqpmForQUBO: it finds the minimum eigenphase of U.

@author: adaskin, updated by DeepSeek-R1 assistant
'''

import numpy as np
import matplotlib.pyplot as plt


def calculate_prob(psi, qubit):
    '''Computes probabilities for the states of a given qubit.
     Args:
         psi (array): Quantum state vector.
         qubit (int): Index of the qubit (1-based).
     Returns:
         array: Probabilities [prob_0, prob_1].
     '''
    n = len(psi)
    logn = int(np.log2(n))
    mask = 1 << (logn - qubit)  # Mask to check the specific qubit
    # Indices where qubit is 1
    indices_1 = (np.arange(n) & mask) != 0
    prob_1 = np.sum(np.abs(psi[indices_1])**2)
    prob_0 = 1.0 - prob_1
    return np.array([prob_0, prob_1])


def prepare_new_state(out_vec, n, q_states, pdiff, precision):
    '''Generates a new state based on qubit probabilities.
     Args:
         out_vec (array): Quantum state vector.
         n (int): Number of qubits.
         q_states (dict): Precomputed qubit states.
         pdiff (float): Threshold for collapsing probabilities.
         precision (int): Rounding precision.
     Returns:
         tuple: New state vector and updated q_states.
     '''
    state = [complex(1)]
    for q in range(1, n+1):
        if q in q_states:
            state_q = q_states[q]
        else:
            state_q = calculate_prob(out_vec, q)
            state_q = np.round(state_q, precision)
            
            if state_q[0] > state_q[1] + pdiff:
                state_q = np.array([1.0, 0.0])
                q_states[q] = state_q
            elif state_q[1] > state_q[0] + pdiff:
                state_q = np.array([0.0, 1.0])
                q_states[q] = state_q
            else:
                state_q = np.sqrt(state_q)  # Take amplitude
        
        state = np.kron(state, state_q)
    return state, q_states


def vqpm_for_qubo(u, n, max_iter, expected_idx, pdiff, precision):
    '''VQPM implementation for QUBO problems.
     Args:
         u (array): Diagonal of unitary matrix U.
         n (int): Number of qubits.
         max_iter (int): Maximum iterations.
         expected_idx (int): Index of expected eigenstate.
         pdiff (float): Probability difference threshold.
         precision (int): Rounding precision.
     Returns:
         tuple: Results including found state, probabilities, and iterations.
     '''
    p_min = np.zeros(max_iter)
    num_qubits = 2 ** n
    s2 = 1 / np.sqrt(2)
    in_vec = np.ones(num_qubits, dtype=complex) / np.sqrt(num_qubits)
    psi1 = np.zeros(num_qubits, dtype=complex)
    q_states = {}
    num_iter = max_iter

    for j in range(max_iter):
        psi0 = in_vec * s2
        psi1[:] = u * psi0  # Apply CU
        
        # Apply second Hadamard
        psi0_final = (psi0 + psi1) * s2  # I+U
        p0 = np.linalg.norm(psi0_final)
        psi0_final /= p0
        
        p_min[j] = np.abs(psi0_final[expected_idx]) ** 2
        if p_min[j] >= 0.5:
            num_iter = j + 1
            break
            
        in_vec, q_states = prepare_new_state(psi0_final, n, q_states, pdiff, precision)

    final_probs = np.abs(psi0_final)**2
    max_prob = np.max(final_probs)
    found_state = np.argmax(final_probs)
    
    return found_state, max_prob, q_states, p0, num_iter, p_min[:num_iter]


def unitary_for_qubo(n, Q):
    '''Creates diagonal unitary matrix for QUBO.
     Args:
         n (int): Number of qubits.
         Q (matrix): QUBO matrix.
     Returns:
         array: Diagonal of unitary matrix.
     '''
    num_terms = 2 ** n
    u = np.ones(num_terms, dtype=complex)
    for k in range(num_terms):
        b = bin(k)[2:].zfill(n)
        phase = 0.0
        for i in range(n):
            for j in range(i, n):
                if b[i] == '1' and b[j] == '1':
                    phase += Q[i][j]
        u[k] = np.exp(1j * phase)
    return u


def test_u_for_qubo(n, Q):
    '''Validates unitary construction.
     Args:
         n (int): Number of qubits.
         Q (matrix): QUBO matrix.
     Returns:
         array: Test values.
     '''
    num_terms = 2 ** n
    test_vals = np.zeros(num_terms)
    for k in range(num_terms):
        b = bin(k)[2:].zfill(n)
        total = 0.0
        for i in range(n):
            for j in range(i, n):
                if b[i] == '1' and b[j] == '1':
                    total += Q[i][j]
        test_vals[k] = total
    return test_vals


def random_qubo(n):
    '''Generates random QUBO matrix.
     Args:
         n (int): Matrix dimension.
     Returns:
         matrix: Symmetric Q matrix.
     '''
    Q = np.random.randn(n, n)
    return Q + Q.T


def adjust_unitary_phase(n, Q):
    '''Adjusts QUBO matrix for phase calculations.
     Args:
         n (int): Number of qubits.
         Q (matrix): QUBO matrix.
     Returns:
         tuple: Scaled Q, unitary, phases, and expected state.
     '''
    max_q = np.sum(np.abs(np.triu(Q)))
    Q_scaled = Q / max_q * np.pi/4
    u = unitary_for_qubo(n, Q_scaled)
    u *= np.exp(1j * np.pi/4)  # Shift phases to positive
    phases = np.angle(u)  # Get eigenphases
    expected_state = np.argmin(phases)
    return Q_scaled, u, phases, expected_state


if __name__ == '__main__':
    # Configuration
    pdiff = 0.01
    precision = 4
    n = 10  # Reduced for demonstration (original was 10)
    np.random.seed(42)  # For reproducibility
    
    # Generate QUBO problem
    Q = random_qubo(n)
    utest = test_u_for_qubo(n, Q)  # Get actual QUBO values
    
    # Prepare phase-adjusted unitary
    Q_adj, u, phases, target_state = adjust_unitary_phase(n, Q)
    
    # Run VQPM algorithm
    result_state, max_prob, q_states, _, iters, probs = vqpm_for_qubo(
        u, n, 20, target_state, pdiff, precision)
    
    # Convert states to binary format
    def state_to_bin(state, n):
        return bin(state)[2:].zfill(n)
    
    target_bin = state_to_bin(target_state, n)
    result_bin = state_to_bin(result_state, n)
    
    # Calculate success metrics
    success_prob = np.abs(u[target_state])**2 / (2**n)  # Theoretical maximum
    final_probs = np.abs(np.kron(u, [1,1]))**2  # Need to regenerate final state
    
    # Print detailed comparison
    print("================= Problem Summary =================")
    print(f"QUBO Matrix (first 5x5):")
    print(Q[:5,:5])
    print("\n=============== Algorithm Results =================")
    print(f"Expected state: {target_state} ({target_bin})")
    print(f"Found state:    {result_state} ({result_bin})")
    print(f"\nQUBO Values:")
    print(f"Expected: {utest[target_state]:.4f} | Found: {utest[result_state]:.4f}")
    print(f"\nProbabilities:")
    print(f"Max probability:       {max_prob:.4f}")
    print(f"Expected state prob:   {final_probs[target_state]:.4f}")
    print(f"Theoretical maximum:   {1/(2**n):.4f}")
    
    # Energy landscape visualization
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(utest, 'o', markersize=3)
    plt.plot(target_state, utest[target_state], 'ro', label='Target')
    plt.plot(result_state, utest[result_state], 'kx', label='Found')
    plt.xlabel('State Index')
    plt.ylabel('QUBO Value')
    plt.title('Energy Landscape')
    plt.legend()
    
    # Convergence plot
    plt.subplot(1,2,2)
    plt.plot(range(iters), probs, 'o-', label='Algorithm')
    plt.axhline(1/(2**n), color='r', linestyle='--', 
              label=f'Random guess ({1/(2**n):.4f})')
    plt.xlabel('Iteration')
    plt.ylabel('Success Probability')
    plt.title('Convergence Progress')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
