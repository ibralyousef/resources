"""
Grover's Search Algorithm - Simulation Methods Comparison
Compares state-vector, density matrix, MPS, unitary, and pulse-level simulations.

Usage:
    python3 grover_comparison.py --qubits 4 --target 5
    python3 grover_comparison.py --qubits 3 --target 3
"""

import argparse
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator

# QuTiP imports for pulse-level simulation
try:
    from qutip_qip.circuit import QubitCircuit
    from qutip_qip.device import LinearSpinChain
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False


def build_oracle(n_qubits, target):
    """
    Build oracle that marks |target> state with a phase flip.
    Oracle: |target> -> -|target>, all other states unchanged.
    """
    oracle = QuantumCircuit(n_qubits, name=f"Oracle_{target}")
    target_binary = format(target, f'0{n_qubits}b')

    # Apply X gates to qubits where target bit is 0
    for i, bit in enumerate(target_binary[::-1]):
        if bit == '0':
            oracle.x(i)

    # Multi-controlled Z gate (MCZ) using H-MCX-H on last qubit
    if n_qubits == 1:
        oracle.z(0)
    else:
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)

    # Undo X gates
    for i, bit in enumerate(target_binary[::-1]):
        if bit == '0':
            oracle.x(i)

    return oracle


def optimal_iterations(n_qubits):
    """Calculate optimal number of Grover iterations."""
    N = 2 ** n_qubits
    return max(1, int(np.round(np.pi / 4 * np.sqrt(N))))


def theoretical_success_prob(n_qubits, num_iterations):
    """
    Calculate theoretical probability of finding the target.
    P = sin^2((2k+1) * theta) where theta = arcsin(1/sqrt(N))
    """
    N = 2 ** n_qubits
    theta = np.arcsin(1 / np.sqrt(N))
    prob = np.sin((2 * num_iterations + 1) * theta) ** 2
    return prob


def build_grover_circuit(n_qubits, target, num_iterations=None):
    """
    Build complete Grover's search circuit.

    Args:
        n_qubits: Number of qubits
        target: Target state to search for (integer)
        num_iterations: Number of Grover iterations (auto if None)

    Returns:
        QuantumCircuit, num_iterations used
    """
    if num_iterations is None:
        num_iterations = optimal_iterations(n_qubits)

    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    qc.h(range(n_qubits))

    # Build oracle
    oracle = build_oracle(n_qubits, target)

    # Use GroverOperator which includes oracle + diffuser
    grover_op = GroverOperator(oracle)

    # Apply Grover iterations
    for _ in range(num_iterations):
        qc.append(grover_op, range(n_qubits))

    return qc, num_iterations


def run_statevector(circuit):
    """State-vector simulation. Memory: O(2^n)."""
    start = time.perf_counter()

    sim = AerSimulator(method='statevector')
    circuit_copy = circuit.copy()
    circuit_copy.save_statevector()

    job = sim.run(transpile(circuit_copy, sim), shots=1)
    result = job.result()
    statevector = result.get_statevector()

    elapsed = time.perf_counter() - start

    return {
        'method': 'State-vector',
        'time': elapsed,
        'state': np.array(statevector.data),
        'memory_bytes': statevector.data.nbytes,
        'scaling': 'O(2^n)'
    }


def run_density_matrix(circuit, n_qubits):
    """Density matrix simulation. Memory: O(4^n)."""
    if n_qubits > 14:
        print(f"    SKIPPED: n={n_qubits} > 14 (would require {4**n_qubits * 16 / 1e9:.1f} GB)")
        return None

    start = time.perf_counter()

    sim = AerSimulator(method='density_matrix')
    circuit_copy = circuit.copy()
    circuit_copy.save_density_matrix()

    job = sim.run(transpile(circuit_copy, sim), shots=1)
    result = job.result()
    dm = result.data()['density_matrix']

    elapsed = time.perf_counter() - start

    # Extract state from density matrix diagonal (pure state case)
    state = np.sqrt(np.abs(np.diag(dm.data)))

    return {
        'method': 'Density Matrix',
        'time': elapsed,
        'state': state,
        'memory_bytes': dm.data.nbytes,
        'scaling': 'O(4^n)'
    }


def run_mps(circuit):
    """Matrix Product State (tensor network) simulation. Memory: O(n * chi^2)."""
    start = time.perf_counter()

    sim = AerSimulator(method='matrix_product_state')
    circuit_copy = circuit.copy()
    circuit_copy.save_statevector()

    job = sim.run(transpile(circuit_copy, sim), shots=1)
    result = job.result()
    statevector = result.get_statevector()

    elapsed = time.perf_counter() - start

    return {
        'method': 'MPS (Tensor Net)',
        'time': elapsed,
        'state': np.array(statevector.data),
        'memory_bytes': statevector.data.nbytes,
        'scaling': 'O(n * chi^2)'
    }


def run_unitary(circuit, n_qubits):
    """Unitary matrix simulation. Memory: O(4^n)."""
    if n_qubits > 12:
        print(f"    SKIPPED: n={n_qubits} > 12 (would require {4**n_qubits * 16 / 1e9:.1f} GB)")
        return None

    start = time.perf_counter()

    sim = AerSimulator(method='unitary')
    circuit_copy = circuit.copy()
    circuit_copy.save_unitary()

    job = sim.run(transpile(circuit_copy, sim), shots=1)
    result = job.result()
    unitary = result.get_unitary()

    elapsed = time.perf_counter() - start

    # Apply unitary to |0...0>
    N = 2 ** n_qubits
    initial_state = np.zeros(N, dtype=complex)
    initial_state[0] = 1.0
    output_state = unitary.data @ initial_state

    return {
        'method': 'Unitary',
        'time': elapsed,
        'state': output_state,
        'memory_bytes': unitary.data.nbytes,
        'scaling': 'O(4^n)'
    }


def run_qutip_pulse(n_qubits, target, num_iterations):
    """
    Pulse-level simulation using qutip-qip.
    Simulates Grover's algorithm via Hamiltonian time evolution.
    Limited to 2 qubits due to multi-controlled gate complexity.
    """
    if not QUTIP_AVAILABLE:
        print("    SKIPPED: qutip-qip not installed")
        return None

    if n_qubits > 2:
        print(f"    SKIPPED: n={n_qubits} > 2 (pulse simulation limited to 2 qubits)")
        return None

    start = time.perf_counter()

    # Build 2-qubit Grover circuit in qutip-qip
    qc = QubitCircuit(n_qubits)

    # Initial superposition: H on all qubits
    for i in range(n_qubits):
        qc.add_gate("SNOT", targets=i)  # SNOT = Hadamard in qutip

    # Helper: CZ using H-CNOT-H decomposition (CZ = H(target) CNOT H(target))
    def add_cz(qc, control, target_q):
        qc.add_gate("SNOT", targets=target_q)
        qc.add_gate("CNOT", controls=control, targets=target_q)
        qc.add_gate("SNOT", targets=target_q)

    # Grover iterations
    for _ in range(num_iterations):
        # Oracle: mark target state with phase flip
        # For 2 qubits, target in {0,1,2,3}
        target_binary = format(target, f'0{n_qubits}b')

        # Apply X gates where target bit is 0 (to convert target to |11>)
        for i, bit in enumerate(target_binary[::-1]):
            if bit == '0':
                qc.add_gate("X", targets=i)

        # CZ gate decomposed for LinearSpinChain
        add_cz(qc, 0, 1)

        # Undo X gates
        for i, bit in enumerate(target_binary[::-1]):
            if bit == '0':
                qc.add_gate("X", targets=i)

        # Diffuser: H -> X -> CZ -> X -> H
        for i in range(n_qubits):
            qc.add_gate("SNOT", targets=i)
        for i in range(n_qubits):
            qc.add_gate("X", targets=i)
        add_cz(qc, 0, 1)
        for i in range(n_qubits):
            qc.add_gate("X", targets=i)
        for i in range(n_qubits):
            qc.add_gate("SNOT", targets=i)

    # Create processor and run pulse-level simulation
    processor = LinearSpinChain(n_qubits)
    processor.load_circuit(qc)

    # Define initial state |00...0>
    from qutip import basis, tensor
    init_state = tensor([basis(2, 0) for _ in range(n_qubits)])

    # Run the pulse simulation
    result = processor.run_state(init_state=init_state)
    final_state = result.states[-1]

    elapsed = time.perf_counter() - start

    # Extract state vector from QuTiP Qobj
    state_array = final_state.full().flatten()

    # QuTiP uses big-endian qubit ordering (first qubit = MSB)
    # Qiskit/standard uses little-endian (first qubit = LSB)
    # Reorder state vector to match Qiskit convention
    N = 2 ** n_qubits
    reordered_state = np.zeros(N, dtype=complex)
    for i in range(N):
        # Reverse bit order: e.g., for 2 qubits, 01 <-> 10
        reversed_idx = int(format(i, f'0{n_qubits}b')[::-1], 2)
        reordered_state[i] = state_array[reversed_idx]

    return {
        'method': 'QuTiP Pulse',
        'time': elapsed,
        'state': reordered_state,
        'memory_bytes': reordered_state.nbytes,
        'scaling': 'O(4^n) Hamiltonian'
    }


def verify_results(results, n_qubits, target, expected_prob):
    """Verify each simulation method found the target state."""
    print("\n" + "=" * 85)
    print("GROVER'S SEARCH VERIFICATION")
    print("=" * 85)
    target_binary = format(target, f'0{n_qubits}b')
    print(f"Target state: |{target_binary}> (decimal {target})")
    print(f"Expected probability: {expected_prob:.4f}")
    print("-" * 85)

    print(f"\n{'Method':<18} {'Target Prob':<12} {'Max State':<14} {'Max Prob':<10} {'Status':<10}")
    print("-" * 75)

    for r in results:
        if r is None:
            continue

        state = r['state']
        method = r['method']
        probs = np.abs(state) ** 2

        # Get target probability
        target_prob = probs[target]

        # Get max probability state
        max_idx = np.argmax(probs)
        max_binary = format(max_idx, f'0{n_qubits}b')
        max_prob = probs[max_idx]

        # Success if target has highest probability
        is_correct = (max_idx == target)
        status = "PASS" if is_correct else "FAIL"

        max_state_str = f"|{max_binary}>"
        print(f"{method:<18} {target_prob:<12.4f} {max_state_str:<14} {max_prob:<10.4f} [{status}]")

    print("-" * 75)


def compare_results(results, n_qubits):
    """Print comparison table of all simulation methods."""
    print("\n" + "=" * 85)
    print("SIMULATION COMPARISON RESULTS")
    print("=" * 85)

    print(f"\n{'Method':<18} {'Time (ms)':<12} {'Memory':<15} {'Scaling':<15}")
    print("-" * 65)

    for r in results:
        if r is None:
            continue

        time_ms = r['time'] * 1000
        mem_kb = r['memory_bytes'] / 1024

        if mem_kb < 1:
            mem_str = f"{r['memory_bytes']} B"
        elif mem_kb < 1024:
            mem_str = f"{mem_kb:.1f} KB"
        else:
            mem_str = f"{mem_kb/1024:.2f} MB"

        print(f"{r['method']:<18} {time_ms:<12.2f} {mem_str:<15} {r['scaling']:<15}")

    print("\n" + "=" * 85)


def print_state_amplitudes(results, n_qubits, target, max_show=8):
    """Print state amplitudes highlighting the target."""
    print("\n" + "=" * 85)
    print(f"STATE PROBABILITIES (showing {min(max_show, 2**n_qubits)} of {2**n_qubits} states)")
    print("=" * 85)

    sv_result = next((r for r in results if r and r['method'] == 'State-vector'), None)
    if sv_result:
        state = sv_result['state']
        probs = np.abs(state) ** 2

        # Sort by probability (descending)
        sorted_idx = np.argsort(probs)[::-1]

        print(f"\nTop {min(max_show, len(state))} states by probability:")
        for rank, idx in enumerate(sorted_idx[:max_show]):
            binary = format(idx, f'0{n_qubits}b')
            prob = probs[idx]
            marker = " <-- TARGET" if idx == target else ""
            print(f"  {rank+1}. |{binary}> (dec {idx:>3}): prob={prob:.6f}{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare quantum simulation methods on Grover's search circuit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 grover_comparison.py --qubits 4 --target 5       # Search for |0101> (all methods)
  python3 grover_comparison.py -n 4 -t 5 -m sv             # State-vector only
  python3 grover_comparison.py -n 4 -t 5 -m sv,mps         # State-vector and MPS
  python3 grover_comparison.py -n 2 -t 3 -m pulse          # Pulse-level (2 qubits only)
  python3 grover_comparison.py -n 2 -t 2 -m sv,pulse       # Compare state-vector and pulse

Methods: sv (state-vector), dm (density matrix), mps, unitary, pulse, all
        """
    )
    parser.add_argument(
        '--qubits', '-n', type=int, default=4,
        help='Number of qubits (default: 4)'
    )
    parser.add_argument(
        '--target', '-t', type=int, default=5,
        help='Target state to search for (default: 5)'
    )
    parser.add_argument(
        '--iterations', '-i', type=int, default=None,
        help='Number of Grover iterations (auto if not specified)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--method', '-m', type=str, default='all',
        help='Method(s) to run: sv, dm, mps, unitary, or all (default: all). '
             'Comma-separated for multiple, e.g., -m sv,mps'
    )

    args = parser.parse_args()
    n_qubits = args.qubits
    target = args.target

    if n_qubits < 2:
        print("Error: Need at least 2 qubits")
        return
    if n_qubits > 20:
        print(f"Warning: {n_qubits} qubits may be very slow")
    if target < 0 or target >= 2**n_qubits:
        print(f"Error: Target must be in range [0, {2**n_qubits - 1}]")
        return

    target_binary = format(target, f'0{n_qubits}b')
    num_iterations = args.iterations if args.iterations else optimal_iterations(n_qubits)
    expected_prob = theoretical_success_prob(n_qubits, num_iterations)

    print(f"\n{'='*85}")
    print(f"GROVER'S SEARCH ALGORITHM - {n_qubits} QUBITS")
    print(f"Target state: |{target_binary}> (decimal {target})")
    print(f"Search space: {2**n_qubits} items")
    print(f"Grover iterations: {num_iterations}")
    print(f"Expected success probability: {expected_prob*100:.1f}%")
    print(f"{'='*85}")

    # Build circuit
    print(f"\nBuilding Grover's search circuit...")
    circuit, _ = build_grover_circuit(n_qubits, target, num_iterations)
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Gate count: {circuit.size()}")

    # Parse method selection
    method_map = {
        'sv': ('State-vector', lambda: run_statevector(circuit)),
        'dm': ('Density Matrix', lambda: run_density_matrix(circuit, n_qubits)),
        'mps': ('MPS (Tensor Network)', lambda: run_mps(circuit)),
        'unitary': ('Unitary', lambda: run_unitary(circuit, n_qubits)),
        'pulse': ('QuTiP Pulse', lambda: run_qutip_pulse(n_qubits, target, num_iterations)),
    }

    if args.method.lower() == 'all':
        methods_to_run = list(method_map.keys())
    else:
        methods_to_run = [m.strip().lower() for m in args.method.split(',')]
        invalid = [m for m in methods_to_run if m not in method_map]
        if invalid:
            print(f"Error: Invalid method(s): {invalid}")
            print(f"Valid options: {', '.join(method_map.keys())}, all")
            return

    # Run simulations
    print("\n" + "-" * 85)
    print("Running simulations...")
    print("-" * 85)

    results = []

    for i, method_key in enumerate(methods_to_run, 1):
        method_name, run_fn = method_map[method_key]
        print(f"\n{i}. {method_name} simulation...")
        results.append(run_fn())

    # Verify results
    verify_results(results, n_qubits, target, expected_prob)

    # Compare performance
    compare_results(results, n_qubits)

    # Show state amplitudes
    print_state_amplitudes(results, n_qubits, target)


if __name__ == "__main__":
    main()
