"""
Chapter 3: Numerical verification of the Bellman equations
Verify the Bellman expectation equation and the Bellman optimality equation with code

This experiment demonstrates:
1. Manually computing the Bellman expectation equation V^π(s) by solving a linear system
2. Iteratively solving V^π(s) with policy evaluation and V*(s) with value iteration
3. Comparing the manual and iterative results to verify consistency

How to run:
    python bellman_equation_verify.py
"""

import numpy as np

# Part 1: Define a simple 3-state MDP

# Number of states and actions
N_STATES = 3
N_ACTIONS = 2

# Transition probabilities: P[s][a] = {next state: probability}
P = {
    0 : { # s0
        0 : {1 : 1.0} , # a0 → s1 with probability 1.0
        1 : {2 : 1.0} , # a1 → s2 with probability 1.0
    },
    1: {  # s1
        0: {0: 0.5, 2: 0.5},  # a0 → s0 with probability 0.5, s2 with probability 0.5
        1: {1: 0.8, 2: 0.2},  # a1 → s1 with probability 0.8, s2 with probability 0.2
    },
    2: {  # s2
        0: {1: 1.0},    # a0 → s1 with probability 1.0
        1: {2: 1.0},    # a1 → s2 with probability 1.0 (self-loop)
    },
}

# Reward function: R[s][a] = immediate reward
R = {
    0: {0: 1, 1: 2},     # s0: a0 reward 1, a1 reward 2
    1: {0: -1, 1: 0},    # s1: a0 reward -1, a1 reward 0
    2: {0: 3, 1: 1},     # s2: a0 reward 3, a1 reward 1
}

GAMMA = 0.9  # Discount factor

# Part 2: Bellman expectation equation -- manual computation
def manual_bellman_expectation():
    """
    Manually compute the Bellman expectation equation V^π(s)

    Given a fixed policy π, the Bellman expectation equation is:
        V^π(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V^π(s')]

    We use a simple uniform random policy:
        π(a|s) = 0.5 (each action chosen with equal probability)

    Manual derivation (let V^π(s0) = v0, V^π(s1) = v1, V^π(s2) = v2):

    V^π(s0) = 0.5 * [R(s0,a0) + γ * V^π(s1)] + 0.5 * [R(s0,a1) + γ * V^π(s2)]
            = 0.5 * [1 + 0.9 * v1] + 0.5 * [2 + 0.9 * v2]
            = 0.5 + 0.45*v1 + 1 + 0.45*v2
            = 1.5 + 0.45*v1 + 0.45*v2  ........................ (Equation 1)

    V^π(s1) = 0.5 * [R(s1,a0) + γ * (0.5*V^π(s0) + 0.5*V^π(s2))]
            + 0.5 * [R(s1,a1) + γ * (0.8*V^π(s1) + 0.2*V^π(s2))]
            = 0.5 * [-1 + 0.9*(0.5*v0 + 0.5*v2)]
            + 0.5 * [0 + 0.9*(0.8*v1 + 0.2*v2)]
            = -0.5 + 0.225*v0 + 0.225*v2 + 0.36*v1 + 0.09*v2
            = -0.5 + 0.225*v0 + 0.36*v1 + 0.315*v2  ............ (Equation 2)

    V^π(s2) = 0.5 * [R(s2,a0) + γ * V^π(s1)] + 0.5 * [R(s2,a1) + γ * V^π(s2)]
            = 0.5 * [3 + 0.9 * v1] + 0.5 * [1 + 0.9 * v2]
            = 1.5 + 0.45*v1 + 0.5 + 0.45*v2
            = 2.0 + 0.45*v1 + 0.45*v2  ........................ (Equation 3)
    """
    print("=" * 60)
    print("  Bellman expectation equation -- manual derivation")
    print("=" * 60)
    print()
    print("Given policy: uniform random π(a|s) = 0.5")
    print(f"Discount factor: γ = {GAMMA}")
    print()
    print("System of equations (v0 = V^π(s0), v1 = V^π(s1), v2 = V^π(s2)):")
    print("  v0 = 1.5   + 0.45*v1 + 0.45*v2  ...... (Equation 1)")
    print("  v1 = -0.5  + 0.225*v0 + 0.36*v1 + 0.315*v2  (Equation 2)")
    print("  v2 = 2.0   + 0.45*v1 + 0.45*v2  ...... (Equation 3)")
    print()

    # Solve the linear system A * v = b
    # Equation 1: v0 - 0.45*v1 - 0.45*v2 = 1.5
    # Equation 2: -0.225*v0 + (1-0.36)*v1 - 0.315*v2 = -0.5
    # Equation 3: -0.45*v1 + (1-0.45)*v2 = 2.0

    A = np.array([
        [1.0 , -0.45 , -0.45] , 
        [-0.225 , 0.64 , -0.315] , 
        [0.0 , -0.45 , 0.55] ,
    ])
    b = np.array([1.5 , -0.5 , 2.0])

    manual_V = np.linalg.solve(A , b)

    print("Solving the linear system by hand gives : ")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {manual_V[i]:.6f}")
    print()
    return manual_V

# Part 3: Policy evaluation -- iteratively solve the Bellman expectation equation
def policy_evaluation(policy , max_iter = 1000 , tol = 1e-8):
    """
    Policy evaluation: iteratively solve the Bellman expectation equation

    Bellman expectation equation (iterative form):
        V(s) ← Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]

    Iterate repeatedly until V(s) converges; the converged V(s) is V^π(s).

    Args:
        policy: policy π(a|s), shape (N_STATES, N_ACTIONS)
        max_iter: maximum number of iterations
        tol: convergence threshold
    Returns:
        V: state value function
        history: record of V values at each iteration (for visualizing convergence)
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            # Bellman expectation equation: weighted sum over all actions
            for a in range(N_ACTIONS):
                # π(a|s) * [R(s,a) + γ * Σ P(s'|s,a) * V(s')]
                action_value = R[s][a]
                for next_s, prob in P[s][a].items():
                    action_value += GAMMA * prob * V[next_s]
                V_new[s] += policy[s][a] * action_value

        # Check for convergence
        delta = np.max(np.abs(V_new - V))
        history.append(V_new.copy())
        V = V_new

        if delta < tol:
            break

    return V , history


# Part 4: Value iteration -- solve the Bellman optimality equation
def value_iteration(max_iter=1000, tol=1e-8):
    """
    Value iteration: solve the Bellman optimality equation to find V*(s)

    Bellman optimality equation (iterative form):
        V(s) ← max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]

    Difference from the Bellman expectation equation:
    - Expectation equation: given a policy π, solve for V^π(s)
    - Optimality equation: optimize over all policies to solve for V*(s)

    V*(s) satisfies:
        V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a) + γ * V*(s')]

    Args:
        max_iter: maximum number of iterations
        tol: convergence threshold
    Returns:
        V_star: optimal state value function
        optimal_policy: optimal policy
        history: convergence history
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            # Compute Q(s, a) for each action
            q_values = []
            for a in range(N_ACTIONS):
                q = R[s][a]
                for next_s , prob in P[s][a].items():
                    q += GAMMA * prob * V[next_s]
                q_values.append(q)

            # Bellman optimality equation: take the max instead of the expectation
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))
        history.append(V_new.copy())
        V = V_new

        if delta < tol :
            break

    # Extract the optimal policy from V*
    optimal_policy = extract_optimal_policy(V)

    return V, optimal_policy, history

def extract_optimal_policy(V):
    """
    Extract the optimal policy π* from the optimal value function V*

    π*(s) = argmax_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V*(s')]
    """
    policy = np.zeros((N_STATES, N_ACTIONS))

    for s in range(N_STATES):
        q_values = []
        for a in range(N_ACTIONS):
            q = R[s][a]
            for next_s , prob in P[s][a].items():
                q += GAMMA * prob * V[next_s]
            q_values.append(q)

        best_action = np.argmax(q_values)
        policy[s][best_action] = 1.0

    return policy


# Part 5: Run the experiments and compare the results
def main():
    """Compare the manual solution, policy evaluation, and value iteration"""
    uniform_policy = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

    # Bellman expectation equation: solve the linear system by hand
    manual_V = manual_bellman_expectation()

    # Bellman expectation equation: solve it iteratively
    iter_V, _ = policy_evaluation(uniform_policy)
    print("Policy evaluation (iterative) result:")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {iter_V[i]:.6f}")
    print()

    print(f"  {'State':<8s}{'Manual':<16s}{'Iterative':<16s}{'Error':<12s}")
    for i in range(N_STATES):
        error = abs(manual_V[i] - iter_V[i])
        print(f"  s{i:<6d}{manual_V[i]:<16.8f}{iter_V[i]:<16.8f}{error:<12.2e}")
    print(f"\n  Consistent: {np.allclose(manual_V, iter_V, atol=1e-6)}")
    print()

    # Bellman optimality equation: value iteration
    V_star, optimal_policy, _ = value_iteration()
    action_names = ['a0', 'a1']

    print("Value iteration (optimal) result:")
    print(f"  {'State':<8s}{'V^π(s)':<16s}{'V*(s)':<16s}{'Gain':<12s}")
    for i in range(N_STATES):
        gain = V_star[i] - iter_V[i]
        print(f"  s{i:<6d}{iter_V[i]:<16.6f}{V_star[i]:<16.6f}{gain:>+11.6f}")
    print()

    print("Optimal policy π*:")
    for s in range(N_STATES):
        best = int(np.argmax(optimal_policy[s]))
        print(f"  π*(s{s}) = {action_names[best]}")
    print()


if __name__ == "__main__":
    main()
