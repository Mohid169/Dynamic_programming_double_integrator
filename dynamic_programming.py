import numpy as np

#Define double integrator system variables
def double_integrator():
    A = np.array([[0 ,1], [0, 0]])
    B = np.array([0, 1])
    C = np.eye(2)
    D = np.zeros((2,1))
    return LinearSystem(A,B,C,D)

#Define Cost Function
def cost_function(state, control, Q, R):
    state_cost = state.T @ Q @ state
    control_cost = control.T @ R @ control
    return state_cost + control_cost

#Define Terminal Cost Function
def terminal_cost_function(terminal_state, Q_f):
    return terminal_state.T @ Q_f @ terminal_state

# System parameters
A, B, _, _ = double_integrator()
Q = np.eye(2)  # State cost matrix
R = np.eye(1)  # Control cost matrix
Q_f = np.eye(2)  # Terminal state cost matrix
num_time_steps = 100
dt = 0.1  # Time step

#Initialize Value Function and policy arrays
J = [None] * (num_time_steps + 1)
u_opt = [None] * num_time_steps

#Terminal Cost 
J[num_time_steps] = np.zeros((2,2))

#Backward dynamic programming recursion

'''
# Backward Recursion
Initialize J_hat to zero for all states
Repeat until convergence:
    For each state s_i:
        For each action a:
            Compute cost = l(s_i, a) + J_hat(f(s_i, a))
        J_hat(s_i) = min(cost over all actions a)
Extract the optimal policy:
    For each state s_i:
        pi_star(s_i) = argmin_a [l(s_i, a) + J_hat(f(s_i, a))]


'''