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
for i = num_time_steps - 1 down to 0:
    min_cost = infinity
    optimal_u = None
    
    for possible_u in possible_control_inputs:
        # Compute the next state using system dynamics
        x_next = A * x[i] + B * possible_u * dt
        
        # Compute the immediate cost
        immediate_cost = cost_function(x[i], possible_u, Q, R)
        
        # Compute the cost-to-go
        cost_to_go = immediate_cost + J[i + 1]
        
        # Update the value function and optimal control if current cost is lower
        if cost_to_go < min_cost:
            min_cost = cost_to_go
            optimal_u = possible_u
    
    # Store the minimum cost and optimal control for the current step
    J[i] = min_cost
    u_opt[i] = optimal_u

'''