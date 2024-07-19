import numpy as np
import matplotlib.pyplot as plt


#Define double integrator system variables
def double_integrator():
    A = np.array([[0 ,1], [0, 0]])
    B = np.array([0, 1])
    C = np.eye(2)
    D = np.zeros((2,1))
    return A, B , C, D  

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
R = np.array([[1]])  # Control cost matrix (1x1 for scalar control)
Q_f = np.eye(2)  # Terminal state cost matrix
num_time_steps = 10
dt = 0.1  # Time step

#Initialize Value Function and policy arrays
J = [None] * (num_time_steps + 1)
u_opt = [None] * num_time_steps

#Define State and Actions
states = [np.array([x, v]) for x in np.linspace(-10, 10, 100) for v in np.linspace(-10,10, 100)]
actions = np.linspace(-1, 1, 21)

#Terminal Cost 
J[num_time_steps] = np.zeros((len(states),1))


#Value Iteration Algorithm 
tolerance = 1e-3
max_iterations = int(num_time_steps / dt)


for iteration in range(max_iterations):
    J_new = [None] * (num_time_steps + 1) 
    J_new[num_time_steps] = J[num_time_steps] # set to the terminal cost

    for t in range(num_time_steps-1, -1, -1):
        J_new[t] = np.zeros((len(states,), 1 )) # Initialize to zeros for current time step
        
        for i, state in enumerate(states):
            min_cost = float('inf')
            best_action = None

            for action in actions:
                action_vec = np.array([[action]])
                next_state = A @ state +  B * action *dt# Apply equation of motion 
                next_state_index = np.argmin( np.linalg.norm(np.array(states)-np.array(next_state), axis =1 ))
                cost = cost_function(state, action_vec, Q, R)  + J[t+1][next_state_index]  #cost of being at the current state + Cost to go
                print(cost)
                if cost < min_cost:
                    min_cost = cost
                    best_action = action
                
                J_new[t][i] = min_cost
                u_opt[t] = best_action
    
  
    #TODO:  Check convergence (optional)
    # Check for convergence
    #if max(abs(J_new[t] - J[t])) < tolerance for t in range(num_time_steps):
     #   J = J_new
      #  break
    
    #J = J_new

#TODO: Implement opitmal control policy extraction
'''
psuedo code:   for iteration in range(max_iterations):
    J_new = array of size (num_time_steps + 1), initialized to None
    J_new[num_time_steps] = J[num_time_steps]
    
    for t in range(num_time_steps - 1, -1, -1):
        J_new[t] = zero_matrix(len(states), 1)
        
        for i, state in enumerate(states):
            min_cost = infinity
            best_action = None
            
            for action in actions:
                next_state = A @ state + B * action * dt
                next_state_index = index_of_closest_state(states, next_state)
                
                cost = cost_function(state, action, Q, R) + J[t + 1][next_state_index]
                
                if cost < min_cost:
                    min_cost = cost
                    best_action = action
            
            J_new[t][i] = min_cost
            u_opt[t] = best_action
    
    # Check for convergence
    if max(abs(J_new[t] - J[t])) < tolerance for t in range(num_time_steps):
        J = J_new
        break
    
    J = J_new

# Extract the optimal policy
pi_star = array of size len(states), initialized to None

for i, state in enumerate(states):
    min_cost = infinity
    best_action = None
    
    for action in actions:
        next_state = A @ state + B * action * dt
        next_state_index = index_of_closest_state(states, next_state)
        
        cost = cost_function(state, action, Q, R) + J[0][next_state_index]
        
        if cost < min_cost:
            min_cost = cost
            best_action = action
    
    pi_star[i] = best_action


'''