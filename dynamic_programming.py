import numpy as np

# Define double integrator system variables
def double_integrator():
    A = np.array([[0, 1], [0, 0]])  # Continuous-time state transition matrix
    B = np.array([[0], [1]])        # Continuous-time control matrix
    C = np.eye(2)
    D = np.zeros((2, 1))
    return A, B, C, D

# Define Cost Function
def cost_function(state, control, Q, R):
    state_cost = state.T @ Q @ state
    control_cost = control.T @ R @ control
    return state_cost + control_cost

# Define Terminal Cost Function
def terminal_cost_function(terminal_state, Q_f):
    return terminal_state.T @ Q_f @ terminal_state

# System parameters
dt = 0.1  # Time step
A, B, _, _ = double_integrator()
Q = np.eye(2)  # State cost matrix
R = np.array([[1]])  # Control cost matrix (1x1 for scalar control)
Q_f = np.eye(2)  # Terminal state cost matrix
num_time_steps = 10

# Define State and Actions
states = [np.array([x, v]) for x in np.linspace(-10, 10, 20) for v in np.linspace(-10, 10, 20)]
actions = np.linspace(-1, 1, 21)

# Initialize Value Function and policy arrays
J = [np.zeros(len(states)) for _ in range(num_time_steps + 1)]
u_opt = [np.zeros(len(states)) for _ in range(num_time_steps)]

# Terminal Cost
J[num_time_steps] = np.array([terminal_cost_function(state.reshape(-1, 1), Q_f) for state in states]).flatten()

# Value Iteration Algorithm
tolerance = 1e-3
max_iterations = 1000

for iteration in range(max_iterations):
    J_new = [np.zeros(len(states)) for _ in range(num_time_steps + 1)]   
    J_new[num_time_steps] = J[num_time_steps].copy()  # set to the terminal cost

    for t in range(num_time_steps - 1, -1, -1):
        J_new[t] = np.zeros((len(states)))  # Initialize to zeros for current time step
        u_opt[t] = np.zeros((len(states)))
        for i, state in enumerate(states):
            state = state.reshape(-1, 1)  # Ensure state is a column vector
            min_cost = float('inf')
            best_action = None

            for action in actions:
                action_vec = np.array([[action]])
                # Calculate the next state using the Euler method for continuous-time dynamics
                next_state = state + (A @ state + B @ action_vec) * dt # euler integration to find the next state 
                next_state = next_state.flatten()  # Flatten for indexing
                next_state_index = np.argmin(np.linalg.norm(np.array(states) - next_state, axis=1))
                cost = cost_function(state, action_vec, Q, R) + J[t + 1][next_state_index]  # cost of being at the current state + Cost to go
                
                if cost < min_cost:
                    min_cost = cost
                    best_action = action
                print(cost  )
            
            J_new[t][i] = min_cost
            u_opt[t] = best_action

    print(J_new[t])
    # Convergence Check
    if np.max(np.abs(np.array(J_new) - np.array(J))) < tolerance:  # Updated convergence check
        J = [arr.copy for arr in J_new]
        print(f'Converged after {iteration + 1} iterations')
        break
    
    J = [arr.copy() for arr in J_new]

print("Optimal cost-to-go J:", J)
print("Optimal policy u:", u_opt)


#TODO: Implement opitmal control policy extraction
# Extract the optimal policy


# Extract the optimal policy
pi_star = [np.zeros(len(states)) for _ in range(num_time_steps + 1)]

for i, state in enumerate(states):
    min_cost = float('inf')
    best_action = None 

    for action in actions:
        next_state = state + (A @ state + B @ state )*dt 
       # next_state =  ß
#for i, state in enumerate(states):
 #   min_cost = infinity
  #  best_action = None
    
    f#or action in actions:
      #  next_state = A @ state + B * action * dt
      #  next_state_index = index_of_closest_state(states, next_state)
        
      #  cost = cost_function(state, action, Q, R) + J[0][next_state_index]
        
       # if cost < min_cost:
       #     min_cost = cost
        #    best_action = action
    
   # pi_star[i] = best_action

