import numpy as np
import matplotlib.pyplot as plt
import imageio 
import io



# Define double integrator system variables
def double_integrator():
    A = np.array([[0, 1], [0, 0]])  # Continuous-time state transition matrix
    B = np.array([[0], [1]])  # Continuous-time control matrix
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
states = [
    np.array([x, v]) for x in np.linspace(-10, 10, 20) for v in np.linspace(-10, 10, 20)
]
actions = np.linspace(-1, 1, 21)

# Initialize Value Function and policy arrays
J = [np.zeros(len(states)) for _ in range(num_time_steps + 1)]
u_opt = [np.zeros(len(states)) for _ in range(num_time_steps)]

# Terminal Cost
J[num_time_steps] = np.array(
    [terminal_cost_function(state.reshape(-1, 1), Q_f) for state in states]
).flatten()

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
            min_cost = float("inf")
            best_action = None

            for action in actions:
                action_vec = np.array([[action]])
                # Calculate the next state using the Euler method for continuous-time dynamics
                next_state = (
                    state + (A @ state + B @ action_vec) * dt
                )  # euler integration to find the next state
                next_state = next_state.flatten()  # Flatten for indexing
                next_state_index = np.argmin(
                    np.linalg.norm(np.array(states) - next_state, axis=1)
                )

                # Debug prints
                immediate_cost = cost_function(state, action_vec, Q, R)
                cost_to_go = J[t + 1][next_state_index]
               

                cost = immediate_cost + cost_to_go

                if cost < min_cost:
                    min_cost = cost
                    best_action = action

            J_new[t][i] = float(min_cost)  # Convert to float to ensure it's a scalar
            u_opt[t][i] = best_action

    # Convergence Check
    if (
        np.max(np.abs(np.array(J_new) - np.array(J))) < tolerance
    ):  # Updated convergence check
        J = [arr.copy for arr in J_new]
        print(f"Converged after {iteration + 1} iterations")
        break

    J = [arr.copy() for arr in J_new]

print("Optimal cost-to-go J:", J)
print("Optimal policy u:", u_opt)


pi_star = u_opt


def simulate_trajectory(intial_state, num_steps):
    """
    Simulates the trajectory of a dynamic system over a specified number of steps.

    Parameters
    ----------
    initial_state : array-like
        The initial state of the system.
    num_steps : int
        The number of time steps to simulate.

    Returns
    -------
    np.ndarray
        An array containing the simulated trajectory of the system.
    """
    trajectory = [intial_state]
    for t in range(num_steps):
        current_state = trajectory[-1]
        state_index = np.argmin
        state_index = np.argmin(np.linalg.norm(np.array(states) - current_state, axis=1))
        control = pi_star[t][state_index]
        next_state = current_state + (A @ current_state.reshape(-1, 1) + B * control).flatten() * dt
        trajectory.append(next_state)
    return np.array(trajectory)



X, V = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
U =np.array(pi_star[0]).reshape(20, 20)
cmap = plt.cm.get_cmap('coolwarm')
norm =plt.Normalize(U.min(), U.max())

initial_states = [
    np.array([-8, 5]),
   
]

trajectories = [simulate_trajectory(initial_state, num_time_steps) for initial_state in initial_states]

# Create frames for the GIF
# Simulate trajectories
num_trajectories = 4
initial_states = [
    np.array([-8, 5]),
    np.array([8, -5]),
    np.array([0, 9]),
    np.array([-5, -8])
]

trajectories = []
for initial_state in initial_states:
    trajectory = [initial_state]
    state = initial_state
    for t in range(num_time_steps):
        state_index = np.argmin(np.linalg.norm(np.array(states) - state, axis=1))
        control = pi_star[t][state_index]
        next_state = state + (A @ state.reshape(-1, 1) + B * control).flatten() * dt
        trajectory.append(next_state)
    trajectories.append(np.array(trajectory))

# Create frames for the GIF
frames = []
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

time_steps = np.arange(0, (num_time_steps + 1) * dt, dt)

for step in range(num_time_steps + 1):
    ax.clear()
    
    # Plot position vs time for each trajectory
    for i, trajectory in enumerate(trajectories):
        ax.plot(time_steps[:step+1], trajectory[:step+1, 0], '-o', linewidth=2, markersize=4, label=f'Particle {i+1}')
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    ax.set_title(f'Particle Positions Over Time (Step {step})')
    ax.set_xlim(0, num_time_steps * dt)
    ax.set_ylim(min(traj[:, 0].min() for traj in trajectories) - 1, 
                max(traj[:, 0].max() for traj in trajectories) + 1)
    ax.axhline(y=0, color='r', linestyle='--', label='Reference')
    ax.legend()
    ax.grid(True)
    
    # Save the figure to a byte buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Read the image from the buffer
    img = imageio.imread(buf)
    frames.append(img)

plt.close(fig)

# Save as GIF
imageio.mimsave('particle_trajectories_over_time.gif', frames, fps=5)

print("Animation has been saved as 'particle_trajectories_over_time.gif'")