import numpy as np
import matplotlib.pyplot as plt
import imageio
import io

# Define double integrator system variables
def double_integrator():
    A = np.array([[0, 1], [0, 0]])  # Continuous-time state transition matrix
    B = np.array([[0], [1]])  # Continuous-time control matrix
    return A, B

# Define Cost Function
def cost_function(state, control, Q, R):
    state_cost = state.T @ Q @ state
    control_cost = control.T @ R @ control
    return state_cost + control_cost

# System parameters
dt = 0.1  # Time step
A, B = double_integrator()
Q = np.diag([1, 0.1])  # State cost matrix
R = np.array([[0.1]])  # Control cost matrix
num_time_steps = 100

# Define State and Actions spaces
x_range = np.linspace(-10, 10, 21)
v_range = np.linspace(-10, 10, 21)
states = np.array([(x, v) for x in x_range for v in v_range])
actions = np.linspace(-5, 5, 51)

# Initialize Value Function and policy arrays
V = np.zeros((len(states),))
pi = np.zeros((len(states),))

# Value Iteration Algorithm 
max_iterations = 1000
tolerance = 1e-3

for iteration in range(max_iterations):
    V_old = V.copy()
    
    for i, state in enumerate(states):
        Q_values = []
        for action in actions:
            next_state = state + (A @ state + B.flatten() * action) * dt
            j = np.argmin(np.sum((states - next_state)**2, axis=1))
            q_value = cost_function(state, np.array([action]), Q, R) + V_old[j]
            Q_values.append(q_value)
        
        V[i] = min(Q_values)
        pi[i] = actions[np.argmin(Q_values)]
    
    if np.max(np.abs(V - V_old)) < tolerance:
        print(f"Converged after {iteration + 1} iterations")
        break

# Simulation function
def simulate_trajectory(initial_state, num_steps):
    trajectory = [initial_state]
    controls = []
    state = initial_state
    for t in range(num_steps):
        i = np.argmin(np.sum((states - state)**2, axis=1))
        u = pi[i]
        controls.append(u)
        state = state + (A @ state + B.flatten() * u) * dt
        trajectory.append(state)
    return np.array(trajectory), np.array(controls)

# Simulate trajectories
initial_states = [
    np.array([-8, 5]),
    np.array([8, -5]),
    np.array([0, 9]),
    np.array([-5, -8])
]

trajectories = []
controls = []
for initial_state in initial_states:
    traj, ctrl = simulate_trajectory(initial_state, num_time_steps)
    trajectories.append(traj)
    controls.append(ctrl)

def value_iteration():
    V = np.zeros(len(states))
    pi = np.zeros(len(states))
    
    for iteration in range(max_iterations):
        V_old = V.copy()
        
        for i, state in enumerate(states):
            Q_values = []
            for action in actions:
                next_state = state + (A @ state + B.flatten() * action) * dt
                j = np.argmin(np.sum((states - next_state)**2, axis=1))
                q_value = cost_function(state, np.array([action]), Q, R) + V_old[j]
                Q_values.append(q_value)
            
            V[i] = min(Q_values)
            pi[i] = actions[np.argmin(Q_values)]
        
        if np.max(np.abs(V - V_old)) < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return V, pi

# Visualization code
frames = []
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), dpi=100)

time_steps = np.arange(0, (num_time_steps + 1) * dt, dt)

for step in range(num_time_steps + 1):
    ax1.clear()
    ax2.clear()
    
    # Plot position vs time for each trajectory
    for i, (trajectory, control) in enumerate(zip(trajectories, controls)):
        ax1.plot(time_steps[:step+1], trajectory[:step+1, 0], '-o', linewidth=2, markersize=4, label=f'Particle {i+1}')
        if step > 0:
            ax2.plot(time_steps[1:step+1], control[:step], '-o', linewidth=2, markersize=4, label=f'Particle {i+1}')
    
    # Add labels and title for position plot
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.set_title(f'Particle Positions Over Time (Step {step})')
    ax1.set_xlim(0, num_time_steps * dt)
    ax1.set_ylim(min(traj[:, 0].min() for traj in trajectories) - 1, 
                 max(traj[:, 0].max() for traj in trajectories) + 1)
    ax1.axhline(y=0, color='r', linestyle='--', label='Reference')
    ax1.legend()
    ax1.grid(True)
    
    # Add labels and title for control plot
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Control Input')
    ax2.set_title('Control Inputs Over Time')
    ax2.set_xlim(0, num_time_steps * dt)
    ax2.set_ylim(min(min(c) for c in controls) - 0.5, max(max(c) for c in controls) + 0.5)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the figure to a byte buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    
    # Read the image from the buffer
    img = imageio.imread(buf)
    frames.append(img)

plt.close(fig)

# Save as GIF
imageio.mimsave('particle_trajectories_and_controls.gif', frames, fps=10)

print("Animation has been saved as 'particle_trajectories_and_controls.gif'")