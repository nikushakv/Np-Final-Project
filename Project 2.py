import numpy as np
import cv2
from scipy.integrate import solve_ivp
from random import uniform
from time import sleep
import matplotlib.pyplot as plt


class TrajectoryPhysics:
    def __init__(self, object_mass=0.145, resistance_coef=0.47, cross_section=0.0042, fluid_density=1.225):
        self.gravity = 9.81
        self.object_mass = object_mass
        self.resistance_coef = resistance_coef
        self.cross_section = cross_section
        self.fluid_density = fluid_density
        self.resistance_factor = 0.5 * self.fluid_density * self.resistance_coef * self.cross_section

    def calculate_final_conditions(self, location_history, time_history):
        location_history = np.array(location_history)
        time_history = np.array(time_history)
        velocity_history = np.diff(location_history, axis=0) / np.diff(time_history)[:, None]

        if not velocity_history.size:
            raise ValueError("Insufficient tracking data for velocity calculation")

        self.terminal_velocity = velocity_history[-1]
        self.terminal_position = location_history[-1]

    def motion_equations(self, t, state_vector):
        pos_x, pos_y, vel_x, vel_y = state_vector
        velocity_magnitude = np.sqrt(vel_x ** 2 + vel_y ** 2)

        d_pos_x = vel_x
        d_pos_y = vel_y
        d_vel_x = -(self.resistance_factor / self.object_mass) * vel_x * velocity_magnitude
        d_vel_y = self.gravity - (self.resistance_factor / self.object_mass) * vel_y * velocity_magnitude

        return [d_pos_x, d_pos_y, d_vel_x, d_vel_y]

    def surface_collision(self, t, state):
        return state[1] - surface_height

    surface_collision.terminal = True
    surface_collision.direction = 1

    def forward_euler_integration(self, initial_state, time_range, step_size):
        time_points = np.arange(0, time_range, step_size)
        path = np.zeros((len(time_points), 4))
        path[0] = initial_state

        for idx in range(1, len(time_points)):
            derivatives = self.motion_equations(time_points[idx - 1], path[idx - 1])
            path[idx] = path[idx - 1] + np.array(derivatives) * step_size

            if path[idx, 1] >= surface_height:
                return time_points[:idx + 1], path[:idx + 1]

        return time_points, path

    def runge_kutta_integration(self, initial_state, time_range, step_size):
        time_points = np.arange(0, time_range, step_size)
        path = np.zeros((len(time_points), 4))
        path[0] = initial_state

        for idx in range(1, len(time_points)):
            t = time_points[idx - 1]
            current_state = path[idx - 1]

            k1 = np.array(self.motion_equations(t, current_state))
            k2 = np.array(self.motion_equations(t + step_size / 2, current_state + step_size * k1 / 2))
            k3 = np.array(self.motion_equations(t + step_size / 2, current_state + step_size * k2 / 2))
            k4 = np.array(self.motion_equations(t + step_size, current_state + step_size * k3))

            path[idx] = current_state + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            if path[idx, 1] >= surface_height:
                return time_points[:idx + 1], path[:idx + 1]

        return time_points, path

    def calculate_trajectory(self, initial_state, time_range, step_size=0.01):
        evaluation_points = np.arange(0, time_range, step_size)
        solution = solve_ivp(
            self.motion_equations,
            [0, time_range],
            initial_state,
            t_eval=evaluation_points,
            events=self.surface_collision,
            max_step=step_size
        )

        if solution.t_events[0].size > 0:
            collision_time = solution.t_events[0][0]
            idx = np.searchsorted(solution.t, collision_time)
            return solution.t[:idx + 1], solution.y.T[:idx + 1]
        return solution.t, solution.y.T


class ObjectTracker:
    def __init__(self, video_source):
        self.capture = cv2.VideoCapture(video_source)
        if not self.capture.isOpened():
            raise ValueError(f"Failed to open video source: {video_source}")

        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_interval = int(1000 / self.frame_rate)
        self.reference_frame = None
        self.location_history = []
        self.time_history = []
        self.current_frame = None

    def track_object(self, frame):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (21, 21), 0)

        if self.reference_frame is None:
            self.reference_frame = blurred
            return None

        frame_delta = cv2.absdiff(self.reference_frame, blurred)
        threshold = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

        kernel = np.ones((5, 5), np.uint8)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:
                center, _ = cv2.minEnclosingCircle(largest_contour)
                return (int(center[0]), int(center[1]))
        return None


def visualize_numerical_methods(physics, initial_state, time_range, step_size=0.01):
    t_euler, path_euler = physics.forward_euler_integration(initial_state, time_range, step_size)
    t_rk4, path_rk4 = physics.runge_kutta_integration(initial_state, time_range, step_size)
    t_ivp, path_ivp = physics.calculate_trajectory(initial_state, time_range, step_size)

    # Customizing plot style to enhance uniqueness
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        print("Fallback to default style as 'seaborn-darkgrid' is unavailable.")
        plt.style.use('default')

    fig, axs = plt.subplots(2, 1, figsize=(16, 8), facecolor='#EAEAF2')

    # Height comparison with unique styles
    axs[0].set_facecolor('#FFFFFF')
    axs[0].plot(t_euler, path_euler[:, 1], color='#1f77b4', label='Euler Method', linewidth=2.5, linestyle='-.')
    axs[0].plot(t_rk4, path_rk4[:, 1], color='#ff7f0e', label='RK4 Method', linewidth=2.5, linestyle=':')
    axs[0].plot(t_ivp, path_ivp[:, 1], color='#2ca02c', label='solve_ivp', linewidth=2.5, linestyle='--')
    axs[0].set_xlabel('Time (s)', fontsize=13)
    axs[0].set_ylabel('Height (m)', fontsize=13)
    axs[0].set_title('Comparison of Heights Using Different Methods', fontsize=15, pad=15)
    axs[0].legend(loc='upper right', fontsize=11)
    axs[0].grid(color='#D3D3D3', linestyle='--', linewidth=0.8)

    # Velocity comparison with unique styles
    v_euler = np.sqrt(path_euler[:, 2] ** 2 + path_euler[:, 3] ** 2)
    v_rk4 = np.sqrt(path_rk4[:, 2] ** 2 + path_rk4[:, 3] ** 2)
    v_ivp = np.sqrt(path_ivp[:, 2] ** 2 + path_ivp[:, 3] ** 2)

    axs[1].set_facecolor('#FFFFFF')
    axs[1].plot(t_euler, v_euler, color='#1f77b4', label='Euler Method', linewidth=2.5, linestyle='-.')
    axs[1].plot(t_rk4, v_rk4, color='#ff7f0e', label='RK4 Method', linewidth=2.5, linestyle=':')
    axs[1].plot(t_ivp, v_ivp, color='#2ca02c', label='solve_ivp', linewidth=2.5, linestyle='--')
    axs[1].set_xlabel('Time (s)', fontsize=13)
    axs[1].set_ylabel('Velocity (m/s)', fontsize=13)
    axs[1].set_title('Comparison of Velocities Using Different Methods', fontsize=15, pad=15)
    axs[1].legend(loc='upper right', fontsize=11)
    axs[1].grid(color='#D3D3D3', linestyle='--', linewidth=0.8)

    plt.tight_layout(pad=3.0)
    plt.show()

    # Calculate and display deviations
    min_len = min(len(path_euler), len(path_ivp))
    euler_deviation = np.mean(np.abs(path_euler[:min_len] - path_ivp[:min_len]))

    min_len = min(len(path_rk4), len(path_ivp))
    rk4_deviation = np.mean(np.abs(path_rk4[:min_len] - path_ivp[:min_len]))

    print("\nNumerical Analysis:")
    print(f"Mean deviation for Euler method: {euler_deviation:.6f}")
    print(f"Mean deviation for RK4 method: {rk4_deviation:.6f}")

    return {
        'euler': (t_euler, path_euler),
        'rk4': (t_rk4, path_rk4),
        'ivp': (t_ivp, path_ivp)
    }

def simulate_trajectories(video_source=r"C:\Users\kvini\PycharmProjects\PythonProject3\.venv\Scripts\NP Projects\Test Cases\Test.mp4"):
    try:
        tracker = ObjectTracker(video_source)
    except ValueError as e:
        print(f"Error: {e}")
        return

    tracked_physics = TrajectoryPhysics()
    interceptor_physics = TrajectoryPhysics(resistance_coef=0.0)
    print("Beginning video analysis...")

    scale = 0.01
    global surface_height
    surface_height = tracker.frame_height * scale

    # Updated window settings
    cv2.namedWindow('Enhanced Simulation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Enhanced Simulation', 1024, 768)  # Increased window size

    frame_index = 0
    processed_sequence = []

    # Define custom colors (BGR format)
    TRACKED_BALL_COLOR = (255, 128, 0)  # Orange
    TRACKED_TRAIL_COLOR = (200, 100, 0)  # Light Orange
    INTERCEPTOR_COLOR = (255, 50, 50)  # Pink
    INTERCEPTOR_TRAIL_COLOR = (200, 40, 40)  # Light Pink
    START_POINT_COLOR = (0, 255, 255)  # Yellow
    IMPACT_POINT_COLOR = (255, 255, 0)  # Cyan
    TEXT_COLOR = (255, 255, 255)  # White

    while True:
        success, frame = tracker.capture.read()
        if not success:
            break

        frame_copy = frame.copy()
        tracker.current_frame = frame_copy

        object_position = tracker.track_object(frame)
        if object_position:
            scaled_position = (object_position[0] * scale, object_position[1] * scale)
            tracker.location_history.append(scaled_position)
            tracker.time_history.append(frame_index / tracker.frame_rate)

            # Draw current position and trail with new colors
            cv2.circle(frame_copy, object_position, 12, TRACKED_BALL_COLOR, -1)
            for prev_pos in tracker.location_history:
                prev_pixel = (int(prev_pos[0] / scale), int(prev_pos[1] / scale))
                cv2.circle(frame_copy, prev_pixel, 3, TRACKED_TRAIL_COLOR, -1)

        processed_sequence.append(frame_copy)
        frame_index += 1

    if len(tracker.location_history) < 3:
        print("Error: Insufficient tracking data")
        return

    print("Computing physics parameters...")
    tracked_physics.calculate_final_conditions(tracker.location_history, tracker.time_history)

    initial_conditions = [
        tracked_physics.terminal_position[0],
        tracked_physics.terminal_position[1],
        tracked_physics.terminal_velocity[0],
        tracked_physics.terminal_velocity[1]
    ]

    print("Analyzing numerical methods...")
    method_results = visualize_numerical_methods(tracked_physics, initial_conditions, 5.0)

    print("Generating trajectories...")
    projection_time = 5.0
    future_times, future_positions = tracked_physics.calculate_trajectory(
        initial_conditions, projection_time)

    if not len(future_times):
        print("Error: Trajectory calculation failed.")
        return

    impact_time = future_times[-1]
    impact_location = future_positions[-1][:2]

    interceptor_x = uniform(
        0.1 * tracker.frame_width * scale,
        0.9 * tracker.frame_width * scale
    )
    interceptor_y = uniform(
        0.1 * tracker.frame_height * scale,
        0.9 * tracker.frame_height * scale
    )

    dx = impact_location[0] - interceptor_x
    dy = impact_location[1] - interceptor_y - 0.5 * interceptor_physics.gravity * impact_time ** 2
    initial_vx = dx / impact_time
    initial_vy = dy / impact_time

    interceptor_start = [interceptor_x, interceptor_y, initial_vx, initial_vy]
    interceptor_times, interceptor_positions = interceptor_physics.calculate_trajectory(
        interceptor_start, impact_time)

    min_points = min(len(future_positions), len(interceptor_positions))
    future_positions = future_positions[:min_points]
    interceptor_positions = interceptor_positions[:min_points]
    future_times = future_times[:min_points]

    print("Starting enhanced visualization...")
    frame_interval = int(1000 / tracker.frame_rate)
    animation_frames = []

    for frame_idx, base_frame in enumerate(processed_sequence):
        enhanced_frame = base_frame.copy()
        current_time = frame_idx / tracker.frame_rate

        # Draw reference information
        cv2.putText(enhanced_frame, f"Time: {current_time:.2f}s",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

        # Draw start points
        start_pixel = (
            int(initial_conditions[0] / scale),
            int(initial_conditions[1] / scale)
        )
        interceptor_start_pixel = (
            int(interceptor_x / scale),
            int(interceptor_y / scale)
        )
        cv2.circle(enhanced_frame, start_pixel, 8, START_POINT_COLOR, -1)
        cv2.circle(enhanced_frame, interceptor_start_pixel, 8, START_POINT_COLOR, -1)

        # Draw impact point
        impact_pixel = (
            int(impact_location[0] / scale),
            int(impact_location[1] / scale)
        )
        cv2.circle(enhanced_frame, impact_pixel, 8, IMPACT_POINT_COLOR, -1)

        # Draw projected trajectories
        future_indices = future_times <= current_time
        if any(future_indices):
            future_pixels = [
                (int(pos[0] / scale), int(pos[1] / scale))
                for pos in future_positions[future_indices]
            ]
            for pixel in future_pixels:
                cv2.circle(enhanced_frame, pixel, 3, TRACKED_TRAIL_COLOR, -1)

            if len(future_pixels) > 0:
                current_pos = future_pixels[-1]
                cv2.circle(enhanced_frame, current_pos, 12, TRACKED_BALL_COLOR, -1)

        # Draw interceptor trajectory
        interceptor_indices = interceptor_times <= current_time
        if any(interceptor_indices):
            interceptor_pixels = [
                (int(pos[0] / scale), int(pos[1] / scale))
                for pos in interceptor_positions[interceptor_indices]
            ]
            for pixel in interceptor_pixels:
                cv2.circle(enhanced_frame, pixel, 3, INTERCEPTOR_TRAIL_COLOR, -1)

            if len(interceptor_pixels) > 0:
                current_interceptor = interceptor_pixels[-1]
                cv2.circle(enhanced_frame, current_interceptor, 12, INTERCEPTOR_COLOR, -1)

        # Add telemetry data
        cv2.putText(enhanced_frame,
                    f"Impact Time: {impact_time:.2f}s",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

        animation_frames.append(enhanced_frame)

        # Display frame
        cv2.imshow('Enhanced Simulation', enhanced_frame)
        key = cv2.waitKey(frame_interval) & 0xFF
        if key == 27:  # ESC key
            break

    print("Simulation complete. Press ESC to exit.")

    # Cleanup
    tracker.capture.release()
    cv2.destroyAllWindows()

    return {
        'tracked_physics': tracked_physics,
        'interceptor_physics': interceptor_physics,
        'impact_time': impact_time,
        'impact_location': impact_location,
        'future_trajectory': (future_times, future_positions),
        'interceptor_trajectory': (interceptor_times, interceptor_positions)
    }


if __name__ == "__main__":
    simulation_results = simulate_trajectories()

    #