import numpy as np
import cv2
from scipy.optimize import fsolve


class Ball:
    def __init__(self, position, radius=0):
        self.position = np.array(position, dtype=np.float64)
        self.radius = radius
        self.hit = False


class Projectile:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array([0, 0], dtype=np.float64)
        self.radius = 5
        self.in_motion = False
        self.current_path = []


def detect_balls_and_move(image_path):
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        return

    # Constants
    GRAVITY = 9.81
    DT = 0.1

    def detect_balls(frame):
        """Detect balls using enhanced method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Clean up image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        balls = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Filter small contours
                continue

            # Fit circle
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # Calculate circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if circularity > 0.7:  # Filter non-circular objects
                balls.append(Ball((int(x), int(y)), int(radius)))

        return balls, thresh

    def calculate_throw_velocity(start_pos, target_pos):
        """Calculate velocity needed to hit target"""
        dx = target_pos[0] - start_pos[0]
        dy = target_pos[1] - start_pos[1]

        # Adjust these parameters to control throw characteristics
        time_to_target = np.sqrt(2 * abs(dy) / GRAVITY)

        vx = dx / time_to_target
        vy = dy / time_to_target - 0.5 * GRAVITY * time_to_target

        return np.array([vx, vy])

    def calculate_trajectory(start_pos, velocity, max_steps=100):
        """Calculate complete trajectory path"""
        path = []
        pos = np.array(start_pos, dtype=np.float64)
        vel = np.array(velocity, dtype=np.float64)

        for _ in range(max_steps):
            path.append(pos.copy())
            pos += vel * DT
            vel[1] += GRAVITY * DT

        return np.array(path)

    def update_projectile(projectile):
        """Update projectile position and velocity"""
        projectile.position += projectile.velocity * DT
        projectile.velocity[1] += GRAVITY * DT
        projectile.current_path.append(projectile.position.copy())

    def check_hit(projectile, ball):
        """Check if projectile hits ball"""
        distance = np.linalg.norm(projectile.position - ball.position)
        return distance < (projectile.radius + ball.radius)

    # Detect balls
    balls, thresh = detect_balls(frame)
    if not balls:
        print("No balls detected!")
        return

    # Sort balls from left to right for sequential hitting
    balls.sort(key=lambda b: b.position[0])

    # Initialize starting position (bottom center)
    height, width = frame.shape[:2]
    start_pos = np.array([width // 2, height - 50], dtype=np.float64)

    # Initialize projectile
    projectile = Projectile(start_pos.copy())
    current_target = 0

    # Create window and video writer
    cv2.namedWindow('Ball Hitting Simulation')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (width, height))

    while True:
        frame_copy = frame.copy()

        # Draw edges (optional, for visualization)
        edges = cv2.Canny(frame, 50, 150)
        edge_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        frame_copy = cv2.addWeighted(frame_copy, 0.7, edge_overlay, 0.3, 0)

        # Draw balls
        for idx, ball in enumerate(balls):
            color = (0, 100, 0) if ball.hit else (0, 255, 0)
            cv2.circle(frame_copy, tuple(ball.position.astype(int)), ball.radius, color, 2)
            cv2.circle(frame_copy, tuple(ball.position.astype(int)), 2, (255, 0, 0), -1)

        # Handle throwing motion
        if current_target < len(balls):
            if not projectile.in_motion:
                # Always throw from the starting position
                projectile.position = start_pos.copy()
                target_pos = balls[current_target].position
                projectile.velocity = calculate_throw_velocity(start_pos, target_pos)
                projectile.in_motion = True
                projectile.current_path = [projectile.position.copy()]

            # Update projectile
            update_projectile(projectile)

            # Draw projectile and its path
            if len(projectile.current_path) > 1:
                path = np.array(projectile.current_path, dtype=np.int32)
                for i in range(len(path) - 1):
                    cv2.line(frame_copy, tuple(path[i]), tuple(path[i + 1]), (255, 0, 0), 1)

            cv2.circle(frame_copy, tuple(projectile.position.astype(int)),
                       projectile.radius, (255, 0, 255), -1)

            # Check for hit
            if check_hit(projectile, balls[current_target]):
                balls[current_target].hit = True
                projectile.in_motion = False
                current_target += 1

            # Check if projectile is out of bounds
            elif (projectile.position[0] < 0 or projectile.position[0] > width or
                  projectile.position[1] < 0 or projectile.position[1] > height):
                projectile.in_motion = False

        # Display frame
        cv2.imshow('Ball Hitting Simulation', frame_copy)
        out.write(frame_copy)

        if current_target >= len(balls):
            print("All balls hit!")
            break

        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    out.release()


# Run simulation
image_path = r"C:\Users\kvini\PycharmProjects\PythonProject3\.venv\Scripts\NP Projects\Test Cases\2.jpg "
detect_balls_and_move(image_path)