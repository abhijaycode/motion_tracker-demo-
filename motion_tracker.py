

import cv2
import numpy as np
import random

# Open laptop camera
camera = cv2.VideoCapture(0)

# Previous frame
success, previous_frame = camera.read()

if not success:
    print("Could not access camera.")
    exit()

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

# Canvas for our drawings
canvas = np.zeros_like(previous_frame)

# Store particle positions
particles = []

while True:

    success, frame = camera.read()

    if not success:
        break

    # Flip camera so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Find movement
    difference = cv2.absdiff(previous_gray, gray)

    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # Make movement areas bigger
    threshold = cv2.dilate(
        threshold,
        None,
        iterations=2
    )

    # Find movement contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the biggest moving area
    if contours:

        biggest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(biggest) > 500:

            # Get center of movement
            x, y, w, h = cv2.boundingRect(biggest)

            center_x = x + w // 2
            center_y = y + h // 2

            # Add particles around movement
            for i in range(8):

                px = center_x + random.randint(-30, 30)
                py = center_y + random.randint(-30, 30)

                particles.append([
                    px,
                    py,
                    random.randint(-3, 3),
                    random.randint(-3, 3),
                    random.randint(2, 6)
                ])

    # Update particles
    for particle in particles:

        particle[0] += particle[2]
        particle[1] += particle[3]

        # Draw particle
        cv2.circle(
            canvas,
            (particle[0], particle[1]),
            particle[4],
            (255, 0, 255),
            -1
        )

        particle[4] -= 0.15

    # Remove tiny particles
    particles = [
        p for p in particles
        if p[4] > 0
    ]

    # Fade old drawings
    canvas = cv2.addWeighted(
        canvas,
        0.92,
        np.zeros_like(canvas),
        0.08,
        0
    )

    # Combine camera + artwork
    result = cv2.addWeighted(
        frame,
        0.75,
        canvas,
        0.9,
        0
    )

    # Display
    cv2.imshow(
        "Hand Motion Art",
        result
    )

    # Update previous frame
    previous_gray = gray

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
