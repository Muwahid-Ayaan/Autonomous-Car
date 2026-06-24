# Histogram
histogram = np.sum(mask[420:, :], axis=0)

midpoint = histogram.shape[0] // 2
left_base = np.argmax(histogram[:midpoint])
right_base = np.argmax(histogram[midpoint:]) + midpoint

print(f"left_base={left_base}, right_base={right_base}")
print(f"hist max={np.max(histogram)}")
print(f"white pixels={cv2.countNonZero(mask)}")

# Debug histogram bases
debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
cv2.line(debug_img, (left_base, 0), (left_base, 480), (0,255,0), 2)
cv2.line(debug_img, (right_base, 0), (right_base, 480), (0,255,0), 2)
cv2.imshow("Histogram Bases", debug_img)

# Sliding window
y = 472

leftx_cords = []
rightx_cords = []

mask_copy = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

while y > 0:

    # ---------------- LEFT WINDOW ----------------

    lx1 = max(0, left_base - 75)
    lx2 = min(mask.shape[1], left_base + 75)

    left_img = mask[y-40:y, lx1:lx2]

    contours, _ = cv2.findContours(
        left_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 20:

            M = cv2.moments(largest)

            if M["m00"] > 0:

                cx = int(M["m10"] / M["m00"])

                left_base = lx1 + cx
                leftx_cords.append(left_base)

    # ---------------- RIGHT WINDOW ----------------

    rx1 = max(0, right_base - 75)
    rx2 = min(mask.shape[1], right_base + 75)

    right_img = mask[y-40:y, rx1:rx2]

    contours, _ = cv2.findContours(
        right_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 20:

            M = cv2.moments(largest)

            if M["m00"] > 0:

                cx = int(M["m10"] / M["m00"])

                right_base = rx1 + cx
                rightx_cords.append(right_base)

    # Draw windows

    cv2.rectangle(
        mask_copy,
        (lx1, y-40),
        (lx2, y),
        (255,255,255),
        2
    )

    cv2.rectangle(
        mask_copy,
        (rx1, y-40),
        (rx2, y),
        (255,255,255),
        2
    )

    y -= 40

cv2.imshow("Sliding Window", mask_copy)