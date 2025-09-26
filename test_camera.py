import cv2

cap = cv2.VideoCapture(0)  # Try to open default webcam
if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

print("✅ Camera opened! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    cv2.imshow("Test Camera", frame)

    # Press Q to quit window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


