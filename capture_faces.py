import cv2
import os

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Save student details
with open("students.csv", "a") as file:
    file.write(student_id + "," + student_name + "\n")

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

count = 0

print("Look at the camera...")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera not found")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        count += 1

        cv2.imwrite(
            f"dataset/User.{student_id}.{count}.jpg",
            gray[y:y+h, x:x+w]
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(100) & 0xff

    if key == 27 or count >= 30:
        break

camera.release()
cv2.destroyAllWindows()

print("Face data captured successfully!")
