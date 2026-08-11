import cv2
import csv
import os
from datetime import datetime

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load student details
students = {}

with open("students.csv", "r") as file:
    reader = csv.reader(file)

    for row in reader:
        students[int(row[0])] = row[1]


# Create attendance folder
if not os.path.exists("attendance"):
    os.makedirs("attendance")

attendance_file = "attendance/attendance.csv"

# Create file if it doesn't exist
if not os.path.exists(attendance_file):

    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Student ID",
            "Name",
            "Date",
            "Time"
        ])


camera = cv2.VideoCapture(0)

marked_students = set()

print("Starting attendance system...")
print("Press ESC to exit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera not found")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        student_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        # Lower confidence = better match
        if confidence < 70:

            name = students.get(
                student_id,
                "Unknown"
            )

            confidence_text = round(
                100 - confidence
            )

            cv2.putText(
                frame,
                f"{name} ({confidence_text}%)",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Mark attendance only once
            if student_id not in marked_students:

                now = datetime.now()

                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open(
                    attendance_file,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        student_id,
                        name,
                        date,
                        time
                    ])

                marked_students.add(student_id)

                print(
                    f"Attendance marked: {name}"
                )

        else:

            cv2.putText(
                frame,
                "Unknown",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )

    cv2.imshow(
        "Face Recognition Attendance",
        frame
    )

    if cv2.waitKey(1) & 0xff == 27:
        break


camera.release()
cv2.destroyAllWindows()

print("Attendance system stopped.")
