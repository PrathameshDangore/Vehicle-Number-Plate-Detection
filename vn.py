import cv2
import numpy as np
import imutils
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edged = cv2.Canny(gray, 100, 200)

    # Find contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate = None
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            plate = approx
            break

    if plate is not None:
        x, y, w, h = cv2.boundingRect(plate)
        plate_img = frame[y:y+h, x:x+w]

        # OCR to extract text
        text = reader.readtext(plate_img, detail=0)

        # Draw rectangle and display text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "".join(text), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Vehicle Number Plate Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 