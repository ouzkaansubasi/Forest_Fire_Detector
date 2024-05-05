import cv2
import numpy as np
import onnxruntime
from yolov8.utils import xywh2xyxy
import time
import smtplib
from email.mime.text import MIMEText

# Email configuration
sender_email = "your_sender_email@example.com"
sender_password = "your_sender_password"
receiver_email = "your_receiver_email@example.com"
smtp_server = "smtp.example.com"
smtp_port = 587

# Load the YOLOv8 model and configure thresholds
model_path = "C:\\Users\\ouzkaan\\Desktop\\yangin\\best.onnx"
session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
conf_threshold = 0.4
iou_threshold = 0

# Get input and output details
input_details = session.get_inputs()
output_details = session.get_outputs()
input_name = input_details[0].name
input_shape = input_details[0].name
output_names = [output_detail.name for output_detail in output_details]

# Start capturing video from camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    img = frame
    start = time.perf_counter()

    # Prepare the input image for inference
    img_height, img_width, _ = frame.shape
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (800, 800))
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    # Perform inference on the input image
    outputs = session.run(output_names, {input_name: input_tensor})
    predictions = np.squeeze(outputs[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        cv2.imshow("Object Detection", frame)
        continue

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = predictions[:, :4]
    boxes = np.divide(boxes, np.array([800, 800, 800, 800], dtype=np.float32))
    boxes = xywh2xyxy(boxes)
    boxes *= np.array([img_width, img_height, img_width, img_height], dtype=np.float32)
    boxes = boxes.astype(np.int32)
    boxes = np.array([box[[1, 0, 3, 2]] for box in boxes])  # Convert boxes to (ymin, xmin, ymax, xmax) format

    # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    print(indices)

    # Draw bounding boxes and labels on the input image
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
	    label = f"Class {class_id} ({score:.2f})"
        cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[1], box[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Send email notification
        subject = "Object Detected"
        message = f"An object of class {class_id} with confidence {score:.2f} has been detected."
        email_text = f"Subject: {subject}\n\n{message}"
        
        # Connect to SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, email_text)
            print("Email notification sent!")

    # Show the output frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()