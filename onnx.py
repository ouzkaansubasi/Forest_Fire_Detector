import cv2
import numpy as np
import onnxruntime
from yolov8.utils import xywh2xyxy
import time
from email.message import EmailMessage
import ssl
import smtplib
import sqlite3 

# Load the YOLOv8 model
model_path = r"C:\Users\oukaa\Desktop\yangin\best.onnx"
session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Get input and output details
input_details = session.get_inputs()
output_details = session.get_outputs()
input_name = input_details[0].name
input_shape = input_details[0].name
output_names = [output_detail.name for output_detail in output_details]

# Define confidence and IOU thresholds
conf_threshold = 0.4
iou_threshold = 0

# Start capturing video from camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame= cap.read()
    img=frame
    start = time.perf_counter()

    # if not ret:
    #     break

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
        # cv2.imshow("Object Detecz5tion",frame)
        
    
        print("we")
        continue
    
    
    elif len(scores) != 1 :
    # Get the class with the highest confidence
     class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
     boxes = predictions[:, :4]
     boxes = np.divide(boxes, np.array([800,800,800,800], dtype=np.float32))
     boxes = xywh2xyxy(boxes)
     boxes *= np.array([img_width, img_height, img_width, img_height], dtype=np.float32)
     boxes = boxes.astype(np.int32)
     boxes = np.array([box[[1, 0, 3, 2]] for box in boxes])  # Convert boxes to (ymin, xmin, ymax, xmax) format

    # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
     indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
     print(indices)

    # Draw bounding boxes and labels on the input image
     for i in indices:
        # i = i[0]
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        label = f"Yangin {class_id} ({score:.2f})"
        cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[1], box[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the output frame
     cv2.imshow("Object Detection", frame)

     conn = sqlite3.connect(r'C:\\Users\\oukaa\\Desktop\\fire\\personelbilgisi.db')
     cursor = conn.cursor()

   # Assuming the email address is stored in the "email" column of the "Personeller" table

     cursor.execute("SELECT email FROM Personeller")
     results = cursor.fetchall()  # Fetch all email addresses

     cursor.close() 
     conn.close()

# Iterate over the email addresses and send emails
     for result in results:
      email_address = result[0]
      print(email_address)
      if email_address:
      # Set up email parameters
         subject = "Fire Alert!"
         body = "A fire has been detected."
         sender_email = 'yanginuyari@gmail.com'
         receiver_email = email_address
         password = 'rbslqtwclmimwyza'  # Enter your email password here

      # Create email message
         message = EmailMessage()
         message["Subject"] = subject
         message["From"] = sender_email
         message["To"] = receiver_email
         message.set_content(body)

      # Send email
         context = ssl.create_default_context()
         with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.send_message(message)
            print("Email sent!")
      else:
         print("No email address found in the database.")

    # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")

     if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


