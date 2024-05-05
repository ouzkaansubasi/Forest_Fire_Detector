import cv2
import numpy as np
import onnxruntime
from yolov8.utils import xywh2xyxy
import time
from email.message import EmailMessage
import ssl
import smtplib
import sqlite3 

model_path = r"C:\\Users\\ouzkaan\\Desktop\\yangin\\best.onnx"
session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

input_details = session.get_inputs()
output_details = session.get_outputs()
input_name = input_details[0].name
input_shape = input_details[0].name
output_names = [output_detail.name for output_detail in output_details]

# Min doğruluk oranı
conf_threshold = 0.4
iou_threshold = 0

cap = cv2.VideoCapture(0)

while True:
   
    ret, frame= cap.read()
    img=frame
    start = time.perf_counter()

    img_height, img_width, _ = frame.shape
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (800, 800))
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    outputs = session.run(output_names, {input_name: input_tensor})
    predictions = np.squeeze(outputs[0]).T

    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    
    if len(scores) == 0:
        
        #cv2.imshow("Yangin tespit",frame)
        print("Herhangi bir yangin tespit edilemedi!")
        continue
    
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Kutucuk
    boxes = predictions[:, :4]
    boxes = np.divide(boxes, np.array([800,800,800,800], dtype=np.float32))
    boxes = xywh2xyxy(boxes)
    boxes *= np.array([img_width, img_height, img_width, img_height], dtype=np.float32)
    boxes = boxes.astype(np.int32)
    boxes = np.array([box[[1, 0, 3, 2]] for box in boxes])  

   
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    print(indices)

    for i in indices:
        # i = i[0]
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        label = f"Yangin {class_id} ({score:.2f})"
        cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
        cv2.putText(frame, label, (box[1], box[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Object Detection", frame)

    conn = sqlite3.connect(r'C:\\Users\\ouzkaan\\Desktop\\fire\\personelbilgisi.db')
    cursor = conn.cursor()


    cursor.execute("SELECT email FROM Personeller")
    results = cursor.fetchall()  

    cursor.close() 
    conn.close()

# mail gönderme
    for result in results:
      email_address = result[0]
      print(email_address)
      if email_address:
    
         subject = "Yangin Uyarisi!"
         body = "Yangin tespit edildi!!!!!"
         sender_email = 'yanginuyari@gmail.com'
         receiver_email = email_address
         password = 'rbslqtwclmimwyza'  

   
         message = EmailMessage()
         message["Subject"] = subject
         message["From"] = sender_email
         message["To"] = receiver_email
         message.set_content(body)

      
         context = ssl.create_default_context()
         with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.send_message(message)
            print("Email gönderildi!")
      else:
         print("Veritabaninda mail adresi bulunamadi")

    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


