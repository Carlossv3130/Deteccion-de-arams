import cv2
import time
import numpy as np

# Rutas ABSOLUTAS a tus archivos .weights, .cfg y .names
weights_path = "C:/Users/zahar/OneDrive/Escritorio/deteccion_armas/yolov3_4000.weights"
config_path = "C:/Users/zahar/OneDrive/Escritorio/deteccion_armas/yolov3.cfg"
classes_path = "C:/Users/zahar/OneDrive/Escritorio/deteccion_armas/yolo.names"  # Cambia si es un archivo .names

# Cargar la red YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Obtener nombres de capas de salida
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except:
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Cargar clases
try:
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de clases en la ruta: {classes_path}")
    exit()

# Buscar una cámara disponible
camera_index = -1
for i in range(5):
    test_cap = cv2.VideoCapture(i)
    if test_cap.read()[0]:
        camera_index = i
        test_cap.release()
        break

if camera_index == -1:
    print("❌ No se detectó ninguna cámara disponible.")
    exit()
else:
    print(f"✅ Usando cámara en índice {camera_index}")

cap = cv2.VideoCapture(camera_index)

while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        print("⚠️ No se pudo leer el frame de la cámara.")
        break

    height, width, channels = frame.shape

    # Preprocesar la imagen
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        try:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            box = boxes[i]
            x, y, w, h = round(box[0]), round(box[1]), round(box[2]), round(box[3])
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except IndexError:
            pass

    cv2.imshow("Detección en Tiempo Real", frame)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}", end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
