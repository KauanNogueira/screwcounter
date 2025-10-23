# This is the last part of the project, and it is responsible for counting screws in a given image using the screw_classifier_model.pkl trained model. 
# Wich was trained with the data collected in data_collector.py and the model trained in train.py. 
# This script loads the image and the model

import cv2
import numpy as np
import joblib
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

BLUR_KSIZE = 3
ADAPTIVE_BLOCK_SIZE = 81
ADAPTIVE_C = 8
MORPH_KSIZE = 5
WATERSHED_MIN_DIST = 32

# config
MODEL_FILE = 'screw_classifier_model.pkl'
IMAGE_TO_TEST = 'dataset/test/20.jpg' # change image here!!!!!!!!
MIN_CONTOUR_AREA = 100

def apply_processing(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (BLUR_KSIZE, BLUR_KSIZE), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KSIZE, MORPH_KSIZE))
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return clean_binary

def separate_objects_with_watershed(binary_image):
    distance_map = ndimage.distance_transform_edt(binary_image)
    local_maxima = peak_local_max(distance_map, min_distance=WATERSHED_MIN_DIST, labels=binary_image)
    markers = np.zeros(distance_map.shape, dtype=bool)
    markers[local_maxima[:, 0], local_maxima[:, 1]] = True
    markers = ndimage.label(markers)[0]
    labels = watershed(-distance_map, markers, mask=binary_image)
    return labels

def main():
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"ERRO: Modelo '{MODEL_FILE}' não encontrado. Execute '2_train_model.py' primeiro.")
        return

    original_image = cv2.imread(IMAGE_TO_TEST)
    if original_image is None:
        print(f"ERRO: Imagem '{IMAGE_TO_TEST}' não encontrada.")
        return

    result_image = original_image.copy()
    screw_count = 0

    binary_image = apply_processing(original_image)
    labels = separate_objects_with_watershed(binary_image)

    for label_id in np.unique(labels):
        if label_id == 0: continue
        mask = np.zeros(binary_image.shape, dtype="uint8")
        mask[labels == label_id] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA: continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(h) / w if w != 0 else 0
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        features = np.array([area, aspect_ratio, extent, solidity]).reshape(1, -1)
        prediction = model.predict(features)
        
        if prediction[0] == 1:
            screw_count += 1
            cv2.drawContours(result_image, [cnt], -1, (0, 255, 0), 2)
        else:
            cv2.drawContours(result_image, [cnt], -1, (0, 0, 255), 2)


    text = f"Parafusos encontrados: {(screw_count/11)*6}"
    # I'd like to highlight this one. The model works by couting countours in an image, and there are a lot of screws that can have more than 2 countours. This 'magic number' is in fact just a weighted average with the probabilities of a screw having 1, 2 or 3 countours.

    print(text)
    cv2.putText(result_image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Resultado Final', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()