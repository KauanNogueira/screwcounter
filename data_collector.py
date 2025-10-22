import cv2
import os
import numpy as np
import csv
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# --- CONFIGURAÇÕES ---
DATASET_FOLDER = 'dataset/train'
OUTPUT_CSV_FILE = 'treino_classificado.csv'
MIN_CONTOUR_AREA = 100

def apply_processing(image, method, blur_ksize, thresh_val, block_size, C, morph_ksize):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 0 and blur_ksize % 2 == 0: blur_ksize += 1
    blurred = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0) if blur_ksize > 0 else gray_image
    binary = None
    if method == 0:
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        if block_size < 3: block_size = 3
        if block_size % 2 == 0: block_size += 1
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C)
    if morph_ksize > 0:
        if morph_ksize % 2 == 0: morph_ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        clean_binary = binary
    return clean_binary

def separate_objects_with_watershed(binary_image, min_distance_val):
    if min_distance_val < 1: return np.zeros(binary_image.shape, dtype=np.int32)
    distance_map = ndimage.distance_transform_edt(binary_image)
    local_maxima = peak_local_max(distance_map, min_distance=min_distance_val, labels=binary_image)
    markers = np.zeros(distance_map.shape, dtype=bool)
    markers[local_maxima[:, 0], local_maxima[:, 1]] = True
    markers = ndimage.label(markers)[0]
    labels = watershed(-distance_map, markers, mask=binary_image)
    return labels

def nothing(x):
    pass

def main():
    try:
        file_names = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(('.jpg', '.png'))]
    except FileNotFoundError:
        print(f"ERRO: O diretório '{DATASET_FOLDER}' não foi encontrado.")
        return
    with open(OUTPUT_CSV_FILE, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['area', 'aspect_ratio', 'extent', 'solidity', 'is_screw']
        csv_writer.writerow(header)
        cv2.namedWindow('Parameter Tuner')
        cv2.createTrackbar('Threshold Method', 'Parameter Tuner', 1, 1, nothing)
        cv2.createTrackbar('Blur Kernel', 'Parameter Tuner', 3, 31, nothing)
        cv2.createTrackbar('Global Thresh Val', 'Parameter Tuner', 127, 255, nothing)
        cv2.createTrackbar('Adaptive Block Size', 'Parameter Tuner', 81, 255, nothing)
        cv2.createTrackbar('Adaptive C', 'Parameter Tuner', 8, 50, nothing)
        cv2.createTrackbar('Morph Kernel (Close)', 'Parameter Tuner', 5, 51, nothing)
        cv2.createTrackbar('Watershed Min Dist', 'Parameter Tuner', 32, 100, nothing)
        color_palette = [np.random.randint(50, 255, size=(3,)).tolist() for _ in range(200)]
        for image_name in file_names:
            full_path = os.path.join(DATASET_FOLDER, image_name)
            original_image = cv2.imread(full_path)
            if original_image is None: continue
            print(f"\n--- AJUSTANDO IMAGEM: {image_name} ---")
            print("1. Ajuste os sliders para obter uma boa segmentação.")
            print("2. Pressione 'ENTER' para começar a classificar os objetos.")
            print("3. Pressione 'q' para pular esta imagem.")
            while True:
                method = cv2.getTrackbarPos('Threshold Method', 'Parameter Tuner')
                blur = cv2.getTrackbarPos('Blur Kernel', 'Parameter Tuner')
                glob_thresh = cv2.getTrackbarPos('Global Thresh Val', 'Parameter Tuner')
                block_size = cv2.getTrackbarPos('Adaptive Block Size', 'Parameter Tuner')
                c_val = cv2.getTrackbarPos('Adaptive C', 'Parameter Tuner')
                morph = cv2.getTrackbarPos('Morph Kernel (Close)', 'Parameter Tuner')
                watershed_dist = cv2.getTrackbarPos('Watershed Min Dist', 'Parameter Tuner')
                binary_img = apply_processing(original_image, method, blur, glob_thresh, block_size, c_val, morph)
                labels = separate_objects_with_watershed(binary_img, watershed_dist)
                viz_img = np.zeros_like(original_image)
                for label in np.unique(labels):
                    if label == 0: continue
                    mask = np.zeros(binary_img.shape, dtype="uint8")
                    mask[labels == label] = 255
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color = color_palette[label % len(color_palette)]
                    cv2.drawContours(viz_img, contours, -1, color, 2)
                cv2.imshow('Ajuste a Segmentacao e Pressione ENTER', viz_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('\r') or key == ord('\n'):
                    break
                if key == ord('q'):
                    labels = []
                    break
            if not np.any(labels):
                continue
            print("\n--- CLASSIFICANDO OBJETOS ---")
            print("Pressione '1' para PARAFUSO, '0' para NAO PARAFUSO, 's' para parar com esta imagem, 'q' para SAIR.")
            for label_id in np.unique(labels):
                if label_id == 0: continue
                mask = np.zeros(binary_img.shape, dtype="uint8")
                mask[labels == label_id] = 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA: continue
                temp_image = original_image.copy()
                cv2.drawContours(temp_image, [cnt], -1, (0, 0, 255), 3)
                cv2.imshow('Classifique:', temp_image)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'): break
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                if key == ord('1') or key == ord('0'):
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(h) / w if w != 0 else 0
                    rect_area = w * h
                    extent = float(area) / rect_area if rect_area != 0 else 0
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area != 0 else 0
                    is_screw = 1 if key == ord('1') else 0
                    row = [area, aspect_ratio, extent, solidity, is_screw]
                    csv_writer.writerow(row)
    cv2.destroyAllWindows()
    print(f"\nClassificação concluída! Dados salvos em '{OUTPUT_CSV_FILE}'.")

if __name__ == "__main__":
    main()