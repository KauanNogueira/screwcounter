import cv2
import os
import numpy as np
import csv

folder_path = 'dataset/train'
output_csv = 'features.csv'
PARAM_ADAPTIVE_BLOCK = 11
PARAM_ADAPTIVE_C = 4

# now that the function works properly, let's extract the contours and save them in a csv file:

def image_processor(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    soft_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(soft_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, PARAM_ADAPTIVE_BLOCK, PARAM_ADAPTIVE_C)
    kernel = np.ones((3, 3), np.uint8)
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    (contours, _) = cv2.findContours(clean_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def main():

    try:
        file_names = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        if not file_names:
            print("Nenhuma imagem foi encontrada")

    except FileNotFoundError:
        print(f'Erro: O diretório {folder_path} não foi encontrado.')

    # dealing with csv:

    with open(output_csv, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        header = ['image_name', 'area', 'aspect_ratio', 'extent', 'solidity']
        csv_writer.writerow(header)
        print('Iniciando características!')

        for image_name in file_names:
            full_path = os.path.join(folder_path, image_name)
            original_image = cv2.imread(full_path)
            if original_image is None:
                print(f'Erro ao carregar a imagem {image_name}. Pulando para a próxima.')
                continue

            contours = image_processor(original_image)
            for contour in contours:

                # proportions

                area = cv2.contourArea(contour)

                if area < 50:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h != 0 else 0
                
                # extent & area

                rect_area = w * h
                extent = float(area) / rect_area if rect_area != 0 else 0

                # solidity

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area != 0 else 0

                row = [image_name, area, aspect_ratio, extent, solidity]
                csv_writer.writerow(row)

            print(f'Imagem OK e processada: {image_name}, com {len(contours)} contornos.')
    print(f'Características salvas em {output_csv}')

if __name__ == "__main__":
    main()