import cv2
import os
import numpy as np
import csv

PASTA_DATASET = 'dataset/train'
ARQUIVO_TREINO = 'treino_classificado.csv'
MIN_AREA_CONTORNO = 100

PARAM_ADAPTIVE_BLOCK = 11
PARAM_ADAPTIVE_C = 4

def image_processor(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    soft_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(soft_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, PARAM_ADAPTIVE_BLOCK, PARAM_ADAPTIVE_C)
    kernel = np.ones((3, 3), np.uint8)
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    (contours, _) = cv2.findContours(clean_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    try:
        file_names = [f for f in os.listdir(PASTA_DATASET) if f.endswith(('.jpg', '.png'))]
    except FileNotFoundError:
        print(f"ERRO: O diretório '{PASTA_DATASET}' não foi encontrado.")
        return

    with open(ARQUIVO_TREINO, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['area', 'aspect_ratio', 'extent', 'solidity', 'is_screw']
        csv_writer.writerow(header)

        print("--- INICIANDO CLASSIFICAÇÃO VISUAL ---")
        print("Pressione '1' para PARAFUSO.")
        print("Pressione '0' para NÃO PARAFUSO.")
        print("Pressione 'q' para SAIR e SALVAR.")

        for image_name in file_names:
            full_path = os.path.join(PASTA_DATASET, image_name)
            original_image = cv2.imread(full_path)
            if original_image is None:
                continue

            contours = image_processor(original_image)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_AREA_CONTORNO:
                    continue

                imagem_temp = original_image.copy()
                cv2.drawContours(imagem_temp, [cnt], -1, (0, 0, 255), 3)
                cv2.imshow('Classifique este contorno:', imagem_temp)

                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    print("Saindo e salvando...")
                    cv2.destroyAllWindows()
                    print(f"Classificação parcial salva em '{ARQUIVO_TREINO}'.")
                    return

                if key == ord('1') or key == ord('0'):
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h if h != 0 else 0
                    
                    rect_area = w * h
                    extent = float(area) / rect_area if rect_area != 0 else 0
                    
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area != 0 else 0
                    
                    classe = 1 if key == ord('1') else 0
                    linha = [area, aspect_ratio, extent, solidity, classe]
                    csv_writer.writerow(linha)

    cv2.destroyAllWindows()
    print(f"\nClassificação concluída! Dados salvos em '{ARQUIVO_TREINO}'.")

if __name__ == "__main__":
    main()