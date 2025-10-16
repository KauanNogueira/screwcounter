import cv2
import os
import numpy as np

PARAM_BLUR = (7, 7)
PARAM_ADAPTIVE_BLOCK = 11
PARAM_ADAPTIVE_C = 4


folder_path = 'dataset/train'


def image_processor(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    soft_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(soft_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, PARAM_ADAPTIVE_BLOCK, PARAM_ADAPTIVE_C)
    kernel = np.ones((3, 3), np.uint8)
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    (contours, _) = cv2.findContours(clean_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return clean_binary, contours




# everything written here above is just to organize better the image distribution

# dealing with processing the images:


def main():

    # simple error handling:

    try:
        nomes_arquivos = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        if not nomes_arquivos:
            print("Nenhuma imagem foi encontrada")

    except FileNotFoundError:
        print(f'Erro: O diretório {folder_path} não foi encontrado.')

    image_name = nomes_arquivos[3] #change here the index of the images !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    full_path = os.path.join(folder_path, image_name)
    original_image = cv2.imread(full_path)


    binary_image, contours = image_processor(original_image)

    countours_image = original_image.copy()
    cv2.drawContours(countours_image, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Imagem original:', original_image)
    # cv2.imshow('Imagem em tons de cinza:', gray_image)
    # cv2.imshow('Imagem com desfoque:', soft_blur)
    cv2.imshow('Imagem Binarizadas:', binary_image)
    cv2.imshow('Imagem com Contornos:', countours_image)

    print("Aperte qualquer tecla para fechar as janelas")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()