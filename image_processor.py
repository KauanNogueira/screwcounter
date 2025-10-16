import cv2
import os

folder_path = 'dataset/train'
image_name_list = []
for file_name in range(1, 16):
    if file_name >= 10:
        image_name_list.append(f'{file_name}.jpg')
    else:
        image_name_list.append(f'0{file_name}.jpg')

image_name = image_name_list[13]
full_path = os.path.join(folder_path, image_name)

# everything written here above is just to organize better the image distribution

# dealing with processing the images:

original_image = cv2.imread(full_path)

# simple error handling:

if original_image is None:
    print(f'não foi possível encontrar a imgem no caminho {full_path}\n')
    print(f'tentando substituir o .jpg por .png:\n')
    image_name = image_name.replace('.jpg', '.png')
    full_path = os.path.join(folder_path, image_name)
    original_image = cv2.imread(full_path)
    if original_image is None:
        raise FileNotFoundError(f'Imagem não encontrada no caminho {full_path}, verifique se o arquivo existe.\n')
    else:
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        soft_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

else:
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    soft_blur = cv2.GaussianBlur(gray_image, (7, 7), 0)
    (T, binary_image) = cv2.threshold(soft_blur, 160, 255, cv2.THRESH_BINARY)


cv2.imshow('Imagem original:', original_image)
cv2.imshow('Imagem em tons de cinza:', gray_image)
cv2.imshow('Imagem com desfoque:', soft_blur)
cv2.imshow('Imagem Binarizadas:', binary_image)
print("Aperte qualquer tecla para fechar as janelas")
cv2.waitKey(0)
cv2.destroyAllWindows()