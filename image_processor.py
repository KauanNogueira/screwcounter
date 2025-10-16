import cv2
import os

folder_path = 'dataset/train'
image_name_list = []
for file_name in range(1, 16):
    if file_name >= 10:
        image_name_list.append(f'{file_name}.jpg')
    else:
        image_name_list.append(f'0{file_name}.jpg')

image_name = image_name_list[14]
full_path = os.path.join(folder_path, image_name)

def mk_images(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    soft_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    (T, binary_image) = cv2.threshold(soft_blur, 160, 255, cv2.THRESH_BINARY)
    (contours, hierarchy) = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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
        # I need to correct this ASAP!! Making 2 times the same thing is not efficient
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        soft_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
        binary_image = cv2.adaptiveThreshold(soft_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                         cv2.THRESH_BINARY_INV, 11, 4)
        (contours, hierarchy) = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

else:
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    soft_blur = cv2.GaussianBlur(gray_image, (7, 7), 0)
    binary_image = cv2.adaptiveThreshold(soft_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv2.THRESH_BINARY_INV, 11, 4)
    (contours, hierarchy) = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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