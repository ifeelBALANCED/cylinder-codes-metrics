from PIL import Image
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage import img_as_ubyte
import numpy as np
import os
from multiprocessing import Pool
import time


def preprocess_image(im):
    # Конвертування зображення у відтінки сірого
    gray = im.convert("L")
    # Зміна розміру зображення за допомогою фільтра LANCZOS
    resized = gray.resize((128, 128), Image.LANCZOS)
    return np.array(resized)


def detect_features(im):
    # Знаходження координат особливостей за допомогою peak_local_max
    coordinates = peak_local_max(im, min_distance=20, num_peaks=10)
    coordinates = coordinates[:, ::-1]  # Переупорядкування координат (рядок, стовпець) на (x, y)
    return coordinates.tolist()


def describe_feature(im, coordinate):
    # Перевірка, що координати знаходяться в межах зображення
    x = np.clip(coordinate[0], 2, im.shape[0] - 3)
    y = np.clip(coordinate[1], 2, im.shape[1] - 3)
    # Отримання області 5x5 навколо особливості
    patch = im[x - 2: x + 3, y - 2: y + 3]
    return patch


def compute_cylinder_code(patch):
    # Конвертування області в 8-бітовий формат без знаку
    patch = img_as_ubyte(patch)
    # Обчислення горизонтальних та вертикальних градієнтів
    sobelx = sobel(patch, axis=1)
    sobely = sobel(patch, axis=0)
    # Обчислення модуля та напрямку градієнту
    magnitude = np.hypot(sobelx, sobely)
    direction = np.arctan2(sobely, sobelx)
    # Порогова обробка модуля та напрямку для створення бінарного циліндричного коду
    magnitude_code = (magnitude > np.mean(magnitude)).astype(int)
    direction_code = (direction > np.mean(direction)).astype(int)
    # Об'єднання кодів для отримання остаточного циліндричного коду
    cylinder_code = np.concatenate((magnitude_code, direction_code))
    return cylinder_code


def compute_cylinder_distance(code1, code2):
    # Обчислення відстані Хеммінга між двома циліндричними кодами
    distance = np.mean(code1 != code2)
    return distance


if __name__ == "__main__":
    dir_path = "samples/"
    image_files = [f for f in os.listdir(dir_path) if f.endswith(".bmp")]


    def compare_fingerprints(pair):
        i, j = pair
        if i == j:
            return image_files[i], image_files[j], 1.0  # Повертаємо 1, якщо порівнюємо зображення з собою

        image1 = Image.open(os.path.join(dir_path, image_files[i])).convert("L")
        image2 = Image.open(os.path.join(dir_path, image_files[j])).convert("L")
        processed_image1 = preprocess_image(image1)
        processed_image2 = preprocess_image(image2)
        coordinates1 = detect_features(processed_image1)
        coordinates2 = detect_features(processed_image2)
        if not coordinates1 or not coordinates2:
            return image_files[i], image_files[j], None
        cylinder_codes1 = [compute_cylinder_code(describe_feature(processed_image1, c)) for c in coordinates1]
        cylinder_codes2 = [compute_cylinder_code(describe_feature(processed_image2, c)) for c in coordinates2]
        if not cylinder_codes1 or not cylinder_codes2:
            return image_files[i], image_files[j], None
        distances = []
        for code1 in cylinder_codes1:
            for code2 in cylinder_codes2:
                distance = compute_cylinder_distance(code1, code2)
                distances.append(distance)
        similarity = np.mean(distances)
        return image_files[i], image_files[j], similarity


    pairs = [(i, j) for i in range(len(image_files)) for j in range(len(image_files))]

    start_time = time.time()

    results = []
    total_pairs = len(pairs)
    processed_pairs = 0

    with Pool() as pool:
        for result in pool.imap_unordered(compare_fingerprints, pairs):
            results.append(result)
            processed_pairs += 1
            progress = processed_pairs / total_pairs * 100
            print("Виконано: {:.2f}%".format(progress), end="\r")

    execution_time = time.time() - start_time
    print("Загальний час виконання: {:.2f} секунд".format(execution_time))

    # Запис метрик циліндрів у файл "cylinder_codes.txt"
    with open("cylinder_codes.txt", "w") as file:
        for image1, image2, similarity in results:
            if similarity is not None:
                file.write("Зображення: {} - {}, Подібність: {:.4f}\n".format(image1, image2, similarity))
            else:
                file.write("Не знайдено дійсних особливостей у відбитку пальця: {} - {}\n".format(image1, image2))

    print("Метрики циліндрів збережено у файлі cylinder_codes.txt")
