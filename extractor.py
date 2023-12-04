from threading import Thread

import cv2
import numpy as np
from keras.models import load_model

# Модель для распознавания отметок
recogniser = load_model("model/recognition.h5", compile=False)
class_names = open("model/labels.txt", "r").readlines()


def find_table_corners(mat: cv2.typing.MatLike) -> np.single:
    # Ищем самой большой контур и берем его примерное очертание
    biggest_contour = cv2.convexHull(
        max(
            cv2.findContours(mat, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0],
            key=cv2.contourArea
        )
    )
    # Приводим контур к прямоугольнику
    box = cv2.approxPolyDP(biggest_contour, 0.1 * cv2.arcLength(biggest_contour, True), True)
    box = np.squeeze(box).astype(np.float32)
    # Находим центр контура и угол его наклона
    moments = cv2.moments(biggest_contour)
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    center = np.array([cx, cy])
    cbox = box - center
    ang = np.arctan2(cbox[:, 1], cbox[:, 0]) * 180 / np.pi
    # Получаем края таблицы
    box = box[ang.argsort()]
    corners = np.float32([box[0], box[1], box[2], box[3]])

    return corners


def warp_perspective(base: cv2.typing.MatLike, mat: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # Края таблицы
    table_corners = find_table_corners(base)
    # Высота и ширина таблицы
    table_width = np.linalg.norm(table_corners[0] - table_corners[1])
    table_height = np.linalg.norm(table_corners[0] - table_corners[3])
    # Соотношение сторон таблицы
    aspect_ratio = table_height / table_width
    # 90% от исходной ширины и пропорциональная ей высота
    width = int(mat.shape[1] * 0.9)
    height = int(width * aspect_ratio)
    # Края для исправленной в перспективе таблицы
    fixed_table_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # Искревление в перспективе
    kernel = cv2.getPerspectiveTransform(table_corners, fixed_table_corners)

    return cv2.warpPerspective(mat, kernel, (width, height))


def extract_grid(mat: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    # Ширина, высота и матрица сдвига
    height, width = mat.shape[:2]
    translation_matrix = np.float32([[1, 0, -8], [0, 1, -8]])
    # Получаем горизонтальные линии
    x_lines = cv2.erode(mat, np.ones((1, 20), np.uint8), iterations=1)
    x_lines = cv2.erode(x_lines, np.ones((1, 5), np.uint8), iterations=1)
    x_lines = cv2.ximgproc.thinning(x_lines)
    # Продлеваем горизонтальные линии
    for i in range(10):
        x_lines = cv2.morphologyEx(x_lines, cv2.MORPH_CLOSE, np.ones((10, 1), np.uint8))
        x_lines = cv2.dilate(x_lines, np.ones((1, 50), np.uint8))
        x_lines = cv2.ximgproc.thinning(x_lines)
    # Достраиваем горзонтальные линии до границ
    x_contours = cv2.findContours(x_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in x_contours:
        # Левая и правая вершины линий
        left = tuple(contour[contour[:, :, 0].argmin()][0])
        right = tuple(contour[contour[:, :, 0].argmax()][0])
        # Проверяем, касается ли линия одной из границ
        if right[0] == width - 1:
            cv2.line(x_lines, left, (0, left[1]), (255, 255, 255), thickness=1)
        elif left[0] == 0:
            cv2.line(x_lines, right, (width, right[1]), (255, 255, 255), thickness=1)
    # Сдвиг изображения
    x_lines = cv2.warpAffine(x_lines, translation_matrix, (width, height))
    x_contours = cv2.findContours(x_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in x_contours:
        # Левая и правая вершины линий
        left = tuple(contour[contour[:, :, 0].argmin()][0])
        right = tuple(contour[contour[:, :, 0].argmax()][0])
        # Проверяем, касается ли линия левой границы
        if left[0] == 0:
            cv2.line(x_lines, right, (width, right[1]), (255, 255, 255), thickness=1)
    # Достраиваем вертикальные линии
    y_lines = cv2.erode(mat, np.ones((20, 1), np.uint8), iterations=1)
    y_lines = cv2.erode(y_lines, np.ones((5, 1), np.uint8), iterations=1)
    y_lines = cv2.ximgproc.thinning(y_lines)
    # Продлеваем вертикальные линии
    for i in range(10):
        y_lines = cv2.morphologyEx(y_lines, cv2.MORPH_CLOSE, np.ones((1, 10), np.uint8))
        y_lines = cv2.dilate(y_lines, np.ones((50, 1), np.uint8))
        y_lines = cv2.ximgproc.thinning(y_lines)
    # Достраиваем вертикальные линии до границ
    y_contours = cv2.findContours(y_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in y_contours:
        # Верхняя и нижняя вершины линий
        top = tuple(contour[contour[:, :, 1].argmin()][0])
        bottom = tuple(contour[contour[:, :, 1].argmax()][0])
        # Проверяем, касается ли линия одной из границ
        if top[1] == 0:
            cv2.line(y_lines, bottom, (bottom[0], height), (255, 255, 255), 1)
        elif top[1] == height - 1:
            cv2.line(y_lines, top, (top[0], 0), (255, 255, 255), 1)
    # Сдвиг изображения
    y_lines = cv2.warpAffine(y_lines, translation_matrix, (width, height))
    y_contours = cv2.findContours(y_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in y_contours:
        # Верхняя и нижняя вершины линий
        top = tuple(contour[contour[:, :, 1].argmin()][0])
        bottom = tuple(contour[contour[:, :, 1].argmax()][0])
        # Проверяем, касается ли линия верхней границы
        if top[1] == 0:
            cv2.line(y_lines, bottom, (bottom[0], height), (255, 255, 255), 1)

    return x_lines, y_lines


def extract_table(xy_mat: cv2.typing.MatLike, x_mat: cv2.typing.MatLike) -> list[list[cv2.typing.MatLike]]:
    # Извлекаем все горизонтальные контуры
    x_contours = sorted(
        cv2.findContours(x_mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0],
        key=lambda ctr: cv2.boundingRect(ctr)[1]
    )[1:]
    # Извлекаем все контуры, описывающие отдельные клетки
    contours = cv2.findContours(
        xy_mat,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )[0]
    # Генерируем таблицу
    table = [[] for _ in range(len(x_contours))]
    for contour in contours:
        # Находим координату центра контура
        moments = cv2.moments(contour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        # Ищем к какой строке относится контур
        for i, x_contour in enumerate(x_contours):
            if cv2.pointPolygonTest(x_contour, center, False) >= 0:
                table[i].append(contour)
                break
    # Соритруем контуры по горизонтальной оси
    for i, contours in enumerate(table):
        table[i] = sorted(
            contours,
            key=lambda ctr: cv2.boundingRect(ctr)[0]
        )

    return table


def process_cell(cell: cv2.typing.MatLike, table: list[list[int | None]], i: int, j: int) -> None:
    # noinspection PyUnresolvedReferences
    upscaler = cv2.dnn_superres.DnnSuperResImpl_create()
    upscaler.readModel("model/upscaling.pb")
    upscaler.setModel("fsrcnn", 3)
    # Увеличиваем разрешение изображения клетки
    cell = upscaler.upsample(cell)
    cell = upscaler.upsample(cell)
    # Обечцвечиваем, очищаем от мусора
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    cell = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 3)
    # Отсеиваем пустые клетки
    if cv2.countNonZero(cell) / (cell.shape[0] * cell.shape[1]) < 0.25:
        return
    # Уменьшаем клетку до удобного для нейросети размера
    aspect_ratio = cell.shape[0] / cell.shape[1]
    if aspect_ratio < 1:
        cell = cv2.resize(cell, (int(aspect_ratio * 224), 224), interpolation=cv2.INTER_AREA)
    else:
        cell = cv2.resize(cell, (224, int(224 / aspect_ratio)), interpolation=cv2.INTER_AREA)
    # Инверируем изображение клетки, чтобы шрифт был черным
    cell = cv2.bitwise_not(cell)
    # Переводим в цветное изоюражение
    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
    # Заполняем все пустое пространство соответсвующим фону цветом
    cell[np.where((cell > [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    # Копируем фон
    back = np.full((224, 224, 3), [255, 255, 255], dtype=np.uint8)
    # Помещяем изображение клетки в центр фона
    h, w = back.shape[:2]
    h1, w1 = cell.shape[:2]
    cy, cx = (h - h1) // 2, (w - w1) // 2
    back[cy:cy + h1, cx:cx + w1] = cell
    # Готовое для распознавания изображение клетки
    cell = back
    # Преобразуем
    # noinspection PyArgumentList
    cell = np.asarray(cell, dtype=np.float32).reshape(1, 224, 224, 3)
    # Нормализуем
    cell = (cell / 127.5) - 1
    # Угадываем
    prediction = recogniser.predict(cell)
    # Находим индекс класса с самым высоким шансом
    index = np.argmax(prediction)
    # Получаем угаданную цифру
    digit = int(class_names[index][0])
    # Заносим полученную цифру в таблицу
    table[i][j] = 10 if digit == 1 else digit

    return


def extract(mat: cv2.typing.MatLike) -> list[list[int | None]]:
    # Исходное изображение
    original_mat = mat.copy()
    # Обечцвечиваем, очищаем от мусора и расшираем пиксели
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    mat = cv2.Canny(mat, 50, 100)
    # noinspection PyTypeChecker
    mat = cv2.dilate(mat, None, iterations=1)
    # Исправляем перспективу
    mat = warp_perspective(mat, original_mat)
    original_mat = mat.copy()
    # Обечцвечиваем, очищаем от мусора
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    mat = cv2.adaptiveThreshold(mat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 11)
    # Находим сетку таблицы
    x_mat, y_mat = extract_grid(mat)
    # Добавляем границу ко всему для последующего поиска клеток
    original_mat = cv2.copyMakeBorder(original_mat, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
    x_mat = cv2.copyMakeBorder(x_mat, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
    y_mat = cv2.copyMakeBorder(y_mat, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
    xy_mat = cv2.bitwise_or(x_mat, y_mat)
    xy_mat = cv2.copyMakeBorder(xy_mat, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
    # Получаем таблицу
    table = extract_table(xy_mat, x_mat)[1:]
    # Будущая готовая таблица (40 строк, 30 колонок)
    extracted_table = [[None for _ in range(30)] for _ in range(40)]
    # Проходимся по каждой ячейке
    index = None
    threads = []
    for i, contours in enumerate(table):
        for j, contour in enumerate(contours):
            # Пропускаем первую колонку с номерами учеников и излишние строки и колонки в конце
            if j == 0 or j >= 25 or (index is not None and index >= 30):
                continue
            # Прямоугольник, описывающий клетку
            br = cv2.boundingRect(contour)
            # Получаем клетку с исходного изображения
            cell = original_mat[br[1]:br[1] + br[3], br[0]:br[0] + br[2]]
            # Отсеиваем некорректные данные
            if cell.shape[1] > 200:
                continue
            # Проверяем, являтся ли клетка отметочной
            if index is None:
                if cell.shape[0] / cell.shape[1] < 1.25 and cell.shape[1] / cell.shape[0] < 1.25:
                    index = 0
                else:
                    break
            # Обрабатываем клетку и заносим результат
            thread = Thread(target=process_cell, args=(cell, extracted_table, index, j - 1,))
            thread.start()
            threads.append(thread)
        # Переходим на следующую строку
        if index is not None:
            index += 1
    # Ждем завершения каждого потока
    for thread in threads:
        thread.join()

    return extracted_table
