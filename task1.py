import random
from matplotlib import pyplot as plt
from typing import List, Tuple

Vertex = Tuple[float, float]
Edge = Tuple[Vertex, Vertex]
Rectangle = Tuple[float, float, float, float]

class RectangleCounter:


    def __init__(self, points: List[Vertex]):
        self.sorted_x = sorted(list(set(p[0] for p in points)))
        self.sorted_y = sorted(list(set(p[1] for p in points)))
        self.Q = self._compute_Q(points)

    def _compute_Q(self, points: List[Vertex]) -> List[List[int]]:
        Q = [[0] * (len(self.sorted_y) + 1) for _ in range(len(self.sorted_x) + 1)]
        for point in points:
            x_idx = self._find_last_less_or_equal(self.sorted_x, point[0]) + 1
            y_idx = self._find_last_less_or_equal(self.sorted_y, point[1]) + 1
            Q[x_idx][y_idx] += 1

        for i in range(1, len(self.sorted_x) + 1):
            for j in range(1, len(self.sorted_y) + 1):
                Q[i][j] += Q[i - 1][j] + Q[i][j - 1] - Q[i - 1][j - 1]

        return Q

    def count_points_inside(self, rectangles: List[Rectangle]) -> List[int]:
        counts = []
        for rect in rectangles:
            p1 = (rect[2], rect[3])  
            p2 = (rect[0], rect[3])  
            p3 = (rect[0], rect[1]) 
            p4 = (rect[2], rect[1])  

            i1 = max(0, self._find_last_less_or_equal(self.sorted_x, p1[0])) + 1
            j1 = max(0, self._find_last_less_or_equal(self.sorted_y, p1[1])) + 1
            
            temp = self._find_last_less(self.sorted_x, p3[0])
            i3 = temp + 1 if temp is not None else 0

            temp = self._find_last_less(self.sorted_y, p3[1])
            j3 = temp + 1 if temp is not None else 0

            temp = self._find_last_less(self.sorted_x, p4[0])
            i4 = temp + 1 if temp is not None else 0

            temp = self._find_last_less(self.sorted_y, p2[1])
            j2 = temp + 1 if temp is not None else 0

            temp = self._find_last_less(self.sorted_x, p2[0])
            i2 = temp + 1 if temp is not None else 0

            temp = self._find_last_less(self.sorted_y, p4[1])
            j4 = temp + 1 if temp is not None else 0

            count = self.Q[i1][j1] - self.Q[i2][j2] - self.Q[i4][j4] + self.Q[i3][j3]
            counts.append(count)

        return counts

    def _find_last_less_or_equal(self, arr, target):
        left = 0
        right = len(arr) - 1
        result = None

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result

    def _find_last_less(self, arr, target):
        left = 0
        right = len(arr) - 1
        result = None

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] < target:
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result


def generate_points(num_points: int, x_range: Vertex, y_range: Vertex) -> List[Vertex]:
    """Генерирует случайные точки в заданных диапазонах x и y.

    Аргументы:
        num_points (int): Количество точек для генерации.
        x_range (Vertex): Диапазон значений x (мин, макс).
        y_range (Vertex): Диапазон значений y (мин, макс).

    Возвращает:
        List[Vertex]: Список сгенерированных точек.
    """
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]


def generate_rectangles(num_rectangles: int, x_range: Vertex, y_range: Vertex) -> List[Rectangle]:
    """Генерирует случайные прямоугольники в заданных диапазонах x и y.

    Аргументы:
        num_rectangles (int): Количество прямоугольников для генерации.
        x_range (Vertex): Диапазон значений x (мин, макс).
        y_range (Vertex): Диапазон значений y (мин, макс).

    Возвращает:
        List[Rectangle]: Список сгенерированных прямоугольников.
    """
    rectangles = []
    for _ in range(num_rectangles):
        x1 = random.uniform(*x_range)
        x2 = random.uniform(x1, x_range[1])
        y1 = random.uniform(*y_range)
        y2 = random.uniform(y1, y_range[1])
        rectangles.append((x1, y1, x2, y2))
    return rectangles


def plot_points_and_rectangles(points: List[Vertex], rectangles: List[Rectangle]) -> None:
    """Строит график точек и прямоугольников.

    Аргументы:
        points (List[Vertex]): Список точек.
        rectangles (List[Rectangle]): Список прямоугольников.

    Возвращает:
        None
    """
    fig, ax = plt.subplots()

    # График точек
    for x, y in points:
        ax.plot(x, y, 'bo')  # 'bo' - синий цвет и круглая метка

    # График прямоугольников и их подписи
    for index, rect in enumerate(rectangles, start=1):  # Начать с 1
        x1, y1, x2, y2 = rect
        rect_plot = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect_plot)
        # Подписи прямоугольников
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(index), color='red', fontsize=30, ha='center', va='center')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Прямоугольники и точки')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def count_points_inside_rectangles(points: List[Vertex], rectangles: List[Rectangle]) -> None:
    """Считает количество точек внутри каждого прямоугольника и выводит результаты.

    Аргументы:
        points (List[Vertex]): Список точек.
        rectangles (List[Rectangle]): Список прямоугольников.

    Возвращает:
        None
    """
    counts = RectangleCounter(points).count_points_inside(rectangles)
    print("Количество точек внутри каждого прямоугольника:")
    for i, count in enumerate(counts):
        print(f"Прямоугольник {i + 1}: {count}")




# Генерация случайных точек и прямоугольников
num_points = 1000
num_rectangles = 150
x_range = (0, 1000)
y_range = (0, 1000)

points = [(1, 1), (2, 2), (3, 3), (1, 4), (4, 1), (4, 3), (4, 2), (3.5, 3), (2, 4)]
rectangles = [(0, 0, 2, 2), (1, 1, 3, 3), (1.5, 1.5, 4, 4)]
count_points_inside_rectangles(points, rectangles)

# Построение графика
plot_points_and_rectangles(points, rectangles)

points = generate_points(num_points, x_range, y_range)
rectangles = generate_rectangles(num_rectangles, x_range, y_range)

print("\nТест с 1000 точками и 150 прямоугольниками:")
count_points_inside_rectangles(points, rectangles)
