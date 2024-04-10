from typing import List, Tuple, Optional

import matplotlib.pyplot as plt

Point = Tuple[float, float]
Edge = Tuple[Point, Point]

# Создание горизонтальных полос на основе y-координат вершин.
def create_horizontal_strips(vertices: List[Point]) -> List[float]:
    """Создает горизонтальные полосы на основе y-координат вершин.

    Args:
        vertices (List[Point]): Список вершин графа.

    Returns:
        List[float]: Отсортированный список y-координат вершин.
    """
    return sorted({Point[1] for Point in vertices})


# Поиск полосы, содержащей заданную точку.
def find_strip_for_point(strips: List[float], point: Point) -> int:
    """Находит полосу, содержащую заданную точку.

    Args:
        strips (List[float]): Список горизонтальных полос.
        point (Point): Точка, для которой нужно определить полосу.

    Returns:
        int: Индекс полосы, содержащей точку.
    """
    low, high = 0, len(strips) - 1
    while low <= high:
        mid = (low + high) // 2
        if strips[mid] == point[1]:
            return mid
        elif strips[mid] < point[1]:
            low = mid + 1
        else:
            high = mid - 1
    return high


# Поиск ребер в полосе.
def find_edges_in_strip(edges: List[Edge], strip_y1: float, strip_y2: float) -> List[Edge]:
    """Находит ребра, пересекающие указанную полосу.

    Args:
        edges (List[Edge]): Список ребер графа.
        strip_y1 (float): Нижняя граница полосы.
        strip_y2 (float): Верхняя граница полосы.

    Returns:
        List[Edge]: Список ребер, пересекающих полосу.
    """
    edges_in_strip = []
    for edge in edges:
        if (edge[0][1] <= strip_y1 and edge[1][1] >= strip_y2) or \
                (edge[1][1] <= strip_y1 and edge[0][1] >= strip_y2):
            edges_in_strip.append(edge)
    return edges_in_strip


# Определение трапеции, содержащей заданную точку в пересекающихся ребрах.
def locate_point_in_trapezoids(edges_in_strip: List[Edge], point: Point) -> Optional[Tuple[Point, Point, Point, Point]]:
    """Определяет трапецию, содержащую заданную точку среди пересекающихся ребер.

    Args:
        edges_in_strip (List[Edge]): Список ребер, пересекающих полосу.
        point (Point): Точка, для которой определяется содержащая трапеция.

    Returns:
        Optional[Tuple[Point, Point, Point, Point]]: Координаты вершин содержащей трапеции,
            или None, если трапеция не найдена.
    """
    sorted_edges = sorted(edges_in_strip, key=lambda edge: min(edge[0][0], edge[1][0]))
    for i, edge in enumerate(sorted_edges):
        if i < len(sorted_edges) - 1:
            next_edge = sorted_edges[i + 1]
            if min(edge[0][0], edge[1][0]) <= point[0] <= max(next_edge[0][0], next_edge[1][0]):
                return edge[0], edge[1], next_edge[0], next_edge[1]
    return None


# Визуализация графа, горизонтальных полос и указанной точки.
def plot_graph_edges_and_point(vertices: List[Point], edges: List[Edge], point: Point, strips: List[float]) -> None:
    """Визуализирует граф, горизонтальные полосы и указанную точку.

    Args:
        vertices (List[Point]): Список вершин графа.
        edges (List[Edge]): Список ребер графа.
        point (Point): Указанная точка.
        strips (List[float]): Список горизонтальных полос.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))

    # Построение графа G
    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'b-')

    # Построение вершин графа
    x_vertices, y_vertices = zip(*vertices)
    plt.scatter(x_vertices, y_vertices, color='red', zorder=5)

    # Построение горизонтальных полос
    for strip in strips:
        plt.axhline(y=strip, color='gray', linestyle='--')

    # Построение точки A
    plt.scatter([point[0]], [point[1]], color='green', zorder=5, label='Point A')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Граф с точкой A')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


vertices: List[Point] = [(1, 1), (10, 10), (8, 2), (6, 0), (11, 5), (14, 2), (11, 0)]
edges: List[Edge] = [((1, 1), (10, 10)), ((10, 10), (8, 2)), ((8, 2), (6, 0)), ((6, 0), (1, 1)),
                     ((5, 5), (11, 5)), ((11, 5), (14, 2)), ((14, 2), (11, 0)), ((11, 0), (5, 0))]
point_A: Point = (6, 3)

horizontal_strips: List[float] = create_horizontal_strips(vertices)
strip_index: int = find_strip_for_point(horizontal_strips, point_A)
if 0 <= strip_index < len(horizontal_strips) - 1:
    strip_y1, strip_y2 = horizontal_strips[strip_index], horizontal_strips[strip_index + 1]
else:
    strip_y1, strip_y2 = None, None

if strip_y1 is not None and strip_y2 is not None:
    edges_in_strip: List[Edge] = find_edges_in_strip(edges, strip_y1, strip_y2)
    trapezoid: Optional[Tuple[Point, Point, Point, Point]] = locate_point_in_trapezoids(edges_in_strip, point_A)
else:
    trapezoid = None

print(f"Трапеция: {trapezoid}")

plot_graph_edges_and_point(vertices, edges, point_A, horizontal_strips)
