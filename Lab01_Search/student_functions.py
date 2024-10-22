from queue import PriorityQueue
import numpy as np
import heapq
from collections import deque

def BFS(matrix, start, end):
    """
    BFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node

    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    queue = deque([start])  # Queue for BFS
    visited = {start: None}  # Dictionary to store the previous node of each visited node

    while queue:
        current = queue.popleft()

        # If we reach the destination node, reconstruct the path
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            path.reverse()
            return visited, path

        # Traverse neighbors
        for neighbor, is_connected in enumerate(matrix[current]):
            if is_connected and neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)

    return visited, []  # If no path is found

def DFS(matrix, start, end):
    """
    DFS algorithm:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node

    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    stack = [start]  # Stack for DFS
    visited = {start: None}  # Dictionary to store the previous node of each visited node

    while stack:
        current = stack.pop()

        # If we reach the destination node, reconstruct the path
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            path.reverse()
            return visited, path

        for neighbor, is_connected in enumerate(matrix[current]):
            if is_connected and neighbor not in visited:
                visited[neighbor] = current
                stack.append(neighbor)

    return visited, []  # If no path is found

def UCS(matrix, start, end):
    """
    Uniform Cost Search algorithm:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node

    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    pq = [(0, start)]  # Priority queue (min-heap) with (cost, node)
    visited = {start: None}  # Dictionary to store the previous node of each visited node
    costs = {start: 0}  # Cost to reach each node

    while pq:
        cost, current = heapq.heappop(pq)

        # If we reach the destination node, reconstruct the path
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            path.reverse()
            return visited, path

        for neighbor, weight in enumerate(matrix[current]):
            if weight and (neighbor not in costs or cost + weight < costs[neighbor]):
                costs[neighbor] = cost + weight
                visited[neighbor] = current
                heapq.heappush(pq, (cost + weight, neighbor))

    return visited, []

def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node

    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    pq = [(0, start)]  # Priority queue using edge weight
    visited = {start: None}  # Dictionary to store the previous node of each visited node

    while pq:
        _, current = heapq.heappop(pq)

        # If we reach the destination node, reconstruct the path
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            path.reverse()
            return visited, path

        # Traverse all neighbors
        for neighbor, weight in enumerate(matrix[current]):
            if weight > 0 and neighbor not in visited:  # weight > 0 means there is an edge
                visited[neighbor] = current
                heapq.heappush(pq, (weight, neighbor))

    return visited, []  # If no path is found

def Astar(matrix, start, end, pos):
    """
    A* Search algorithm:
    heuristic: Euclidean distance based on positions parameter.

    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        Starting node
    end: integer
        Ending node
    pos: dictionary
        Keys are nodes, values are positions (tuples or lists representing coordinates)
        Example: {node: (x, y)}

    Returns:
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which was visited before it.
    path: list
        Founded path from start to end
    """

    # Euclidean distance heuristic
    def heuristic(node, goal):
        x1, y1 = pos[node]
        x2, y2 = pos[goal]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Priority queue for A* with (f_cost, node)
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, end), start))  # f_cost = g_cost + heuristic
    g_costs = {start: 0}  # Dictionary to store g_costs (distance from start to current node)
    visited = {start: None}  # Store the previous node for path reconstruction

    while open_list:
        _, current = heapq.heappop(open_list)

        # If we reach the destination node, reconstruct the path
        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            path.reverse()
            return visited, path

        # Explore neighbors
        for neighbor, is_connected in enumerate(matrix[current]):
            if is_connected:  # Check if there is an edge to the neighbor
                tentative_g_cost = g_costs[current] + 1  # Assuming edge cost is 1, adjust if needed

                # If this path to the neighbor is better, record it
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic(neighbor, end)  # f_cost = g_cost + h_cost
                    heapq.heappush(open_list, (f_cost, neighbor))
                    visited[neighbor] = current  # Set the previous node

    return visited, []  # If no path is found

def beam_search(matrix, start, end, beam_width):
    """
    Beam Search algorithm:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    beam_width: integer
        The number of paths to keep at each level of the search

    Returns
    ---------------------
    visited, path
        The dictionary contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
        A list that represents the shortest path from start to end.
    """
    visited = {start: None}
    pq = PriorityQueue()
    pq.put((0, [start]))

    while not pq.empty():
        current_level = []
        for _ in range(min(beam_width, pq.qsize())):
            cost, path = pq.get()
            current = path[-1]

            if current == end:
                return visited, path

            for neighbor, weight in enumerate(matrix[current]):
                if weight != np.inf and neighbor not in visited:
                    new_cost = cost + weight
                    new_path = path + [neighbor]
                    current_level.append((new_cost, new_path))
                    visited[neighbor] = current

        # Sort and select the best beam_width paths
        current_level.sort(key=lambda x: x[0])
        for item in current_level[:beam_width]:
            pq.put(item)

    return visited, []