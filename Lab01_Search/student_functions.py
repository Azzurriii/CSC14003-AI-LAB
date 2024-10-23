import math
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
    heuristic: Euclidean distance based on positions' parameter.

    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix where matrix[i][j] is the edge weight between node i and node j.
    start: integer
        Starting node.
    end: integer
        Ending node.
    pos: dictionary
        Keys are nodes, values are positions (tuples or lists representing coordinates).
        Example: {node: (x, y)}

    Returns:
    ---------------------
    visited: dict
        A dictionary where keys are visited nodes and values are tuples of (predecessor node, cost to reach the node).
    path: list
        The list of nodes representing the found path from start to end.
    """

    def heuristic(pos1, pos2):
        # Euclidean distance
        return math.sqrt((pos[pos1][0] - pos[pos2][0]) ** 2 + (pos[pos1][1] - pos[pos2][1]) ** 2)

    # Initialize the variables
    path = []
    visited = {}
    pq = []  # Priority queue using total cost (g + h)
    heapq.heappush(pq, (0, start))  # Push the start node with cost 0
    visited[start] = (None, 0) # Mark the start node as visited

    # Process the nodes in the priority queue
    while pq:
        _, node = heapq.heappop(pq)  # Pop the node with the lowest cost + heuristic

        # If the end node is reached, stop the search
        if node == end:
            break

        # Explore the neighbors of the current node
        for neighbor, cost in enumerate(matrix[node]):
            if cost > 0:  # If there is a connection between the nodes (non-zero edge)
                new_cost = visited[node][1] + cost  # Calculate the new cost to reach this neighbor

                # If this neighbor hasn't been visited, or we've found a cheaper path
                if neighbor not in visited or new_cost < visited[neighbor][1]:
                    total_cost = new_cost + heuristic(neighbor, end)  # Calculate total cost (g + h)
                    heapq.heappush(pq, (total_cost, neighbor))  # Push the neighbor with the total cost
                    visited[neighbor] = (node, new_cost)  # Mark the neighbor as visited

    # Reconstruct the path from end to start if the end node was reached
    if end in visited:
        node = end
        while node is not None:
            path.insert(0, node)  # Insert the node at the beginning of the path list
            node = visited[node][0]  # Move to the predecessor

    return visited, path  # Return the visited nodes and the reconstructed path

def beam_search(matrix, start, end, beam_width):
    """
    Beam Search algorithm:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix.
    start: integer
        Starting node.
    end: integer
        Ending node.
    beam_width: integer
        The number of paths to keep at each level of the search.

    Returns:
    ---------------------
    visited: dict
        A dictionary where keys are visited nodes and values are the node's predecessor.
    path: list
        A list that represents the shortest path from start to end.
    """
    visited = {start: None}  # Dictionary to keep track of visited nodes and their predecessors
    pq = []  # Initialize a heapq priority queue
    heapq.heappush(pq, (0, [start]))  # Push the start node with an initial cost of 0

    while pq:
        current_level = []

        # Process up to beam_width number of paths in the current level
        for _ in range(min(beam_width, len(pq))):
            cost, path = heapq.heappop(pq)  # Pop the path with the lowest cost
            current = path[-1]  # Get the current node (last node in the path)

            if current == end:  # If we reached the end node, return the result
                return visited, path

            # Explore neighbors of the current node
            for neighbor, weight in enumerate(matrix[current]):
                if weight != np.inf and neighbor not in visited:  # Check if the neighbor is reachable and unvisited
                    new_cost = cost + weight  # Calculate the new cost to reach the neighbor
                    new_path = path + [neighbor]  # Create a new path by adding the neighbor to the current path
                    current_level.append((new_cost, new_path))  # Add the new path to the current level
                    visited[neighbor] = current  # Mark the neighbor as visited and set its predecessor

        # Sort the current level by cost and keep only the best beam_width paths
        current_level.sort(key=lambda x: x[0])

        # Add the selected paths back to the priority queue
        for item in current_level[:beam_width]:
            heapq.heappush(pq, item)

    return visited, []  # If no path is found