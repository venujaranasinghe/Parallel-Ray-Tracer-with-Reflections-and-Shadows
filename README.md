# Barnes-Hut-N-Body-Simulation-for-Parallel-Computing
The Barnes-Hut algorithm simulates the motion of N bodies (stars, planets, particles) under gravitational forces. Unlike the direct O(N²) approach that calculates all pairwise interactions, Barnes-Hut uses a spatial octree/quadtree to group distant bodies, reducing complexity to O(N log N).
