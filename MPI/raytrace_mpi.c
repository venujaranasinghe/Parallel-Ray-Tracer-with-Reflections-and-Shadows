/*
 * Parallel Ray Tracer using MPI
 * 
 * This implementation parallelizes the ray tracing algorithm by distributing
 * rows of the image across multiple MPI processes. Each process computes a 
 * portion of the image independently, then results are gathered at the root.
 *
 * Parallelization Strategy:
 * - Master-Worker pattern with row-level decomposition
 * - Each process computes a contiguous block of rows
 * - Process 0 (root) gathers all results and writes the final image
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

// Vector structure
typedef struct {
    double x, y, z;
} Vec3;

// Ray structure
typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

// Sphere structure
typedef struct {
    Vec3 center;
    double radius;
    Vec3 color;
} Sphere;

// Light structure
typedef struct {
    Vec3 position;
    Vec3 color;
} Light;

// Scene configuration
#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 600
#define MAX_SPHERES 5
#define MAX_LIGHTS 2
#define MAX_RAY_DEPTH 3

// Vector operations
Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 vec3_scale(Vec3 v, double s) {
    return (Vec3){v.x * s, v.y * s, v.z * s};
}

double vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double vec3_length(Vec3 v) {
    return sqrt(vec3_dot(v, v));
}

Vec3 vec3_normalize(Vec3 v) {
    double len = vec3_length(v);
    if (len > 0) {
        return vec3_scale(v, 1.0 / len);
    }
    return v;
}

// Ray-sphere intersection
int ray_sphere_intersect(Ray ray, Sphere sphere, double* t) {
    Vec3 oc = vec3_sub(ray.origin, sphere.center);
    double a = vec3_dot(ray.direction, ray.direction);
    double b = 2.0 * vec3_dot(oc, ray.direction);
    double c = vec3_dot(oc, oc) - sphere.radius * sphere.radius;
    double discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) return 0;
    
    double sqrt_d = sqrt(discriminant);
    double t0 = (-b - sqrt_d) / (2.0 * a);
    double t1 = (-b + sqrt_d) / (2.0 * a);
    
    if (t0 > 0) {
        *t = t0;
        return 1;
    } else if (t1 > 0) {
        *t = t1;
        return 1;
    }
    return 0;
}

// Find closest intersection
int find_closest_intersection(Ray ray, Sphere spheres[], int num_spheres, 
                             double* t, int* sphere_index) {
    double closest_t = DBL_MAX;
    int hit = 0;
    
    for (int i = 0; i < num_spheres; i++) {
        double current_t;
        if (ray_sphere_intersect(ray, spheres[i], &current_t)) {
            if (current_t < closest_t) {
                closest_t = current_t;
                *sphere_index = i;
                hit = 1;
            }
        }
    }
    
    if (hit) {
        *t = closest_t;
        return 1;
    }
    return 0;
}

// Calculate lighting
Vec3 calculate_lighting(Vec3 point, Vec3 normal, Vec3 view_dir, 
                       Sphere sphere, Light lights[], int num_lights) {
    Vec3 color = {0, 0, 0};
    
    // Ambient light
    Vec3 ambient = vec3_scale(sphere.color, 0.1);
    color = vec3_add(color, ambient);
    
    for (int i = 0; i < num_lights; i++) {
        Vec3 light_dir = vec3_normalize(vec3_sub(lights[i].position, point));
        
        // Diffuse lighting
        double diff = fmax(vec3_dot(normal, light_dir), 0.0);
        Vec3 diffuse = vec3_scale(sphere.color, diff);
        color = vec3_add(color, diffuse);
        
        // Specular lighting (simplified)
        Vec3 reflect_dir = vec3_sub(vec3_scale(normal, 2.0 * vec3_dot(normal, light_dir)), light_dir);
        double spec = pow(fmax(vec3_dot(view_dir, reflect_dir), 0.0), 32);
        Vec3 specular = vec3_scale(lights[i].color, spec * 0.5);
        color = vec3_add(color, specular);
    }
    
    // Clamp color values
    color.x = fmin(fmax(color.x, 0.0), 1.0);
    color.y = fmin(fmax(color.y, 0.0), 1.0);
    color.z = fmin(fmax(color.z, 0.0), 1.0);
    
    return color;
}

// Cast ray and compute color
Vec3 cast_ray(Ray ray, Sphere spheres[], int num_spheres, 
             Light lights[], int num_lights, int depth) {
    if (depth >= MAX_RAY_DEPTH) {
        return (Vec3){0, 0, 0}; // Background color
    }
    
    double t;
    int sphere_index;
    
    if (find_closest_intersection(ray, spheres, num_spheres, &t, &sphere_index)) {
        Vec3 point = vec3_add(ray.origin, vec3_scale(ray.direction, t));
        Vec3 normal = vec3_normalize(vec3_sub(point, spheres[sphere_index].center));
        Vec3 view_dir = vec3_normalize(vec3_scale(ray.direction, -1));
        
        Vec3 color = calculate_lighting(point, normal, view_dir, 
                                      spheres[sphere_index], lights, num_lights);
        
        // Simple reflection
        if (depth < MAX_RAY_DEPTH - 1) {
            Vec3 reflect_dir = vec3_sub(ray.direction, 
                vec3_scale(normal, 2.0 * vec3_dot(ray.direction, normal)));
            Ray reflect_ray = {point, vec3_normalize(reflect_dir)};
            Vec3 reflect_color = cast_ray(reflect_ray, spheres, num_spheres, 
                                        lights, num_lights, depth + 1);
            color = vec3_add(vec3_scale(color, 0.7), vec3_scale(reflect_color, 0.3));
        }
        
        return color;
    }
    
    // Background gradient
    double t_bg = 0.5 * (ray.direction.y + 1.0);
    return vec3_add(vec3_scale((Vec3){1.0, 1.0, 1.0}, 1.0 - t_bg),
                   vec3_scale((Vec3){0.5, 0.7, 1.0}, t_bg));
}

// Write PPM image file
void write_ppm(const char* filename, Vec3** image) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "P6\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            unsigned char pixel[3] = {
                (unsigned char)(image[y][x].x * 255),
                (unsigned char)(image[y][x].y * 255),
                (unsigned char)(image[y][x].z * 255)
            };
            fwrite(pixel, 1, 3, fp);
        }
    }
    
    fclose(fp);
    printf("Image saved as %s\n", filename);
}

int main(int argc, char** argv) {
    int rank, size;
    double start_time, end_time;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Only process 0 prints initial information
    if (rank == 0) {
        printf("Starting MPI Ray Tracer...\n");
        printf("Number of processes: %d\n", size);
        printf("Image size: %dx%d\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    }
    
    // Initialize scene (same on all processes)
    Sphere spheres[MAX_SPHERES] = {
        {{0, 0, -5}, 1.0, {1.0, 0.2, 0.2}},     // Red sphere
        {{2, 0, -7}, 1.5, {0.2, 1.0, 0.2}},     // Green sphere
        {{-2, 0, -6}, 1.2, {0.2, 0.2, 1.0}},    // Blue sphere
        {{0, -5000, 0}, 5000, {0.8, 0.8, 0.8}}, // Large floor
        {{0, 2, -4}, 0.8, {1.0, 1.0, 0.2}}      // Yellow sphere
    };
    
    Light lights[MAX_LIGHTS] = {
        {{-5, 5, 0}, {1.0, 1.0, 1.0}},  // White light 1
        {{5, 3, -2}, {0.8, 0.8, 1.0}}   // Blueish light 2
    };
    
    // Camera setup
    Vec3 camera_pos = {0, 0, 0};
    double aspect_ratio = (double)IMAGE_WIDTH / IMAGE_HEIGHT;
    double viewport_height = 2.0;
    double viewport_width = aspect_ratio * viewport_height;
    
    // Calculate workload distribution
    // Each process computes a contiguous block of rows
    int rows_per_process = IMAGE_HEIGHT / size;
    int remainder = IMAGE_HEIGHT % size;
    
    // Calculate start and end row for this process
    // Distribute remainder rows to first 'remainder' processes
    int start_row, end_row, local_rows;
    if (rank < remainder) {
        local_rows = rows_per_process + 1;
        start_row = rank * local_rows;
    } else {
        local_rows = rows_per_process;
        start_row = rank * rows_per_process + remainder;
    }
    end_row = start_row + local_rows;
    
    if (rank == 0) {
        printf("Rendering scene...\n");
    }
    
    // Allocate memory for local image portion
    Vec3* local_image = (Vec3*)malloc(local_rows * IMAGE_WIDTH * sizeof(Vec3));
    if (!local_image) {
        printf("Process %d: Error allocating local image memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Each process renders its assigned rows
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            double u = (double)x / (IMAGE_WIDTH - 1);
            double v = (double)y / (IMAGE_HEIGHT - 1);
            
            Vec3 ray_dir = {
                (u - 0.5) * viewport_width,
                (0.5 - v) * viewport_height,  // Flip y-axis
                -1.0
            };
            ray_dir = vec3_normalize(ray_dir);
            
            Ray ray = {camera_pos, ray_dir};
            
            // Store in local image array (row-major order)
            int local_y = y - start_row;
            local_image[local_y * IMAGE_WIDTH + x] = 
                cast_ray(ray, spheres, MAX_SPHERES, lights, MAX_LIGHTS, 0);
        }
    }
    
    // Synchronize before gathering
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Print rendering time
    if (rank == 0) {
        printf("Rendering time: %.4f seconds\n", end_time - start_time);
    }
    
    // Prepare for gathering results at root process
    Vec3* full_image = NULL;
    int* recvcounts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        // Allocate full image buffer on root
        full_image = (Vec3*)malloc(IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(Vec3));
        if (!full_image) {
            printf("Error: Could not allocate full image memory\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Calculate receive counts and displacements for MPI_Gatherv
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            int proc_rows;
            if (i < remainder) {
                proc_rows = rows_per_process + 1;
            } else {
                proc_rows = rows_per_process;
            }
            
            // Each Vec3 has 3 doubles, and we have proc_rows * IMAGE_WIDTH pixels
            recvcounts[i] = proc_rows * IMAGE_WIDTH * 3;
            
            // Calculate displacement
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
    }
    
    // Gather all image data to root process
    // Using MPI_Gatherv to handle potentially unequal distribution
    int sendcount = local_rows * IMAGE_WIDTH * 3;  // 3 doubles per Vec3
    
    MPI_Gatherv(local_image, sendcount, MPI_DOUBLE,
                full_image, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Root process writes the image file
    if (rank == 0) {
        // Convert flat array to 2D array for write_ppm function
        Vec3** image_2d = (Vec3**)malloc(IMAGE_HEIGHT * sizeof(Vec3*));
        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            image_2d[y] = &full_image[y * IMAGE_WIDTH];
        }
        
        write_ppm("raytrace_parallel.ppm", image_2d);
        
        // Print final statistics
        printf("Total execution time: %.4f seconds\n", end_time - start_time);
        printf("Ray tracing completed!\n");
        
        // Free root-specific memory
        free(image_2d);
        free(full_image);
        free(recvcounts);
        free(displs);
    }
    
    // Free local memory
    free(local_image);
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
