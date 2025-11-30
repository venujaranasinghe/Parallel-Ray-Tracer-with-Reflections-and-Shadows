#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

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
__device__ Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ Vec3 vec3_scale(Vec3 v, double s) {
    return (Vec3){v.x * s, v.y * s, v.z * s};
}

__device__ double vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ double vec3_length(Vec3 v) {
    return sqrt(vec3_dot(v, v));
}

__device__ Vec3 vec3_normalize(Vec3 v) {
    double len = vec3_length(v);
    if (len > 0) {
        return vec3_scale(v, 1.0 / len);
    }
    return v;
}

// Ray-sphere intersection
__device__ int ray_sphere_intersect(Ray ray, Sphere sphere, double* t) {
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
__device__ int find_closest_intersection(Ray ray, Sphere spheres[], int num_spheres, 
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
__device__ Vec3 calculate_lighting(Vec3 point, Vec3 normal, Vec3 view_dir, 
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
__device__ Vec3 cast_ray(Ray ray, Sphere spheres[], int num_spheres, 
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

// CUDA kernel for ray tracing
__global__ void ray_trace_kernel(Vec3* image, Sphere* spheres, Light* lights, 
                                int num_spheres, int num_lights) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) return;
    
    // Camera setup
    Vec3 camera_pos = {0, 0, 0};
    double aspect_ratio = (double)IMAGE_WIDTH / IMAGE_HEIGHT;
    double viewport_height = 2.0;
    double viewport_width = aspect_ratio * viewport_height;
    
    // Calculate pixel coordinates
    double u = (double)x / (IMAGE_WIDTH - 1);
    double v = (double)y / (IMAGE_HEIGHT - 1);
    
    Vec3 ray_dir = {
        (u - 0.5) * viewport_width,
        (0.5 - v) * viewport_height,  // Flip y-axis
        -1.0
    };
    ray_dir = vec3_normalize(ray_dir);
    
    Ray ray = {camera_pos, ray_dir};
    Vec3 color = cast_ray(ray, spheres, num_spheres, lights, num_lights, 0);
    
    // Store result in global memory
    int index = y * IMAGE_WIDTH + x;
    image[index] = color;
}

// Write PPM image file
void write_ppm(const char* filename, Vec3* image) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Could not open file %s\n", filename);
        return;
    }
    
    fprintf(fp, "P6\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int index = y * IMAGE_WIDTH + x;
            unsigned char pixel[3] = {
                (unsigned char)(image[index].x * 255),
                (unsigned char)(image[index].y * 255),
                (unsigned char)(image[index].z * 255)
            };
            fwrite(pixel, 1, 3, fp);
        }
    }
    
    fclose(fp);
    printf("Image saved as %s\n", filename);
}

int main() {
    printf("Starting CUDA Ray Tracer...\n");
    printf("Image size: %dx%d\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    
    // Initialize scene on host
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
    
    // Allocate device memory
    Sphere* d_spheres;
    Light* d_lights;
    Vec3* d_image;
    
    cudaMalloc(&d_spheres, MAX_SPHERES * sizeof(Sphere));
    cudaMalloc(&d_lights, MAX_LIGHTS * sizeof(Light));
    cudaMalloc(&d_image, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3));
    
    // Copy scene data to device
    cudaMemcpy(d_spheres, spheres, MAX_SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, lights, MAX_LIGHTS * sizeof(Light), cudaMemcpyHostToDevice);
    
    printf("Rendering scene with CUDA...\n");
    
    // Start timing
    clock_t start_time = clock();
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x,
                  (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    ray_trace_kernel<<<gridSize, blockSize>>>(d_image, d_spheres, d_lights, MAX_SPHERES, MAX_LIGHTS);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // End timing
    clock_t end_time = clock();
    double rendering_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Rendering time: %.2f seconds\n", rendering_time);
    
    // Allocate host memory for image and copy from device
    Vec3* h_image = (Vec3*)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3));
    cudaMemcpy(h_image, d_image, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3), cudaMemcpyDeviceToHost);
    
    // Save image
    write_ppm("raytrace_cuda.ppm", h_image);
    
    // Free memory
    free(h_image);
    cudaFree(d_spheres);
    cudaFree(d_lights);
    cudaFree(d_image);
    
    printf("CUDA Ray tracing completed!\n");
    
    return 0;
}