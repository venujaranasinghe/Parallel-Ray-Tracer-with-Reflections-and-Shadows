#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// Vector structure
typedef struct {
    float x, y, z;
} Vec3;

// Ray structure
typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

// Sphere structure
typedef struct {
    Vec3 center;
    float radius;
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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device vector operations
__device__ Vec3 vec3_add(Vec3 a, Vec3 b) {
    Vec3 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
}

__device__ Vec3 vec3_sub(Vec3 a, Vec3 b) {
    Vec3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
}

__device__ Vec3 vec3_scale(Vec3 v, float s) {
    Vec3 result;
    result.x = v.x * s;
    result.y = v.y * s;
    result.z = v.z * s;
    return result;
}

__device__ float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float vec3_length(Vec3 v) {
    return sqrtf(vec3_dot(v, v));
}

__device__ Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    if (len > 0) {
        return vec3_scale(v, 1.0f / len);
    }
    return v;
}

// Ray-sphere intersection
__device__ int ray_sphere_intersect(Ray ray, Sphere sphere, float* t) {
    Vec3 oc = vec3_sub(ray.origin, sphere.center);
    float a = vec3_dot(ray.direction, ray.direction);
    float b = 2.0f * vec3_dot(oc, ray.direction);
    float c = vec3_dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) return 0;
    
    float sqrt_d = sqrtf(discriminant);
    float t0 = (-b - sqrt_d) / (2.0f * a);
    float t1 = (-b + sqrt_d) / (2.0f * a);
    
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
__device__ int find_closest_intersection(Ray ray, Sphere* spheres, int num_spheres, 
                                        float* t, int* sphere_index) {
    float closest_t = FLT_MAX;
    int hit = 0;
    
    for (int i = 0; i < num_spheres; i++) {
        float current_t;
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
                                   Sphere sphere, Light* lights, int num_lights) {
    Vec3 color;
    color.x = 0; color.y = 0; color.z = 0;
    
    // Ambient light
    Vec3 ambient = vec3_scale(sphere.color, 0.1f);
    color = vec3_add(color, ambient);
    
    for (int i = 0; i < num_lights; i++) {
        Vec3 light_dir = vec3_normalize(vec3_sub(lights[i].position, point));
        
        // Diffuse lighting
        float diff = fmaxf(vec3_dot(normal, light_dir), 0.0f);
        Vec3 diffuse = vec3_scale(sphere.color, diff);
        color = vec3_add(color, diffuse);
        
        // Specular lighting
        Vec3 reflect_dir = vec3_sub(vec3_scale(normal, 2.0f * vec3_dot(normal, light_dir)), light_dir);
        float spec = powf(fmaxf(vec3_dot(view_dir, reflect_dir), 0.0f), 32.0f);
        Vec3 specular = vec3_scale(lights[i].color, spec * 0.5f);
        color = vec3_add(color, specular);
    }
    
    // Clamp color values
    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);
    
    return color;
}

// Cast ray and compute color (iterative version to avoid recursion limits)
__device__ Vec3 cast_ray(Ray ray, Sphere* spheres, int num_spheres, 
                        Light* lights, int num_lights) {
    Vec3 final_color;
    final_color.x = 0; final_color.y = 0; final_color.z = 0;
    
    float reflection_weight = 1.0f;
    Ray current_ray = ray;
    
    for (int depth = 0; depth < MAX_RAY_DEPTH; depth++) {
        float t;
        int sphere_index;
        
        if (find_closest_intersection(current_ray, spheres, num_spheres, &t, &sphere_index)) {
            Vec3 point = vec3_add(current_ray.origin, vec3_scale(current_ray.direction, t));
            Vec3 normal = vec3_normalize(vec3_sub(point, spheres[sphere_index].center));
            Vec3 view_dir = vec3_normalize(vec3_scale(current_ray.direction, -1.0f));
            
            Vec3 color = calculate_lighting(point, normal, view_dir, 
                                          spheres[sphere_index], lights, num_lights);
            
            // Accumulate color with current reflection weight
            final_color = vec3_add(final_color, vec3_scale(color, reflection_weight * 0.7f));
            
            // Prepare for reflection
            if (depth < MAX_RAY_DEPTH - 1) {
                Vec3 reflect_dir = vec3_sub(current_ray.direction, 
                    vec3_scale(normal, 2.0f * vec3_dot(current_ray.direction, normal)));
                current_ray.origin = point;
                current_ray.direction = vec3_normalize(reflect_dir);
                reflection_weight *= 0.3f;
            } else {
                break;
            }
        } else {
            // Background gradient
            float t_bg = 0.5f * (current_ray.direction.y + 1.0f);
            Vec3 bg_color = vec3_add(vec3_scale((Vec3){1.0f, 1.0f, 1.0f}, 1.0f - t_bg),
                                    vec3_scale((Vec3){0.5f, 0.7f, 1.0f}, t_bg));
            final_color = vec3_add(final_color, vec3_scale(bg_color, reflection_weight));
            break;
        }
    }
    
    return final_color;
}

// CUDA kernel for ray tracing
__global__ void raytrace_kernel(Vec3* image, Sphere* spheres, Light* lights,
                               Vec3 camera_pos, float viewport_width, float viewport_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= IMAGE_WIDTH || y >= IMAGE_HEIGHT) return;
    
    float u = (float)x / (IMAGE_WIDTH - 1);
    float v = (float)y / (IMAGE_HEIGHT - 1);
    
    Vec3 ray_dir;
    ray_dir.x = (u - 0.5f) * viewport_width;
    ray_dir.y = (0.5f - v) * viewport_height;  // Flip y-axis
    ray_dir.z = -1.0f;
    ray_dir = vec3_normalize(ray_dir);
    
    Ray ray;
    ray.origin = camera_pos;
    ray.direction = ray_dir;
    
    int pixel_index = y * IMAGE_WIDTH + x;
    image[pixel_index] = cast_ray(ray, spheres, MAX_SPHERES, lights, MAX_LIGHTS);
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
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA-capable device found!\n");
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using GPU: %s\n", deviceProp.name);
    
    // Initialize scene on host
    Sphere h_spheres[MAX_SPHERES] = {
        {{0, 0, -5}, 1.0f, {1.0f, 0.2f, 0.2f}},     // Red sphere
        {{2, 0, -7}, 1.5f, {0.2f, 1.0f, 0.2f}},     // Green sphere
        {{-2, 0, -6}, 1.2f, {0.2f, 0.2f, 1.0f}},    // Blue sphere
        {{0, -5000, 0}, 5000, {0.8f, 0.8f, 0.8f}},  // Large floor
        {{0, 2, -4}, 0.8f, {1.0f, 1.0f, 0.2f}}      // Yellow sphere
    };
    
    Light h_lights[MAX_LIGHTS] = {
        {{-5, 5, 0}, {1.0f, 1.0f, 1.0f}},   // White light 1
        {{5, 3, -2}, {0.8f, 0.8f, 1.0f}}    // Blueish light 2
    };
    
    // Camera setup
    Vec3 camera_pos = {0, 0, 0};
    float aspect_ratio = (float)IMAGE_WIDTH / IMAGE_HEIGHT;
    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    
    // Allocate device memory
    Sphere* d_spheres;
    Light* d_lights;
    Vec3* d_image;
    
    CUDA_CHECK(cudaMalloc(&d_spheres, MAX_SPHERES * sizeof(Sphere)));
    CUDA_CHECK(cudaMalloc(&d_lights, MAX_LIGHTS * sizeof(Light)));
    CUDA_CHECK(cudaMalloc(&d_image, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3)));
    
    // Copy scene data to device
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres, MAX_SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lights, h_lights, MAX_LIGHTS * sizeof(Light), cudaMemcpyHostToDevice));
    
    printf("Rendering scene on GPU...\n");
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x,
                  (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    printf("Grid size: (%d, %d), Block size: (%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernel
    raytrace_kernel<<<gridSize, blockSize>>>(d_image, d_spheres, d_lights,
                                             camera_pos, viewport_width, viewport_height);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Rendering time: %.2f seconds\n", milliseconds / 1000.0f);
    
    // Allocate host memory for result
    Vec3* h_image = (Vec3*)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3));
    if (!h_image) {
        printf("Error: Could not allocate host memory for image\n");
        cudaFree(d_spheres);
        cudaFree(d_lights);
        cudaFree(d_image);
        return 1;
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_image, d_image, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3), 
                         cudaMemcpyDeviceToHost));
    
    // Save image
    write_ppm("raytrace_cuda.ppm", h_image);
    
    // Cleanup
    free(h_image);
    cudaFree(d_spheres);
    cudaFree(d_lights);
    cudaFree(d_image);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Ray tracing completed!\n");
    
    return 0;
}
