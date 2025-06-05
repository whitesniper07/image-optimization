#ifndef LINE_ALGORITHM_H
#define LINE_ALGORITHM_H

#include <vector>
#include <utility> // for std::pair
#include <cmath>   // for std::sqrt
#include <iostream>
#include <algorithm>
#include "basic_setup_tools.h"

using namespace std;


// 3D Vector structure for computations
struct Vector3 {
    float x, y, z;
    Vector3(float x = 0.0, float y = 0.0, float z = 0.0) : x(x), y(y), z(z) {}
};

// Vector addition
Vector3 operator+(const Vector3& a, const Vector3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

// Vector subtraction
Vector3 operator-(const Vector3& a, const Vector3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

// Scalar multiplication
Vector3 operator*(float s, const Vector3& v) {
    return {s * v.x, s * v.y, s * v.z};
}

// Dot product
float dot(const Vector3& a, const Vector3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross product
Vector3 cross(const Vector3& a, const Vector3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// Vector norm (magnitude)
float norm(const Vector3& v) {
    return std::sqrt(dot(v, v));
}

// Normalize vector
Vector3 normalize(const Vector3& v) {
    float n = norm(v);
    if (n == 0.0) return v; // Return unchanged if zero vector
    return {v.x / n, v.y / n, v.z / n};
}

// Matrix-vector multiplication for 3x3 matrix
Vector3 matrix_multiply(const float mat[3][3], const Vector3& v) {
    return {
        mat[0][0] * v.x + mat[0][1] * v.y + mat[0][2] * v.z,
        mat[1][0] * v.x + mat[1][1] * v.y + mat[1][2] * v.z,
        mat[2][0] * v.x + mat[2][1] * v.y + mat[2][2] * v.z
    };
}

// Power iteration to find dominant eigenvector
Vector3 power_iteration(const float mat[3][3],const int max_iter = 20,const float tol = 1e-6) {
    Vector3 v = {1.0, 1.0, 1.0}; // Initial guess
    for (int i = 0; i < max_iter; ++i) {
        Vector3 w = matrix_multiply(mat, v);
        float n = norm(w);
        if (n < tol) {
            return {1.0, 0.0, 0.0}; // Arbitrary direction if matrix is near-zero
        }
        Vector3 v_new = {w.x / n, w.y / n, w.z / n};
        Vector3 diff = v_new - v;
        if (norm(diff) < tol) {
            break;
        }
        v = v_new;
    }
    return v;
}

// Main function to check if new_pixel is on the line
bool is_on_line(vector<RGB>& pixels,vector<float>& pixels_distance,const RGB& new_pixel,const float& threshold) {
    int n = pixels.size();
    if (n < 2) {
        return false; // Need at least 2 points to define a line
    }

    // Compute centroid
    Vector3 centroid = {0.0, 0.0, 0.0};
    for (const auto& p : pixels) {
        centroid.x += p.r;
        centroid.y += p.g;
        centroid.z += p.b;
    }
    centroid = (1.0 / n) * centroid;

    // Compute covariance matrix
    float cov[3][3] = {{0.0}};
    for (const auto& p : pixels) {
        Vector3 centered = {p.r - centroid.x, p.g - centroid.y, p.b - centroid.z};
        cov[0][0] += centered.x * centered.x;
        cov[0][1] += centered.x * centered.y;
        cov[0][2] += centered.x * centered.z;
        cov[1][0] += centered.y * centered.x;
        cov[1][1] += centered.y * centered.y;
        cov[1][2] += centered.y * centered.z;
        cov[2][0] += centered.z * centered.x;
        cov[2][1] += centered.z * centered.y;
        cov[2][2] += centered.z * centered.z;
    }

    // Find line direction using power iteration
    Vector3 direction = power_iteration(cov);
    direction = normalize(direction); // Ensure unit vector

    // Compute distance from new_pixel to the line
    Vector3 new_vec = {new_pixel.r, new_pixel.g, new_pixel.b};
    Vector3 diff = new_vec - centroid;
    Vector3 cross_prod = cross(diff, direction);
    float distance = norm(cross_prod);

    // Check threshold and update pixels if within
    if (distance <= threshold) {
        pixels.push_back(new_pixel);
        pixels_distance.push_back(distance);
        return true;
    }
    return false;
}

// Main function to adjust the line and include new_pixel
bool can_be_on_line(vector<RGB>& pixels,vector<float>& pixels_distance,const RGB& new_pixel,const float& threshold) {
    // Handle edge case: need at least one point to define a line with new_pixel
    if (pixels.empty()) {
        pixels.push_back(new_pixel);
        pixels_distance.push_back(0.0);
        return true;
    }

    // Collect all points including new_pixel
    std::vector<RGB> all_points = pixels;
    all_points.push_back(new_pixel);
    int n = all_points.size();

    // Step 1: Compute the centroid
    Vector3 centroid = {0.0, 0.0, 0.0};
    for (const auto& p : all_points) {
        centroid.x += p.r;
        centroid.y += p.g;
        centroid.z += p.b;
    }
    centroid = (1.0 / n) * centroid;

    // Step 2: Compute the covariance matrix
    float cov[3][3] = {{0.0}};
    for (const auto& p : all_points) {
        Vector3 centered = {p.r - centroid.x, p.g - centroid.y, p.b - centroid.z};
        cov[0][0] += centered.x * centered.x;
        cov[0][1] += centered.x * centered.y;
        cov[0][2] += centered.x * centered.z;
        cov[1][0] += centered.y * centered.x;
        cov[1][1] += centered.y * centered.y;
        cov[1][2] += centered.y * centered.z;
        cov[2][0] += centered.z * centered.x;
        cov[2][1] += centered.z * centered.y;
        cov[2][2] += centered.z * centered.z;
    }

    // Step 3: Find the line direction (dominant eigenvector)
    Vector3 direction = normalize(power_iteration(cov));

    // Step 4: Compute distances to the new line and check threshold
    bool all_within_threshold = true;
    std::vector<float> new_distances(n - 1, 0.0); // Store distances for existing pixels
    float new_pixel_distance = 0.0;

    for (int i = 0; i < n; ++i) {
        Vector3 point = {all_points[i].r, all_points[i].g, all_points[i].b};
        Vector3 diff = point - centroid;
        Vector3 cross_prod = cross(diff, direction);
        float distance = norm(cross_prod);

        if (distance > threshold) {
            all_within_threshold = false;
            break;
        }

        if (i < n - 1) {
            new_distances[i] = distance;
        } else {
            new_pixel_distance = distance;
        }
    }

    // Step 5: If all points are within threshold, update pixels and return true
    if (all_within_threshold) {
        for (int i = 0; i < pixels_distance.size(); ++i) {
            pixels_distance[i] = new_distances[i];
        }
        pixels.push_back(new_pixel);
        pixels_distance.push_back(new_pixel_distance);
        return true;
    }

    return false;
}


bool search_on_the_line(vector<RGB>& line, vector<float>& line_distance,const RGB& new_point,const float& threshold) {
    return is_on_line(line, line_distance, new_point, threshold) || 
        can_be_on_line(line, line_distance, new_point, threshold);
}

auto calculate_lines(const vector<RGB>& pixels_list,float threshold) {
    if (pixels_list.empty()) {
        return vector<vector<RGB>>{};
    }

    // Preallocate an estimated number of lines to reduce reallocations.
    vector<vector<RGB>> lines;
    vector<vector<float>> lines_distences;

    lines.reserve(pixels_list.size() / 2);
    lines_distences.reserve(pixels_list.size() / 2);

    vector<RGB>* last_line = nullptr;
    vector<float>* last_line_distence = nullptr;

    // Process pixels
    for (size_t i = 0; i < pixels_list.size(); ++i) {
        bool added = false;
        const RGB& current_pixel = pixels_list[i];


        // Try extending the most recent line first.
        if (last_line && search_on_the_line(*last_line,*last_line_distence, current_pixel,(current_pixel.is_setted <= 1) ? threshold/2 : threshold)) {
            added = true;
        } else if (!lines.empty()) {
            // Only check up to the 16 most recent lines.
            size_t end = min(lines.size(), static_cast<size_t>(32));
            for (int j = static_cast<int>(end) - 1; j >= 0; --j) {
                if (search_on_the_line(lines[j], lines_distences[j], current_pixel,(current_pixel.is_setted <= 1) ? threshold/2 : threshold)) {
                    added = true;
                    last_line = &lines[j];
                    last_line_distence = &lines_distences[j];
                    break;
                }
            }
        }

        // If not added, start a new line.
        if (!added) {
            lines.push_back({current_pixel});
            lines_distences.push_back({0.0});

            last_line = &lines.back();
            last_line_distence = &lines_distences.back();

            // If possible, initialize new line with the next pixel.
            if (i + 1 < pixels_list.size()) {
                lines.back().push_back(pixels_list[i + 1]);
                lines_distences.back().push_back(0.0);
                ++i;
            }
        }
    }
    return lines;
}

pair<RGB,RGB> find_end_points(const vector<RGB>& pixels){
    if (pixels.size() == 2)
        return {pixels[0], pixels[1]};
    
    float max_dist_sq = 0.0;
    pair<int, int> final_index = {0, 1};
    for (size_t x = 0; x < pixels.size(); x++)
    {
        for (size_t y = x + 1; y < pixels.size(); y++)
        {
            float dr = pixels[x].r - pixels[y].r;
            float dg = pixels[x].g - pixels[y].g;
            float db = pixels[x].b - pixels[y].b;
            float dist_sq = dr * dr + dg * dg + db * db;
            if (dist_sq > max_dist_sq) {
                max_dist_sq = dist_sq;
                final_index.first = x;
                final_index.second = y;
            }
        }
    }
    return {pixels[final_index.first], pixels[final_index.second]};
}

#endif