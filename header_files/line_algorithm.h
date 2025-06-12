#ifndef LINE_ALGORITHM_H
#define LINE_ALGORITHM_H

#include <utility> // for std::pair
#include <cmath>   // for std::sqrt
#include "basic_setup_tools.h"


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
pair<bool, float> is_on_line(
    vector<MRGB>& pixels,
    vector<int>& pixels_indexs,
    vector<float>& pixels_distance,
    const int new_pixel_index,
    const float& threshold) {


    int n = pixels_indexs.size();
    if (n < 2) {
        return {false, NULL}; // Need at least 2 points to define a line
    }

    // Compute centroid
    Vector3 centroid = {0.0, 0.0, 0.0};
    for (const auto& p : pixels_indexs) {
        centroid.x += pixels[p].value.r;
        centroid.y += pixels[p].value.g;
        centroid.z += pixels[p].value.b;
    }
    centroid = (1.0 / n) * centroid;

    // Compute covariance matrix
    float cov[3][3] = {{0.0}};
    for (const auto& p : pixels_indexs) {
        Vector3 centered = {pixels[p].value.r - centroid.x,
                            pixels[p].value.g - centroid.y,
                            pixels[p].value.b - centroid.z};

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
    Vector3 new_vec = {pixels[new_pixel_index].value.r,
                       pixels[new_pixel_index].value.g,
                       pixels[new_pixel_index].value.b};
    
    Vector3 diff = new_vec - centroid;
    Vector3 cross_prod = cross(diff, direction);
    float distance = norm(cross_prod);

    // Check threshold and update pixels if within
    if (distance <= threshold) {
        // pixels_indexs.push_back(new_pixel_index);
        // pixels_distance.push_back(distance);
        return {true, distance};
    }
    return {false, NULL};
}


// Main function to adjust the line and include new_pixel
pair<bool,pair<float, vector<float>>> can_be_on_line(
    vector<MRGB>& pixels,
    vector<int>& pixels_indexs,
    vector<float>& pixels_distance,
    const int& new_pixel_index,
    const float& threshold) {

    // Handle edge case: need at least one point to define a line with new_pixel
    if (pixels_indexs.empty()) {
        pixels_indexs.push_back(new_pixel_index);
        pixels_distance.push_back(0.0);
        return {true, {0.0, pixels_distance}};
    }

    // Collect all points including new_pixel
    vector<int> all_points = pixels_indexs;
    all_points.push_back(new_pixel_index);
    int n = all_points.size();

    // Step 1: Compute the centroid
    Vector3 centroid = {0.0, 0.0, 0.0};
    for (const auto& p : all_points) {
        centroid.x += pixels[p].value.r;
        centroid.y += pixels[p].value.g;
        centroid.z += pixels[p].value.b;
    }
    centroid = (1.0 / n) * centroid;

    // Step 2: Compute the covariance matrix
    float cov[3][3] = {{0.0}};
    for (const auto& p : all_points) {
        Vector3 centered = {
            pixels[p].value.r - centroid.x,
            pixels[p].value.g - centroid.y,
            pixels[p].value.b - centroid.z
        };


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
    vector<float> new_distances(n - 1, 0.0); // Store distances for existing pixels
    float new_pixel_distance = 0.0;

    for (int i = 0; i < n; ++i) {
        Vector3 point = {
            pixels[all_points[i]].value.r,
            pixels[all_points[i]].value.g,
            pixels[all_points[i]].value.b
        };

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
        // for (int i = 0; i < pixels_distance.size(); ++i) {
        //     pixels_distance[i] = new_distances[i];
        // }
        // pixels_indexs.push_back(new_pixel_index);
        // pixels_distance.push_back(new_pixel_distance);
        return {true, {new_pixel_distance, new_distances}};
    }

    return {false, {NULL, {}}};
}

auto search_on_the_line(vector<int>* last_line, vector<float>* last_line_distence, const int current_pixel, const float threshold) {

}

auto calculate_lines(const vector<MRGB>& pixels, const float threshold) {
    if (pixels.empty()) {
        return vector<vector<int>>{};
    }

    vector<vector<int>> lines;
    vector<vector<float>> lines_distences;
    vector<int> pixels_indexs;

    pixels_indexs.reserve(pixels.size());
    for (size_t i = 0; i < pixels.size(); i++) {
        pixels_indexs[i] = i;
    }
    

    lines.reserve(pixels.size() / 2);
    lines_distences.reserve(pixels.size() / 2);

    vector<int>* last_line = nullptr;
    vector<float>* last_line_distence = nullptr;

    // Process pixels
    for (size_t i = 0; i < pixels_indexs.size(); ++i) {
        bool added = false;
        const int current_pixel = i;

        // && search_on_the_line(*last_line,*last_line_distence, current_pixel,(current_pixel.is_setted <= 1) ? threshold/2 : threshold)
        if (last_line) {
            added = true;
        } 
        else if (!lines.empty()) {
            int end = min(static_cast<int>(lines.size()), 32);

            for (int j = end - 1; j >= 0; --j) {
                // search_on_the_line(lines[j], lines_distences[j], current_pixel,(current_pixel.is_setted <= 1) ? threshold/2 : threshold)
                if (true) {
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
            if (i + 1 < pixels_indexs.size()) {
                lines.back().push_back(i + 1);
                lines_distences.back().push_back(0.0);
                ++i;
            }
        }
    }
    return lines;
}


pair<RGB,RGB> find_end_points(const vector<MRGB>& pixels, vector<int>& pixels_indexs){
    if (pixels_indexs.size() == 2)
        return {pixels[pixels_indexs[0]].value, pixels[pixels_indexs[1]].value};
    
    float max_dist_sq = 0.0;
    pair<int, int> final_index = {0, 1};
    
    for (size_t x = 0; x < pixels_indexs.size(); x++) {
        for (size_t y = x + 1; y < pixels_indexs.size(); y++) {
            float dr = pixels[pixels_indexs[x]].value.r - pixels[pixels_indexs[y]].value.r;
            float dg = pixels[pixels_indexs[x]].value.g - pixels[pixels_indexs[y]].value.g;
            float db = pixels[pixels_indexs[x]].value.b - pixels[pixels_indexs[y]].value.b;
            float dist_sq = dr * dr + dg * dg + db * db;
            if (dist_sq > max_dist_sq) {
                max_dist_sq = dist_sq;
                final_index.first = x;
                final_index.second = y;
            }
        }
    }
    return {pixels[final_index.first].value, pixels[final_index.second].value};
}



#endif