#ifndef LINE_ALGORITHM_H
#define LINE_ALGORITHM_H

#include <utility> // for std::pair
#include <cmath>   // for std::sqrt
#include "basic_setup_tools.h"
#include <map>
#include <immintrin.h>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <cfloat>
#include <thread>
#include <chrono>

class LoadingBar {
private:
    int maxValue;
    int currentValue;
    int barWidth;
    std::string fillChar;
    std::string emptyChar;

public:
    // Constructor to initialize the loading bar
    LoadingBar(int max, int width = 50, std::string fill = "█", std::string empty = "░") 
        : maxValue(max), currentValue(0), barWidth(width), fillChar(std::move(fill)), emptyChar(empty) {}
    
    // Update the current value and display the bar
    void update(int value) {
        currentValue = value;
        if (currentValue > maxValue) {
            currentValue = maxValue;
        }
        display();
    }
    
    // Increment the current value by a specific amount
    void increment(int step = 1) {
        currentValue += step;
        if (currentValue > maxValue) {
            currentValue = maxValue;
        }
        display();
    }
    
    // Display the loading bar
    void display() {
        float progress = static_cast<float>(currentValue) / maxValue;
        int filledWidth = static_cast<int>(progress * barWidth);
        
        // Clear the current line and move cursor to beginning
        std::cout << "\r";
        
        // Draw the bar
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i) {
            if (i < filledWidth) {
                std::cout << fillChar;
            } else {
                std::cout << emptyChar;
            }
        }
        std::cout << "] ";
        
        // Show percentage and current/max values
        std::cout << static_cast<int>(progress * 100) << "% ";
        std::cout << "(" << currentValue << "/" << maxValue << ")";
        
        std::cout.flush();
        
        // Add newline if completed
        if (currentValue >= maxValue) {
            std::cout << " - Complete!" << std::endl;
        }
    }
    
    // Reset the loading bar
    void reset() {
        currentValue = 0;
        display();
    }
    
    // Check if loading is complete
    bool isComplete() {
        return currentValue >= maxValue;
    }
    
    // Get current progress as percentage
    float getProgress() {
        return static_cast<float>(currentValue) / maxValue * 100;
    }
};



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
    const vector<MRGB>& pixels,
    vector<int>& pixels_indexs,
    vector<float>& pixels_distance,
    const RGB& new_pixel,
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
    Vector3 new_vec = {new_pixel.r,
                       new_pixel.g,
                       new_pixel.b};
    
    Vector3 diff = new_vec - centroid;
    Vector3 cross_prod = cross(diff, direction);
    float distance = norm(cross_prod);

    // Check threshold and update pixels if within
    if (distance <= threshold) {
        return {true, distance};
    }
    return {false, NULL};
}


// Main function to adjust the line and include new_pixel
pair<bool,pair<float, vector<float>>> can_be_on_line(
    const vector<MRGB>& pixels,
    vector<int>& pixels_indexs,
    vector<float>& pixels_distance,
    const int& new_pixel_index,
    const RGB& new_pixel,
    const float& threshold) {

    // Handle edge case: need at least one point to define a line with new_pixel
    if (pixels_indexs.empty()) {
        pixels_indexs.push_back(new_pixel_index);
        pixels_distance.push_back(0.0);
        return {true, {0.0, pixels_distance}};
    }

    // Collect all points including new_pixel
    vector<RGB> all_points;
    for (size_t i = 0; i < pixels_indexs.size(); i++) {
        all_points.push_back(pixels[pixels_indexs[i]].value);
    }
    
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
        Vector3 centered = {
            p.r - centroid.x,
            p.g - centroid.y,
            p.b - centroid.z
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
            all_points[i].r,
            all_points[i].g,
            all_points[i].b
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
        return {true, {new_pixel_distance, new_distances}};
    }

    return {false, {NULL, {}}};
}

vector<pair<int,int>> findLocations(vector<pair<int,int>>& locations, pair<int,int> current, int target) {
    int x0 = current.first;
    int y0 = current.second;
    int n = locations.size();

    // Step 1: Pair each location with its Chebyshev distance
    vector<pair<int, pair<int,int>>> dist_loc;
    for (int i = 0; i < n; i++) {
        int x = locations[i].first;
        int y = locations[i].second;
        int d = max(abs(x - x0), abs(y - y0));
        dist_loc.push_back({d, {x, y}});
    }

    // Step 2: Sort by distance
    sort(dist_loc.begin(), dist_loc.end());

    // Step 3: Collect locations until target is met or exceeded
    vector<pair<int,int>> result;
    int i = 0;
    while (i < n && result.size() < target) {
        int current_d = dist_loc[i].first;
        // Add all locations with this distance
        while (i < n && dist_loc[i].first == current_d) {
            result.push_back(dist_loc[i].second);
            i++;
        }
    }

    return result;
}

auto push_in_L (
    vector<vector<int>>& lines,
    const int& l_index,
    const int& l_item) {
    lines[l_index].push_back(l_item);
}

auto push_in_D (
    vector<vector<float>>& line_distance,
    const int& d_index,
    const float& d_item) {
    line_distance[d_index].push_back(d_item);
}

auto update_D (
    vector<vector<float>>& line_distance,
    const int& index_of_line,
    const vector<float>& item_vec
    ) {
    for (size_t i = 0; i < line_distance[index_of_line].size(); i++) {
        line_distance[index_of_line][i] = item_vec[i];
    }
}


auto search_on_the_line(
    const vector<MRGB>& pixels,
    vector<vector<int>>& lines,
    vector<vector<float>>& line_distance,
    const int& current_pixel, 
    const RGB current_pixel_value,
    const vector<pair<int,RGB>>& combined_pixels,
    const float& threshold,
    const bool calculate_from_current = false) {
    
    vector<pair<int, float>> distances_list;

    int end = min(static_cast<int>(lines.size()), 32);

    // first check if current_pixel is on the line. if it comes the collect the distance.
    if (!calculate_from_current) {
        for (int i = end - 1; i >= 0; --i) {
            auto is_on = is_on_line(pixels, lines[i], line_distance[i], current_pixel_value, threshold);
            
            if (is_on.first) {
                distances_list.push_back({i, is_on.second});
            }
        }
    }

    else {
        auto is_on = is_on_line(pixels, lines.back(), line_distance.back(), current_pixel_value, threshold);
        
        if (is_on.first) {
            distances_list.push_back({lines.size() - 1, is_on.second});
        }
    }
    
    if (distances_list.size() > 0) {
        
        pair<int, float> minimum_distance = distances_list[0];
        for (auto && iter_distance : distances_list) {
            if (iter_distance.second < minimum_distance.second) {
                minimum_distance = iter_distance;
            }
        }

        for (auto &&each_pixel : combined_pixels) {
            push_in_L(lines, minimum_distance.first, each_pixel.first);
            push_in_D(line_distance, {minimum_distance.first}, {minimum_distance.second});
        }
        
        return minimum_distance.first;
    }

    vector<pair<int, pair<float, vector<float>>>> second_dis_list;

    if (!calculate_from_current) {
        for (int i = end - 1; i >= 0; i--) {
            auto can_be = can_be_on_line(pixels, lines[i], line_distance[i], current_pixel,current_pixel_value, threshold);
            
            if (can_be.first) {
                second_dis_list.push_back({i, can_be.second});
            }
        }
    } 
    else {
        auto can_be = can_be_on_line(pixels, lines.back(), line_distance.back(), current_pixel,current_pixel_value, threshold);

        if (can_be.first) {
            second_dis_list.push_back({lines.size() - 1, can_be.second});
        }
    }

    if (second_dis_list.size() > 0) {
        auto minimum_distance = second_dis_list[0];
        
        for (auto && iter_distance : second_dis_list) {
            if (iter_distance.second.first < minimum_distance.second.first) {
                minimum_distance = iter_distance;
            }
        }
        update_D(line_distance, minimum_distance.first, minimum_distance.second.second);

        
        for (auto &&each_pixel : combined_pixels) {
            push_in_L(lines, minimum_distance.first, each_pixel.first);
            push_in_D(line_distance, minimum_distance.first, minimum_distance.second.first);
        }

        return minimum_distance.first;
    }
    return -1;
}


auto calculate_lines(const vector<MRGB>& pixels, const float threshold) {
    if (pixels.empty()) {
        return vector<vector<int>>{};
    }

    vector<vector<int>> lines;
    vector<vector<float>> lines_distances;

    lines.reserve(pixels.size() / 2);
    lines_distances.reserve(pixels.size() / 2);

    vector<int>* last_line = nullptr;
    vector<float>* last_line_distance = nullptr;

    // Process pixels
    int have_to_jump = 0;

    LoadingBar loader(pixels.size(), 50);

    for (size_t i = 0; i < pixels.size(); ++i) {

        i += have_to_jump;
        have_to_jump = 0;

        const int current_pixel = i;
        int search_result = -1;

        
        // First try to add to the last line if it exists
        if (last_line && !last_line->empty()) {

            const int make_them_single = 4;
            
            vector<pair<int, RGB>> have_to_be_single = {{current_pixel, pixels[current_pixel].value}};
            for (size_t iter = 0; iter < make_them_single; iter++) {
                if (iter >= pixels.size())break;
                const float
                dr = abs(pixels[have_to_be_single.back().first].value.r - pixels[current_pixel + iter].value.r),
                dg = abs(pixels[have_to_be_single.back().first].value.g - pixels[current_pixel + iter].value.g),
                db = abs(pixels[have_to_be_single.back().first].value.b - pixels[current_pixel + iter].value.b);

                if (dr > 5.0f ||
                    dg > 5.0f ||
                    db > 5.0f) {
                    break;
                }
                else {
                    have_to_jump++;
                    have_to_be_single.push_back({current_pixel + iter, pixels[current_pixel + iter].value});
                }
            }
            
            RGB ave_value = pixels[have_to_be_single[0].first].value;
            for (size_t iter = 1; iter < have_to_be_single.size(); iter++)
            {
                ave_value.r += pixels[have_to_be_single[iter].first].value.r;
                ave_value.g += pixels[have_to_be_single[iter].first].value.g;
                ave_value.b += pixels[have_to_be_single[iter].first].value.b;
            }
            ave_value.r = ave_value.r / have_to_be_single.size();
            ave_value.g = ave_value.g / have_to_be_single.size();
            ave_value.b = ave_value.b / have_to_be_single.size();
            
            // cout << have_to_be_single.size() << '\n';
            search_result = search_on_the_line(pixels, lines, lines_distances, current_pixel, pixels[current_pixel].value, have_to_be_single, threshold, true);
            
        } 

        // If not added to last line, search all lines
        if (search_result == -1 && !lines.empty()) {
            search_result = search_on_the_line(
                pixels,lines, lines_distances, current_pixel, pixels[current_pixel].value, {{current_pixel, pixels[current_pixel].value}},threshold, false);
            
            if (search_result != -1) {
                last_line = &lines[search_result];
                last_line_distance = &lines_distances[search_result];
            }
        }

        // If not added to any existing line, start a new line
        if (search_result == -1) {
            lines.push_back({current_pixel});
            lines_distances.push_back({0.0});

            last_line = &lines.back();
            last_line_distance = &lines_distances.back();

            // If possible, initialize new line with the next pixel
            if (i + 1 < pixels.size()) {
                lines.back().push_back(i + 1);
                lines_distances.back().push_back(0.0);
                ++i; // Skip the next pixel since we already added it
            }
        }
        loader.update(i);
    }
    return lines;
}

pair<RGB,RGB> find_end_points(const vector<MRGB>& pixels, const vector<int>& pixels_indexs) {
    if (pixels_indexs.size() == 2)
        return {pixels[pixels_indexs[0]].value, pixels[pixels_indexs[1]].value};
    
    if (pixels_indexs.size() < 2)
        return {{0,0,0}, {0,0,0}}; // Handle edge case
    
    // Step 1: Calculate the centroid (mean) of all RGB values
    float mean_r = 0, mean_g = 0, mean_b = 0;
    for (int idx : pixels_indexs) {
        mean_r += pixels[idx].value.r;
        mean_g += pixels[idx].value.g;
        mean_b += pixels[idx].value.b;
    }
    mean_r /= pixels_indexs.size();
    mean_g /= pixels_indexs.size();
    mean_b /= pixels_indexs.size();
    
    // Step 2: Build covariance matrix (3x3 for RGB)
    float cov[3][3] = {{0}};
    
    for (int idx : pixels_indexs) {
        float dr = pixels[idx].value.r - mean_r;
        float dg = pixels[idx].value.g - mean_g;
        float db = pixels[idx].value.b - mean_b;
        
        cov[0][0] += dr * dr;
        cov[0][1] += dr * dg;
        cov[0][2] += dr * db;
        cov[1][1] += dg * dg;
        cov[1][2] += dg * db;
        cov[2][2] += db * db;
    }
    
    // Symmetric matrix
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];
    
    // Step 3: Find the principal eigenvector (largest eigenvalue)
    // Using power iteration method for simplicity
    float eigenvec[3] = {1.0f, 1.0f, 1.0f}; // Initial guess
    
    for (int iter = 0; iter < 20; iter++) { // Power iteration
        float new_vec[3] = {0};
        
        // Matrix-vector multiplication: new_vec = cov * eigenvec
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                new_vec[i] += cov[i][j] * eigenvec[j];
            }
        }
        
        // Normalize the vector
        float norm = sqrt(new_vec[0]*new_vec[0] + new_vec[1]*new_vec[1] + new_vec[2]*new_vec[2]);
        if (norm < 1e-10) break; // Avoid division by zero
        
        eigenvec[0] = new_vec[0] / norm;
        eigenvec[1] = new_vec[1] / norm;
        eigenvec[2] = new_vec[2] / norm;
    }
    
    // Step 4: Project all points onto the principal component line
    float min_proj = FLT_MAX, max_proj = -FLT_MAX;
    
    for (int idx : pixels_indexs) {
        const float dr = pixels[idx].value.r - mean_r;
        float dg = pixels[idx].value.g - mean_g;
        float db = pixels[idx].value.b - mean_b;
        
        // Dot product with eigenvector gives projection onto the line
        float projection = dr * eigenvec[0] + dg * eigenvec[1] + db * eigenvec[2];
        
        min_proj = min(min_proj, projection);
        max_proj = max(max_proj, projection);
    }
    
    // Step 5: Calculate the endpoints on the line
    RGB endpoint1{}, endpoint2;
    
    endpoint1.r = static_cast<int>(mean_r + min_proj * eigenvec[0]);
    endpoint1.g = static_cast<int>(mean_g + min_proj * eigenvec[1]);
    endpoint1.b = static_cast<int>(mean_b + min_proj * eigenvec[2]);
    
    endpoint2.r = static_cast<int>(mean_r + max_proj * eigenvec[0]);
    endpoint2.g = static_cast<int>(mean_g + max_proj * eigenvec[1]);
    endpoint2.b = static_cast<int>(mean_b + max_proj * eigenvec[2]);
    
    // Clamp values to valid RGB range [0, 255]
    endpoint1.r = max(0.f, min(255.f, endpoint1.r));
    endpoint1.g = max(0.f, min(255.f, endpoint1.g));
    endpoint1.b = max(0.f, min(255.f, endpoint1.b));
    
    endpoint2.r = max(0.f, min(255.f, endpoint2.r));
    endpoint2.g = max(0.f, min(255.f, endpoint2.g));
    endpoint2.b = max(0.f, min(255.f, endpoint2.b));
    
    return {endpoint1, endpoint2};
}



// // Optimized 3D Vector structure with SIMD alignment
// struct alignas(16) Vector3 {
//     float x, y, z, w;  // w padding for SIMD alignment
    
//     Vector3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z), w(0.0f) {}
    
//     // SIMD operations
//     Vector3 operator+(const Vector3& other) const {
//         __m128 a = _mm_load_ps(&x);
//         __m128 b = _mm_load_ps(&other.x);
//         __m128 result = _mm_add_ps(a, b);
//         Vector3 ret;
//         _mm_store_ps(&ret.x, result);
//         return ret;
//     }
    
//     Vector3 operator-(const Vector3& other) const {
//         __m128 a = _mm_load_ps(&x);
//         __m128 b = _mm_load_ps(&other.x);
//         __m128 result = _mm_sub_ps(a, b);
//         Vector3 ret;
//         _mm_store_ps(&ret.x, result);
//         return ret;
//     }
    
//     Vector3 operator*(float scalar) const {
//         __m128 v = _mm_load_ps(&x);
//         __m128 s = _mm_set1_ps(scalar);
//         __m128 result = _mm_mul_ps(v, s);
//         Vector3 ret;
//         _mm_store_ps(&ret.x, result);
//         return ret;
//     }
// };

// // Fast SIMD dot product
// inline float dot_simd(const Vector3& a, const Vector3& b) {
//     __m128 va = _mm_load_ps(&a.x);
//     __m128 vb = _mm_load_ps(&b.x);
//     __m128 mul = _mm_mul_ps(va, vb);
    
//     // Horizontal add for first 3 components only
//     __m128 hadd1 = _mm_hadd_ps(mul, mul);
//     __m128 hadd2 = _mm_hadd_ps(hadd1, hadd1);
//     return _mm_cvtss_f32(hadd2) - a.w * b.w; // Subtract w component
// }

// // Fast cross product
// inline Vector3 cross_simd(const Vector3& a, const Vector3& b) {
//     return Vector3(
//         a.y * b.z - a.z * b.y,
//         a.z * b.x - a.x * b.z,
//         a.x * b.y - a.y * b.x
//     );
// }

// // Fast inverse square root (Quake III algorithm)
// inline float fast_inv_sqrt(float x) {
//     union { float f; uint32_t i; } conv = { .f = x };
//     conv.i = 0x5f3759df - (conv.i >> 1);
//     conv.f *= 1.5f - (x * 0.5f * conv.f * conv.f);
//     return conv.f;
// }

// // Fast normalize using inverse square root
// inline Vector3 normalize_fast(const Vector3& v) {
//     float dot_val = v.x * v.x + v.y * v.y + v.z * v.z;
//     if (dot_val < 1e-8f) return Vector3(1.0f, 0.0f, 0.0f);
    
//     float inv_norm = fast_inv_sqrt(dot_val);
//     return Vector3(v.x * inv_norm, v.y * inv_norm, v.z * inv_norm);
// }

// // Optimized matrix-vector multiplication with loop unrolling
// inline Vector3 matrix_multiply_fast(const float mat[3][3], const Vector3& v) {
//     return Vector3(
//         mat[0][0] * v.x + mat[0][1] * v.y + mat[0][2] * v.z,
//         mat[1][0] * v.x + mat[1][1] * v.y + mat[1][2] * v.z,
//         mat[2][0] * v.x + mat[2][1] * v.y + mat[2][2] * v.z
//     );
// }

// // Highly optimized power iteration with early convergence
// Vector3 power_iteration_fast(const float mat[3][3], const int max_iter = 10) {
//     Vector3 v(1.0f, 0.0f, 0.0f);
//     const float tol_sq = 1e-12f; // Square of tolerance for faster comparison
    
//     for (int i = 0; i < max_iter; ++i) {
//         Vector3 w = matrix_multiply_fast(mat, v);
//         float norm_sq = w.x * w.x + w.y * w.y + w.z * w.z;
        
//         if (norm_sq < 1e-16f) {
//             return Vector3(1.0f, 0.0f, 0.0f);
//         }
        
//         float inv_norm = fast_inv_sqrt(norm_sq);
//         Vector3 v_new(w.x * inv_norm, w.y * inv_norm, w.z * inv_norm);
        
//         // Check convergence using squared difference
//         Vector3 diff = v_new - v;
//         float diff_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        
//         if (diff_sq < tol_sq) break;
//         v = v_new;
//     }
//     return v;
// }

// // Cache for storing computed line parameters
// struct LineCache {
//     Vector3 centroid;
//     Vector3 direction;
//     float cov[3][3];
//     bool valid = false;
//     size_t hash = 0;
// };

// // Simple hash function for pixel indices
// inline size_t hash_indices(const vector<int>& indices) {
//     size_t hash = 0;
//     for (int idx : indices) {
//         hash ^= std::hash<int>{}(idx) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
//     }
//     return hash;
// }

// // Optimized function to compute line parameters with caching
// pair<Vector3, Vector3> compute_line_params_cached(
//     const vector<MRGB>& pixels,
//     const vector<int>& pixel_indices,
//     LineCache& cache) {
    
//     size_t current_hash = hash_indices(pixel_indices);
//     if (cache.valid && cache.hash == current_hash) {
//         return {cache.centroid, cache.direction};
//     }
    
//     const int n = pixel_indices.size();
//     const float inv_n = 1.0f / n;
    
//     // Compute centroid with loop unrolling
//     Vector3 centroid(0.0f, 0.0f, 0.0f);
//     int i = 0;
    
//     // Process 4 pixels at a time
//     for (; i + 3 < n; i += 4) {
//         const auto& p1 = pixels[pixel_indices[i]].value;
//         const auto& p2 = pixels[pixel_indices[i+1]].value;
//         const auto& p3 = pixels[pixel_indices[i+2]].value;
//         const auto& p4 = pixels[pixel_indices[i+3]].value;
        
//         centroid.x += p1.r + p2.r + p3.r + p4.r;
//         centroid.y += p1.g + p2.g + p3.g + p4.g;
//         centroid.z += p1.b + p2.b + p3.b + p4.b;
//     }
    
//     // Handle remaining pixels
//     for (; i < n; ++i) {
//         const auto& p = pixels[pixel_indices[i]].value;
//         centroid.x += p.r;
//         centroid.y += p.g;
//         centroid.z += p.b;
//     }
    
//     centroid = centroid * inv_n;
    
//     // Compute covariance matrix with optimized memory access
//     float cov[3][3] = {{0.0f}};
    
//     for (int idx : pixel_indices) {
//         const auto& p = pixels[idx].value;
//         float dx = p.r - centroid.x;
//         float dy = p.g - centroid.y;
//         float dz = p.b - centroid.z;
        
//         // Symmetric matrix - compute upper triangle only
//         cov[0][0] += dx * dx;
//         cov[0][1] += dx * dy;
//         cov[0][2] += dx * dz;
//         cov[1][1] += dy * dy;
//         cov[1][2] += dy * dz;
//         cov[2][2] += dz * dz;
//     }
    
//     // Fill symmetric elements
//     cov[1][0] = cov[0][1];
//     cov[2][0] = cov[0][2];
//     cov[2][1] = cov[1][2];
    
//     Vector3 direction = power_iteration_fast(cov);
    
//     // Update cache
//     cache.centroid = centroid;
//     cache.direction = direction;
//     cache.hash = current_hash;
//     cache.valid = true;
//     memcpy(cache.cov, cov, sizeof(cov));
    
//     return {centroid, direction};
// }

// // Highly optimized is_on_line function
// pair<bool, float> is_on_line_optimized(
//     const vector<MRGB>& pixels,
//     vector<int>& pixels_indices,
//     vector<float>& pixels_distance,
//     const int new_pixel_index,
//     const float threshold,
//     LineCache& cache) {
    
//     if (pixels_indices.size() < 2) {
//         return {false, 0.0f};
//     }
    
//     auto [centroid, direction] = compute_line_params_cached(pixels, pixels_indices, cache);
    
//     // Compute distance from new pixel to line
//     RGB new_pixel = pixels[new_pixel_index].value;
  
//     Vector3 new_vec(new_pixel.r, new_pixel.g, new_pixel.b);
//     Vector3 diff = new_vec - centroid;
//     Vector3 cross_prod = cross_simd(diff, direction);
    
//     float distance = sqrtf(cross_prod.x * cross_prod.x + 
//                           cross_prod.y * cross_prod.y + 
//                           cross_prod.z * cross_prod.z);
    
//     return {distance <= threshold, distance};
// }

// // Optimized can_be_on_line function
// pair<bool, pair<float, vector<float>>> can_be_on_line_optimized(
//     const vector<MRGB>& pixels,
//     vector<int>& pixels_indices,
//     vector<float>& pixels_distance,
//     const int new_pixel_index,
//     const float threshold) {
    
//     if (pixels_indices.empty()) {
//         return {true, {0.0f, {}}};
//     }
    
//     // Create temporary vector with all points
//     vector<int> all_points;
//     all_points.reserve(pixels_indices.size() + 1);
//     all_points.insert(all_points.end(), pixels_indices.begin(), pixels_indices.end());
//     all_points.push_back(new_pixel_index);
    
//     const int n = all_points.size();
//     const float inv_n = 1.0f / n;
    
//     // Compute centroid
//     Vector3 centroid(0.0f, 0.0f, 0.0f);
//     for (int idx : all_points) {
//         const auto& p = pixels[idx].value;
//         centroid.x += p.r;
//         centroid.y += p.g;
//         centroid.z += p.b;
//     }
//     centroid = centroid * inv_n;
    
//     // Compute covariance matrix
//     float cov[3][3] = {{0.0f}};
//     for (int idx : all_points) {
//         const auto& p = pixels[idx].value;
//         float dx = p.r - centroid.x;
//         float dy = p.g - centroid.y;
//         float dz = p.b - centroid.z;
        
//         cov[0][0] += dx * dx;
//         cov[0][1] += dx * dy;
//         cov[0][2] += dx * dz;
//         cov[1][1] += dy * dy;
//         cov[1][2] += dy * dz;
//         cov[2][2] += dz * dz;
//     }
    
//     cov[1][0] = cov[0][1];
//     cov[2][0] = cov[0][2];
//     cov[2][1] = cov[1][2];
    
//     Vector3 direction = power_iteration_fast(cov);
    
//     // Check all distances with early exit
//     vector<float> new_distances;
//     new_distances.reserve(n - 1);
//     float new_pixel_distance = 0.0f;
//     const float threshold_sq = threshold * threshold;
    
//     for (int i = 0; i < n; ++i) {
//         const auto& p = pixels[all_points[i]].value;
//         Vector3 point(p.r, p.g, p.b);
//         Vector3 diff = point - centroid;
//         Vector3 cross_prod = cross_simd(diff, direction);
        
//         float distance_sq = cross_prod.x * cross_prod.x + 
//                            cross_prod.y * cross_prod.y + 
//                            cross_prod.z * cross_prod.z;
        
//         if (distance_sq > threshold_sq) {
//             return {false, {0.0f, {}}};
//         }
        
//         float distance = sqrtf(distance_sq);
//         if (i < n - 1) {
//             new_distances.push_back(distance);
//         } else {
//             new_pixel_distance = distance;
//         }
//     }
    
//     return {true, {new_pixel_distance, std::move(new_distances)}};
// }

// // Thread-local cache for line computations
// thread_local LineCache g_line_cache;

// // Wrapper functions maintaining original interface
// pair<bool, float> is_on_line(
//     const vector<MRGB>& pixels,
//     vector<int>& pixels_indices,
//     vector<float>& pixels_distance,
//     const int new_pixel_index,
//     const float& threshold
//     ) {
    
//     return is_on_line_optimized(pixels, pixels_indices, pixels_distance, 
//                                new_pixel_index, threshold, g_line_cache);
// }

// pair<bool, pair<float, vector<float>>> can_be_on_line(
//     const vector<MRGB>& pixels,
//     vector<int>& pixels_indices,
//     vector<float>& pixels_distance,
//     const int& new_pixel_index,
//     const float& threshold) {
    
//     return can_be_on_line_optimized(pixels, pixels_indices, pixels_distance, 
//                                    new_pixel_index, threshold);
// }


// // Pre-allocated memory pools to avoid dynamic allocations
// class MemoryPool {
// private:
//     static constexpr size_t POOL_SIZE = 10000;
//     vector<pair<int, float>> distance_pool;
//     vector<pair<int, pair<float, vector<float>>>> second_pool;
//     size_t distance_pool_index = 0;
//     size_t second_pool_index = 0;
    
// public:
//     MemoryPool() {
//         distance_pool.reserve(POOL_SIZE);
//         second_pool.reserve(POOL_SIZE);
        
//         // Pre-allocate objects
//         for (size_t i = 0; i < POOL_SIZE; ++i) {
//             distance_pool.emplace_back();
//             second_pool.emplace_back();
//             second_pool[i].second.second.reserve(32); // Reserve space for distances
//         }
//     }
    
//     vector<pair<int, float>>& get_distance_list() {
//         distance_pool_index = 0;
//         return distance_pool;
//     }
    
//     vector<pair<int, pair<float, vector<float>>>>& get_second_list() {
//         second_pool_index = 0;
//         return second_pool;
//     }
    
//     void add_distance(vector<pair<int, float>>& list, int line_idx, float dist) {
//         if (distance_pool_index < POOL_SIZE) {
//             list[distance_pool_index++] = {line_idx, dist};
//         }
//     }
    
//     void add_second(vector<pair<int, pair<float, vector<float>>>>& list, 
//                    int line_idx, float dist, const vector<float>& distances) {
//         if (second_pool_index < POOL_SIZE) {
//             auto& item = list[second_pool_index++];
//             item.first = line_idx;
//             item.second.first = dist;
//             item.second.second.clear();
//             item.second.second.insert(item.second.second.end(), distances.begin(), distances.end());
//         }
//     }
    
//     size_t distance_size() const { return distance_pool_index; }
//     size_t second_size() const { return second_pool_index; }
// };

// // Thread-local memory pool
// thread_local MemoryPool g_memory_pool;

// // Cache for line fitness scores to avoid recomputation
// struct LineFitnessCache {
//     unordered_map<size_t, float> fitness_cache;
    
//     float get_fitness(const vector<int>& line_indices, const vector<float>& distances) {
//         size_t hash = 0;
//         for (int idx : line_indices) {
//             hash ^= std::hash<int>{}(idx) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
//         }
        
//         auto it = fitness_cache.find(hash);
//         if (it != fitness_cache.end()) {
//             return it->second;
//         }
        
//         // Compute fitness: prioritize lines with more points and lower average distance
//         float avg_distance = 0.0f;
//         if (!distances.empty()) {
//             for (float d : distances) {
//                 avg_distance += d;
//             }
//             avg_distance /= distances.size();
//         }
        
//         float fitness = static_cast<float>(line_indices.size()) / (1.0f + avg_distance);
//         fitness_cache[hash] = fitness;
//         return fitness;
//     }
    
//     void clear() {
//         fitness_cache.clear();
//     }
// };

// thread_local LineFitnessCache g_fitness_cache;

// // Optimized search function with early termination and memory pooling
// inline int search_on_the_line_optimized(
//     const vector<MRGB>& pixels,
//     vector<vector<int>>& lines,
//     vector<vector<float>>& line_distance,
//     const int current_pixel,
//     const float threshold,
//     const bool calculate_from_current = false) {
    
//     // Early exit for empty lines
//     if (lines.empty()) return -1;
    
//     const int end = calculate_from_current ? 1 : min(static_cast<int>(lines.size()), 32);
//     const int start_idx = calculate_from_current ? lines.size() - 1 : end - 1;
    
//     // Phase 1: Check if pixel is already on existing lines
//     int best_line_idx = -1;
//     float best_distance = threshold + 1.0f;
    
//     // Use single pass to find best line instead of collecting all candidates
//     for (int i = start_idx; i >= (calculate_from_current ? start_idx : 0); --i) {
//         // Skip lines that are too small
//         if (lines[i].size() < 2) continue;
        
//         auto is_on = is_on_line(pixels, lines[i], line_distance[i], current_pixel, threshold);
        
//         if (is_on.first && is_on.second < best_distance) {
//             best_distance = is_on.second;
//             best_line_idx = i;
            
//             // Early termination if we find a very close match
//             if (best_distance < threshold * 0.1f) break;
//         }
//     }
    
//     if (best_line_idx != -1) {
//         lines[best_line_idx].push_back(current_pixel);
//         line_distance[best_line_idx].push_back(best_distance);
//         return best_line_idx;
//     }
    
//     // Phase 2: Check if pixel can be added to recomputed lines
//     best_line_idx = -1;
//     best_distance = threshold + 1.0f;
//     vector<float> best_distances;
    
//     for (int i = start_idx; i >= (calculate_from_current ? start_idx : 0); --i) {
//         // Skip if line is empty
//         if (lines[i].empty()) continue;
        
//         auto can_be = can_be_on_line(pixels, lines[i], line_distance[i], current_pixel, threshold);
        
//         if (can_be.first) {
//             float fitness = g_fitness_cache.get_fitness(lines[i], line_distance[i]);
//             float adjusted_distance = can_be.second.first / (1.0f + fitness * 0.1f);
            
//             if (adjusted_distance < best_distance) {
//                 best_distance = adjusted_distance;
//                 best_line_idx = i;
//                 best_distances = std::move(can_be.second.second);
                
//                 // Early termination for very good fits
//                 if (adjusted_distance < threshold * 0.2f) break;
//             }
//         }
//     }
    
//     if (best_line_idx != -1) {
//         lines[best_line_idx].push_back(current_pixel);
        
//         // Update distances efficiently
//         for (size_t j = 0; j < line_distance[best_line_idx].size() && j < best_distances.size(); ++j) {
//             line_distance[best_line_idx][j] = best_distances[j];
//         }
//         line_distance[best_line_idx].push_back(best_distance);
        
//         return best_line_idx;
//     }
    
//     return -1;
// }

// // Highly optimized calculate_lines with multiple improvements
// auto calculate_lines_optimized(const vector<MRGB>& pixels, const float threshold) {
//     if (pixels.empty()) {
//         return vector<vector<int>>{};
//     }
    
//     const size_t pixel_count = pixels.size();
//     vector<vector<int>> lines;
//     vector<vector<float>> lines_distances;
    
//     // Pre-allocate based on heuristics
//     const size_t estimated_lines = max(pixel_count / 10, size_t(1));
//     lines.reserve(estimated_lines);
//     lines_distances.reserve(estimated_lines);
    
//     // Cache for last used line to improve locality
//     int last_line_idx = -1;
    
//     // Batch processing parameters
//     constexpr size_t BATCH_SIZE = 8;
//     vector<int> batch_pixels;
//     batch_pixels.reserve(BATCH_SIZE);
    
//     // Process pixels in batches for better cache performance
//     for (size_t batch_start = 0; batch_start < pixel_count; batch_start += BATCH_SIZE) {
//         size_t batch_end = min(batch_start + BATCH_SIZE, pixel_count);
        
//         // Fill batch
//         batch_pixels.clear();
//         for (size_t i = batch_start; i < batch_end; ++i) {
//             batch_pixels.push_back(static_cast<int>(i));
//         }
        
//         // Process batch
//         for (int current_pixel : batch_pixels) {
//             int search_result = -1;
            
//             // Try last used line first (locality optimization)
//             if (last_line_idx != -1 && last_line_idx < static_cast<int>(lines.size()) && 
//                 !lines[last_line_idx].empty()) {
                
//                 auto is_on = is_on_line(pixels, lines[last_line_idx], 
//                                        lines_distances[last_line_idx], current_pixel, threshold);
//                 if (is_on.first) {
//                     lines[last_line_idx].push_back(current_pixel);
//                     lines_distances[last_line_idx].push_back(is_on.second);
//                     search_result = last_line_idx;
//                 }
//             }
            
//             // If not added to last line, search from most recent lines
//             if (search_result == -1 && !lines.empty()) {
//                 search_result = search_on_the_line_optimized(pixels, lines, lines_distances, 
//                                                            current_pixel, threshold, false);
//                 if (search_result != -1) {
//                     last_line_idx = search_result;
//                 }
//             }
            
//             // Create new line if needed
//             if (search_result == -1) {
//                 lines.emplace_back();
//                 lines_distances.emplace_back();
                
//                 auto& new_line = lines.back();
//                 auto& new_distances = lines_distances.back();
                
//                 new_line.reserve(16); // Reserve space for typical line size
//                 new_distances.reserve(16);
                
//                 new_line.push_back(current_pixel);
//                 new_distances.push_back(0.0f);
                
//                 last_line_idx = static_cast<int>(lines.size()) - 1;
                
//                 // Try to initialize with next pixel if available and in same batch
//                 auto next_it = find(batch_pixels.begin(), batch_pixels.end(), current_pixel);
//                 if (next_it != batch_pixels.end() && next_it + 1 != batch_pixels.end()) {
//                     int next_pixel = *(next_it + 1);
                    
//                     // Simple distance check for initialization
//                     const auto& curr_rgb = pixels[current_pixel].value;
//                     const auto& next_rgb = pixels[next_pixel].value;
                    
//                     float color_dist = sqrtf(
//                         (curr_rgb.r - next_rgb.r) * (curr_rgb.r - next_rgb.r) +
//                         (curr_rgb.g - next_rgb.g) * (curr_rgb.g - next_rgb.g) +
//                         (curr_rgb.b - next_rgb.b) * (curr_rgb.b - next_rgb.b)
//                     );
                    
//                     if (color_dist <= threshold * 2.0f) { // More lenient for initialization
//                         new_line.push_back(next_pixel);
//                         new_distances.push_back(0.0f);
                        
//                         // Remove next_pixel from batch to avoid double processing
//                         batch_pixels.erase(next_it + 1);
//                     }
//                 }
//             }
//         }
//     }
    
//     // Post-processing: merge similar lines and remove small lines
//     if (lines.size() > 1) {
//         // Remove lines with only one pixel
//         auto it = lines.begin();
//         auto dist_it = lines_distances.begin();
        
//         while (it != lines.end()) {
//             if (it->size() <= 1) {
//                 it = lines.erase(it);
//                 dist_it = lines_distances.erase(dist_it);
//             } else {
//                 ++it;
//                 ++dist_it;
//             }
//         }
        
//         // Merge very similar lines (optional optimization)
//         const float merge_threshold = threshold * 0.5f;
//         for (size_t i = 0; i < lines.size(); ++i) {
//             for (size_t j = i + 1; j < lines.size(); ++j) {
//                 if (lines[i].size() < 3 || lines[j].size() < 3) continue;
                
//                 // Check if lines can be merged by testing a few points
//                 bool can_merge = true;
//                 int test_count = min(static_cast<int>(lines[j].size()), 3);
                
//                 for (int k = 0; k < test_count; ++k) {
//                     auto test_result = is_on_line(pixels, lines[i], lines_distances[i], 
//                                                  lines[j][k], merge_threshold);
//                     if (!test_result.first) {
//                         can_merge = false;
//                         break;
//                     }
//                 }
                
//                 if (can_merge) {
//                     // Merge line j into line i
//                     lines[i].insert(lines[i].end(), lines[j].begin(), lines[j].end());
//                     lines_distances[i].insert(lines_distances[i].end(), 
//                                             lines_distances[j].begin(), lines_distances[j].end());
                    
//                     // Remove line j
//                     lines.erase(lines.begin() + j);
//                     lines_distances.erase(lines_distances.begin() + j);
//                     --j;
//                 }
//             }
//         }
//     }
    
//     // Clear fitness cache for next run
//     g_fitness_cache.clear();
    
//     return lines;
// }


// auto calculate_lines(const vector<MRGB>& pixels, const float threshold) {
//     return calculate_lines_optimized(pixels, threshold);
// }

#endif
