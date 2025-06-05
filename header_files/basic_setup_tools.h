#ifndef BASIC_SETUP_TOOLS_H
#define BASIC_SETUP_TOOLS_H


#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <math.h>
using namespace std;

struct RGB {
    float r;
    float g;
    float b;
};

struct rgb {
    int r;
    int g;
    int b;
};

struct MRGB;

struct CRGB {
    RGB value;
    int main_RGB_index = -1;
};

struct MRGB {
    RGB value = {0,0,0};
    pair<int, int> coordinate = {0, 0};
    vector<pair<int,int>> connected_RGBs;
};


// Global variable to store the start time
static std::chrono::time_point<std::chrono::high_resolution_clock> timer_start;

// Function to start the timer
inline void startTimer() {
    timer_start = std::chrono::high_resolution_clock::now();
}

// Function to stop the timer and return time in milliseconds
inline float stopTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - timer_start;
    return duration.count();
}
inline void printrgb(rgb color) {
    const unsigned char c = 220;
    
    std::cout << "\033[38;2;" << color.r << ";" << color.g << ";" << color.b << "m"
              << c
              << "\033[0m";
}
inline void printRGB(RGB color) {
    const unsigned char c = 220;
    
    std::cout << "\033[38;2;" << static_cast<int>(color.r) << ";" << static_cast<int>(color.g) << ";" << static_cast<int>(color.b) << "m"
              << c
              << "\033[0m";
}


inline RGB indexed_value(RGB START, RGB END, int limit, int INDEX) {
    if (limit <= 1) return START;
    if (INDEX <= 1) return START;
    if (INDEX >= limit) return END;
    float t = static_cast<float>(INDEX - 1) / (limit - 1);
    return {
        START.r + t * (END.r - START.r),
        START.g + t * (END.g - START.g),
        START.b + t * (END.b - START.b)
    };
}

inline int nearest_index(RGB value_to_match, RGB START, RGB END, int limit) {
    int result = 1;
    float min_distance = std::numeric_limits<float>::max();
    for (int i = 1; i <= limit; i++) {
        RGB n = indexed_value(START, END, limit, i);
        float dr = n.r - value_to_match.r;
        float dg = n.g - value_to_match.g;
        float db = n.b - value_to_match.b;
        float distance = dr * dr + dg * dg + db * db; // Euclidean distance squared
        if (distance < min_distance) {
            min_distance = distance;
            result = i;
        }
    }
    return result;
}


inline RGB average(RGB v1, RGB v2) {
    return {(v1.r + v2.r)/2, (v1.g + v2.g)/2, (v1.b + v2.b)/2};
}

inline RGB average_clamped(RGB v1, RGB v2, float weight) {
    // Clamp weight to [0.0, 1.0] range
    if (weight < 0.0) weight = 0.0f;
    if (weight > 1.0) weight = 1.0f;
    
    return {
        (v1.r * weight + v2.r * (1.0f - weight)),
        (v1.g * weight + v2.g * (1.0f - weight)),
        (v1.b * weight + v2.b * (1.0f - weight))
    };
}
inline rgb to_rgb(const RGB& value) {
    return {static_cast<int>(value.r), static_cast<int>(value.g), static_cast<int>(value.b)};
}

inline RGB to_RGB(const rgb& value) {
    return {static_cast<float>(value.r), static_cast<float>(value.g), static_cast<float>(value.b)};
}

#endif
