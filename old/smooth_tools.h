#ifndef SMOOTH_TOOLS_H
#define SMOOTH_TOOLS_H

#include <cstddef>
#include <iostream>
#include <vector>
#include "basic_setup_tools.h"
#include <cmath>
using namespace std;


inline auto interpolate_values(const vector<rgb> values,vector<float> weights){

    if (values.size() != weights.size()) {
        cout << "true";
        return RGB{0,0,0};
    }

    float total_weight = 0;
    for (size_t i = 0; i < weights.size(); i++) {
        total_weight += weights[i];
    }
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] /= total_weight;
    }

    float r = static_cast<float>(values[0].r) * weights[0];
    float g = static_cast<float>(values[0].g) * weights[0];
    float b = static_cast<float>(values[0].b) * weights[0];
    
    for (size_t i = 1; i < values.size(); i++) {
        r += static_cast<float>(values[i].r) * weights[i];
        g += static_cast<float>(values[i].g) * weights[i];
        b += static_cast<float>(values[i].b) * weights[i];
    }
    
    return RGB{r,g,b};
}

auto give_abs(float target,float value) {
    return abs(target - value);
}
auto cal(float limit,float dx,float dy){
    float result = ((limit - sqrt(dx * dx + dy * dy)) / limit) * 100;
    if(result < 0.0f)result = 0.0f;
    return result;
}

auto give_1D_distance(float range, float check_point, float value){
    float result = (give_abs(check_point, value) / range) * 100;
    return result > 100 ? 0 : (100 - result);
}

RGB indexed(RGB START,RGB END,float limit,float index){
    float t = (index) / (limit);
    return {
        START.r + t * (END.r - START.r),
        START.g + t * (END.g - START.g),
        START.b + t * (END.b - START.b)
    };
}

int msd(float p,int d){
    return abs(p - d);
}

float ms(const float& m,const float& t){
    return m - t < 0 ? 0 : m - t;
}

float give_distance(const pair<float,float> p,const pair<int,int> d,const float& limit,const float& threshold){
    return ms(cal(limit, msd(p.first, d.first), msd(p.second, d.second)), threshold);
}



auto seperate_rgb(vector<pair<int, rgb>> p){
    vector<rgb> c;
    for (auto &&i : p)c.push_back(i.second);
    return c;
}

auto generate_list(pair<int, int> set){
    vector<pair<int,int>> position_list;

    for (size_t i = 0; i < set.second; i++)
    position_list.push_back({0, i});
    
    for (size_t i = 0; i < set.second; i++)
    position_list.push_back({set.second - 1, i});

    for (size_t i = 0; i < set.first; i++)
    position_list.push_back({i, 0});

    for (size_t i = 0; i < set.first; i++)
    position_list.push_back({i, set.first});
    
    return position_list;
}

void mix_sides_rgb(std::vector<std::pair<int, rgb>> &top, std::pair<int, int> &set);

inline auto smooth(vector<vector<rgb>> &pixels, const vector<BlockAverage> &block_indexes)
{
    vector<pair<int,int>> set_type = {{16, 16}, {8, 8}, {4, 4}, {2, 2}, {1, 2}, {2, 1}, {1, 1}};

    vector<vector<rgb>> result_pixels = pixels;

    for (size_t i = 0; i < block_indexes.size(); i++) {
        if(block_indexes[i].average.is_setted >= 4 || block_indexes[i].average.is_setted == -1) continue;
        pair<int,int> set = set_type[block_indexes[i].average.is_setted];

        vector<pair<int,rgb>> top,down,left,right;

        int set_is = block_indexes[i].average.is_setted;

        auto get_edges = [&](vector<pair<int,rgb>>& vec,const int startX,const int startY,const int dx,const int dy,const int count) {
            for (int i = 0; i < count; i++) {
                int x = startX + dx * i, y = startY + dy * i;
                if (x < 0 || y < 0 || x >= pixels.size() || y >= pixels[0].size())
                    break;
                if(pixels[x][y].set_type_position != -1)
                vec.push_back({pixels[x][y].set_type_position,pixels[x][y]});
            }
        };
        // Function to get corner values (top-left, top-right, bottom-left, bottom-right)
        auto get_corners = [&](int startX, int startY, int width, int height) {
            pair<int, rgb> top_left = {-1, {0, 0, 0}};
            pair<int, rgb> top_right = {-1, {0, 0, 0}};
            pair<int, rgb> bottom_left = {-1, {0, 0, 0}};
            pair<int, rgb> bottom_right = {-1, {0, 0, 0}};
            
            // Top-left corner
            if (startX - 1 >= 0 && startY - 1 >= 0) {
                if (pixels[startX - 1][startY - 1].set_type_position != -1)
                top_left = {pixels[startX - 1][startY - 1].set_type_position, pixels[startX - 1][startY - 1]};
            }
            
            // Top-right corner
            if (startX - 1 >= 0 && startY + height < pixels[0].size()) {
                if (pixels[startX - 1][startY + height].set_type_position != -1)
                top_right = {pixels[startX - 1][startY + height].set_type_position, pixels[startX - 1][startY + height]};
            }
            
            // Bottom-left corner
            if (startX + width < pixels.size() && startY - 1 >= 0) {
                if (pixels[startX + width][startY - 1].set_type_position != -1)
                bottom_left = {pixels[startX + width][startY - 1].set_type_position, pixels[startX + width][startY - 1]};
            }
            
            // Bottom-right corner
            if (startX + width < pixels.size() && startY + height < pixels[0].size()) {
                if (pixels[startX + width][startY + height].set_type_position != -1)
                bottom_right = {pixels[startX + width][startY + height].set_type_position, pixels[startX + width][startY + height]};
            }
            
            return make_tuple(top_left, top_right, bottom_left, bottom_right);
        };
        
        auto is_same_block = [](const pair<int,rgb>& p,int set_is) {
            return p.first == set_is;
        };
        
        auto mix_pixels = [&](const vector<pair<int, rgb>>& side1, const vector<pair<int, rgb>>& side2, 
                                     const rgb& corner_value, pair<int,int> d, pair<int,int> c, int X, int Y) {
            double feature = 0.5f; // Default value, can be adjusted between 0 and 1
            
            float main_weight = (set.first - c.first) * (set.second - c.second) / static_cast<float>(set.first * set.second);
            float top_weight = c.first * (set.second - c.second) / static_cast<float>(set.first * set.second);
            float left_weight = (set.first - c.first) * c.second / static_cast<float>(set.first * set.second);
            float corner_weight = c.first * c.second / static_cast<float>(set.first * set.second);

            // Adjust weights based on feature value
            main_weight *= feature;
            top_weight *= (1.0 - feature);
            left_weight *= (1.0 - feature);
            corner_weight *= (1.0 - feature);

            return crgb(interpolate_values({result_pixels[X][Y],side1[d.second].second,side2[d.first].second,corner_value},
                                           {main_weight, top_weight, left_weight, corner_weight}));
            
        };

        const float far_percent = 0.45;

        auto interpolate_edge = [&](const vector<pair<int, rgb>>& edge, int edge_size, float position, int index, bool is_vertical, int X, int Y) {
            if (edge.size() == edge_size) {
                float percent = position / (static_cast<float>(edge_size) / 2);
                RGB middle_value = interpolate_values({edge[index].second, result_pixels[X][Y]}, {1 - far_percent,far_percent});
                result_pixels[X][Y] = crgb(interpolate_values({crgb(middle_value), result_pixels[X][Y]}, {1 - percent, percent}));
            }
        };

        
        const int X_location = block_indexes[i].groupX * set.first;
        const int Y_location = block_indexes[i].groupY * set.second;

        auto [top_left, top_right, bottom_left, bottom_right] = get_corners(X_location, Y_location, set.first, set.second);
        vector<pair<int, rgb>> corner_values_types = {top_left, top_right, bottom_left, bottom_right};
 
        get_edges(top, X_location - 1, Y_location, 0, 1, set.second);
        get_edges(right, X_location, Y_location + set.second, 1, 0, set.first);
        get_edges(down, X_location + set.first, Y_location, 0, 1, set.second);
        get_edges(left, X_location, Y_location - 1, 1, 0, set.first);

        // mix_sides_rgb(top, set);
        // mix_sides_rgb(down, set);
        // mix_sides_rgb(left, set);
        // mix_sides_rgb(right, set);
        
        for (size_t dx = 0; dx < set.first; dx++)
        for (size_t dy = 0; dy < set.second; dy++)
        {
            const int X = block_indexes[i].groupX * set.first + dx;
            const int Y = block_indexes[i].groupY * set.second + dy;
            
            const bool is_top = dx + 1 <= set.first / 2;
            const bool is_left = dy + 1 <= set.second / 2;

            const auto side1 = is_top ? top : down;
            const auto side2 = is_left ? left : right;
            
            const int cx = give_abs((set.first - 1) / 2, dx);
            const int cy = give_abs((set.second - 1) / 2, dy);

            float left_position = static_cast<float>(dy + 1);
            float right_position = static_cast<float>(set.second - dy);
            float top_position = static_cast<float>(dx + 1);
            float down_position = static_cast<float>(set.first - dx);
            
            int corner_index;

            if (is_top && is_left)corner_index = 0;
            if (is_top && !is_left)corner_index = 1;
            if (!is_top && is_left)corner_index = 2;
            if (!is_top && !is_left)corner_index = 3;

            if (side1.size() == set.first && side2.size() == set.second) {
                result_pixels[X][Y] = mix_pixels(side1, side2, corner_values_types[corner_index].second, {dx, dy}, {cx, cy}, X, Y);
            }
            else if (is_top && is_left) {
                interpolate_edge(top, set.first, top_position, dy, false, X, Y);
                interpolate_edge(left, set.second, left_position, dx, true, X, Y);
            }
            else if (is_top && !is_left) {
                interpolate_edge(top, set.first, top_position, dy, false, X, Y);
                interpolate_edge(right, set.second, right_position, dx, true, X, Y);
            }
            else if (!is_top && !is_left) {
                interpolate_edge(down, set.first, down_position, dy, false, X, Y);
                interpolate_edge(right, set.second, right_position, dx, true, X, Y);
            }
            else if (!is_top && is_left) {
                interpolate_edge(down, set.first, down_position, dy, false, X, Y);
                interpolate_edge(left, set.second, left_position, dx, true, X, Y);
            }
            // else {
            //     interpolate_edge(side1, set.first, is_top ? dx + 1 : set.first - dx, dy, false, X, Y);
            //     interpolate_edge(side2, set.second, is_left ? dy + 1 : set.second - dy, dx, true, X, Y);
            // }
        }
    }
    pixels = result_pixels;
}

inline void mix_sides_rgb(std::vector<std::pair<int, rgb>> &top, std::pair<int, int> &set) {
    vector<rgb> top_rgb;

    for (size_t a = 0; a < top.size(); a++)
    {
        vector<float> percent_list;
        for (size_t b = 0; b < top.size(); b++)
        {
            percent_list.push_back(give_1D_distance(set.second / 2, a, b));
        }
        top_rgb.push_back(crgb(interpolate_values(seperate_rgb(top), percent_list)));
    }
    for (size_t i = 0; i < top.size(); i++)
    {
        top[i].second = top_rgb[i];
        printrgb(top_rgb[i]);
    }
    cout << ' ';
}

#endif
