#ifndef IMAGE_TOOLS_H
#define IMAGE_TOOLS_H
#include <ios>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "basic_setup_tools.h"

using namespace std;

static int position = 0;

inline std::vector<std::vector<rgb>> openImage(const std::string& imagePath) {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath.c_str(), &width, &height, &channels, 3); // Force 3 channels (RGB)

    if (!img) {
        throw std::runtime_error("Could not open or find the image: " + imagePath);
    }

    std::vector<std::vector<rgb>> pixels(height, std::vector<rgb>(width));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = (i * width + j) * 3;
            pixels[i][j] = { static_cast<int>(img[index]), static_cast<int>(img[index + 1]), static_cast<int>(img[index + 2]) };
        }
    }

    stbi_image_free(img);
    return pixels;
}

static RGB differnce;
static bool compare_differnce = false;
static bool threshold_limit = true;

inline auto give_average(const vector<vector<rgb>>& image, const pair<size_t,size_t>& group, const pair<int,int>& set, const float threshold,const bool compare){
    pair<bool,rgb> result = {true,{}};
    int r_sum = 0, g_sum = 0, b_sum = 0;
    int min_r = 255, max_r = 0, min_g = 255, max_g = 0, min_b = 255, max_b = 0;

    for (int dx = 0; dx < set.first; ++dx)
    for (int dy = 0; dy < set.second; ++dy) {
        int x = group.first * set.first + dx;
        int y = group.second * set.second + dy;
        const rgb& pixel = image[x][y];
        r_sum += pixel.r; min_r = std::min(min_r, pixel.r); max_r = std::max(max_r, pixel.r);
        g_sum += pixel.g; min_g = std::min(min_g, pixel.g); max_g = std::max(max_g, pixel.g);
        b_sum += pixel.b; min_b = std::min(min_b, pixel.b); max_b = std::max(max_b, pixel.b);
    }
    
    // Quick check: if range exceeds 2*threshold, skip detailed comparison
    if (threshold_limit && (max_r - min_r > 2 * threshold || max_g - min_g > 2 * threshold || max_b - min_b > 2 * threshold))
    result.first = false;
    
    if(compare_differnce)differnce = {static_cast<float>(max_r - min_r), static_cast<float>(max_g - min_g), static_cast<float>(max_b - min_b)};

    // Compute average
    int area = set.first * set.second;
    rgb average = {r_sum / area, g_sum / area, b_sum / area};
    result.second = average;

    if(compare)
    for (int dx = 0; dx < set.first && result.first; ++dx)
    for (int dy = 0; dy < set.second && result.first; ++dy) {
        int x = group.first * set.first + dx;
        int y = group.second * set.second + dy;
        const rgb& pixel = image[x][y];
        if (abs(pixel.r - average.r) > threshold ||
            abs(pixel.g - average.g) > threshold ||
            abs(pixel.b - average.b) > threshold) {
            result.first = false;
        }
    }
    
    return result;
}

auto push_averages(vector<vector<rgb>>& image,const pair<size_t,size_t>& group,const pair<int,int>& set,const rgb& average,int setcode) {

    RGB Average = {static_cast<float>(average.r), static_cast<float>(average.g), static_cast<float>(average.b),position};
    BlockAverage result = {group.first, group.second, Average};


    for (int dx = 0; dx < set.first; ++dx)
    for (int dy = 0; dy < set.second; ++dy) {
        int x = group.first * set.first + dx;
        int y = group.second * set.second + dy;
        image[x][y] = average;
        image[x][y].set_type_position = setcode;
    }
    position++;

    return result;
}

static int count16x16 = 0;
static int count8x8 = 0;
static int count4x4 = 0;
static int count2x2 = 0;
static int count1x2 = 0;
static int count2x1 = 0;

static int total16x16 = 0;
static int total8x8 = 0;
static int total4x4 = 0;
static int total2x2 = 0;
static int total1x2 = 0;
static int total2x1 = 0;
static int total1x1 = 0;


inline auto make_block(vector<vector<rgb>>& image) {

    vector<BlockAverage> block_averages;
    const int T16x16 = 5;
    const int T8x8 = 10;
    const int T4x4 = 15;
    const int T2x2 = 20;
    const int T1x2 = 20;
    const int T2x1 = 20;
    const bool not_1x1 = true;

    // Try to process a block and update counters.
    auto process_block = [&](int baseX, int baseY, int blockW, int blockH, int threshold, int setCode, int &total, int &count) -> bool {
        pair<size_t, size_t> group{ static_cast<size_t>(baseX / blockW), static_cast<size_t>(baseY / blockH) };
        pair<int, int> set{ blockW, blockH };
        auto avg = give_average(image, group, set, threshold, true);
        total++;
        if (!avg.first)return false;
        block_averages.push_back(push_averages(image, group, set, avg.second,setCode));
        block_averages.back().average.is_setted = setCode;
        count++;
        return true;
    };

    RGB d1,d2;
    // Process as 1x2 block.
    auto process_1x2 = [&](int subBaseX, int subBaseY) -> bool {
        pair<int, int> newSet{ 1, 2 };
        pair<size_t, size_t> group1{ static_cast<size_t>(subBaseX), static_cast<size_t>(subBaseY / 2) };
        pair<size_t, size_t> group2{ static_cast<size_t>(subBaseX + 1), static_cast<size_t>(subBaseY / 2) };

        compare_differnce = true;
        auto avgfirst = give_average(image, group1, newSet, T1x2, true);
        
        if (avgfirst.first) {
            d1 = differnce;

            block_averages.push_back(push_averages(image, group1, newSet, avgfirst.second,4));
            block_averages.back().average.is_setted = 4;

            auto avg2 = give_average(image, group2, newSet, T1x2, false);
            d2 = differnce;
            compare_differnce = false;
            block_averages.push_back(push_averages(image, group2, newSet, avg2.second,4));
            block_averages.back().average.is_setted = 4;
            return true;
        }
        // Alternate method.
        auto avgsecond = give_average(image, group2, newSet, T1x2, true);

        if (avgsecond.first) {
            d2 = differnce;
            block_averages.push_back(push_averages(image, group2, newSet, avgsecond.second,4));
            block_averages.back().average.is_setted = 4;
            
            auto avg2 = give_average(image, group1, newSet, T1x2, false);
            d1 = differnce;
            compare_differnce = false;
            block_averages.push_back(push_averages(image, group1, newSet, avg2.second,4));
            block_averages.back().average.is_setted = 4;
            return true;
        }
        compare_differnce = false;
        return false;
    };

    RGB D1,D2;
    // Process as 2x1 block.
    auto process_2x1 = [&](int subBaseX, int subBaseY) -> bool {
        pair<int, int> newSet{ 2, 1 };
        pair<size_t, size_t> group1{ static_cast<size_t>(subBaseX / 2), static_cast<size_t>(subBaseY) };
        pair<size_t, size_t> group2{ static_cast<size_t>(subBaseX / 2), static_cast<size_t>(subBaseY + 1) };

        compare_differnce = true;
        auto avgfirst = give_average(image, group1, newSet, T2x1, true);

        if (avgfirst.first) {
            D1 = differnce;

            block_averages.push_back(push_averages(image, group1, newSet, avgfirst.second,5));
            block_averages.back().average.is_setted = 5;
            auto avg = give_average(image, group2, newSet, T2x1, false);
            D2 = differnce;
            compare_differnce = false;

            block_averages.push_back(push_averages(image, group2, newSet, avg.second,5));
            block_averages.back().average.is_setted = 5;
            return true;
        }
        auto avgsecond = give_average(image, group2, newSet, T2x1, true);
        if (avgsecond.first) {
            D2 = differnce;

            block_averages.push_back(push_averages(image, group2, newSet, avgsecond.second,5));
            block_averages.back().average.is_setted = 5;
            auto avg = give_average(image, group1, newSet, T2x1, false);
            D1 = differnce;
            compare_differnce = false;

            block_averages.push_back(push_averages(image, group1, newSet, avg.second,5));
            block_averages.back().average.is_setted = 5;
            return true;
        }
        compare_differnce = false;
        return false;
    };

    int int_x = image.size() / 16;
    int int_y = image[0].size() / 16;
    for (size_t groupX = 0; groupX < int_x; ++groupX)
    for (size_t groupY = 0; groupY < int_y; ++groupY) {
        int baseX = groupX * 16;
        int baseY = groupY * 16;
        // Try full 16x16 block.
        if (process_block(baseX, baseY, 16, 16, T16x16, 0, total16x16, count16x16))
            continue;
        for (int v = 0; v < 2; v++)
        for (int c = 0; c < 2; c++) {
            int subBaseX8 = baseX + v * 8;
            int subBaseY8 = baseY + c * 8;
            if (process_block(subBaseX8, subBaseY8, 8, 8, T8x8, 1, total8x8, count8x8))
                continue;
            for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++) {
                int subBaseX4 = subBaseX8 + i * 4;
                int subBaseY4 = subBaseY8 + j * 4;
                if (process_block(subBaseX4, subBaseY4, 4, 4, T4x4, 2, total4x4, count4x4))
                    continue;
                for (int m = 0; m < 2; m++) 
                for (int n = 0; n < 2; n++) {
                    int subBaseX2 = subBaseX4 + m * 2;
                    int subBaseY2 = subBaseY4 + n * 2;
                    if (process_block(subBaseX2, subBaseY2, 2, 2, T2x2, 3, total2x2, count2x2))
                        continue;

                    REPEAT:
                    if (process_1x2(subBaseX2, subBaseY2) && process_2x1(subBaseX2, subBaseY2)) {
                        int FD1 = d1.r + d2.r + d1.g + d2.g + d1.b + d2.b;
                        int FD2 = D1.r + D2.r + D1.g + D2.g + D1.b + d2.b;

                        if (FD2 >= FD1) {
                            process_1x2(subBaseX2, subBaseY2);
                            total1x2++;
                            count1x2++;
                        } else {
                            process_2x1(subBaseX2, subBaseY2);
                            total2x1++;
                            count2x1++;
                        }
                    }
                    else if (process_1x2(subBaseX2, subBaseY2)){total1x2++;count1x2++;continue;}
                    else if (process_2x1(subBaseX2, subBaseY2)){total2x1++;count2x1++;continue;}

                    if (!threshold_limit) {
                        continue;
                    }
                    if (not_1x1) {
                        threshold_limit = false;
                        goto REPEAT;
                    }
                    else {
                        total1x1 += 4;
                        int offsets[4][2] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
                        for (auto& off : offsets) {
                            int x = subBaseX2 + off[0];
                            int y = subBaseY2 + off[1];
                            block_averages.push_back(BlockAverage{
                                static_cast<unsigned long long>(x), static_cast<unsigned long long>(y),
                                { static_cast<float>(image[x][y].r),
                                  static_cast<float>(image[x][y].g),
                                  static_cast<float>(image[x][y].b),
                                  position }
                            });
                            image[x][y].set_type_position = 6;
                        }
                    }
                    
                }
            }
        }
    }

    return block_averages;
}

inline void apply_averages_second(vector<vector<rgb>>& image, const vector<BlockAverage>& setted_blocks){

    vector<pair<int,int>> set_type = {{16, 16}, {8, 8}, {4, 4}, {2, 2}, {1, 2}, {2, 1}, {1, 1}};
    for (size_t i = 0; i < setted_blocks.size(); i++) {

        if(setted_blocks[i].average.is_setted == -1)continue;
        pair<int,int> set = set_type[setted_blocks[i].average.is_setted];
        for (int dx = 0; dx < set.first; ++dx)
        for (int dy = 0; dy < set.second; ++dy) {
            int x = setted_blocks[i].groupX * set.first + dx;
            int y = setted_blocks[i].groupY * set.second + dy;
            
            if (x >= 0 && x < image.size() && y >= 0 && y < image[0].size()) {
                image[x][y] = {static_cast<int>(setted_blocks[i].average.r), 
                               static_cast<int>(setted_blocks[i].average.g),
                               static_cast<int>(setted_blocks[i].average.b),
                               image[x][y].set_type_position};
            }
        }
    }
}

inline void saveImage(const std::vector<std::vector<rgb>>& image, const std::string& outPath) {
    int height = image.size();
    int width = image[0].size();
    std::vector<unsigned char> data;
    data.reserve(width * height * 3);
    for (const auto& row : image) {
        for (const auto& pixel : row) {
            data.push_back(static_cast<unsigned char>(pixel.r));
            data.push_back(static_cast<unsigned char>(pixel.g));
            data.push_back(static_cast<unsigned char>(pixel.b));
        }
    }
    if (!stbi_write_png(outPath.c_str(), width, height, 3, data.data(), width * 3)) {
        throw std::runtime_error("Failed to write image to " + outPath);
    }
}

#endif // IMAGE_TOOLS_H
