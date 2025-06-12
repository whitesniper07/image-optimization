#ifndef IMAGE_TOOLS2_H
#define IMAGE_TOOLS2_H

#include "basic_setup_tools.h"
#define STB_IMAGE_IMPLEMENTATION
#include "libraries/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libraries/stb_image_write.h"

using namespace std;


inline std::vector<std::vector<rgb>> openImage(const std::string& imagePath) {
    int width, height, channels;
    unsigned char* img = stbi_load(imagePath.c_str(), &width, &height, &channels, 3); // Force 3 channels (RGB)

    if (!img) {
        throw std::runtime_error("Could not open or find the image: " + imagePath);
    }

    std::vector<std::vector<rgb>> pixels(height, std::vector<rgb>(width));

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const int index = (i * width + j) * 3;
            pixels[i][j] = { static_cast<int>(img[index]), static_cast<int>(img[index + 1]), static_cast<int>(img[index + 2]) };
        }
    }

    stbi_image_free(img);
    return pixels;
}

inline void saveImage(const std::vector<std::vector<rgb>>& image, const std::string& outPath) {
    const int height = image.size();
    const int width = image[0].size();
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

class block {
    private:
    
    float difference = 0.0f;

    // Check if a block’s pixels are similar; return average if true
    pair<bool, RGB> give_average_block(
        const vector<vector<RGB>>& original_pixels,
        const int& X,
        const int& Y,
        const int& blockX,
        const int& blockY,
        const float& threshold) {

        difference = 0.0f;
        float r_sum = 0, g_sum = 0, b_sum = 0;
        float min_r = 255, max_r = 0, min_g = 255, max_g = 0, min_b = 255, max_b = 0;
        int count = 0;

        for (int dx = 0; dx < blockX; dx++) {
            for (int dy = 0; dy < blockY; dy++) {
                int x = X + dx;
                int y = Y + dy;
                const RGB& pixel = original_pixels[x][y];
                r_sum += pixel.r; min_r = min(min_r, pixel.r); max_r = max(max_r, pixel.r);
                g_sum += pixel.g; min_g = min(min_g, pixel.g); max_g = max(max_g, pixel.g);
                b_sum += pixel.b; min_b = min(min_b, pixel.b); max_b = max(max_b, pixel.b);
                count++;
            }
        }
        RGB average = {r_sum / count, g_sum / count, b_sum / count};

        if (count == 0 || max_r - min_r > 2 * threshold || max_g - min_g > 2 * threshold || max_b - min_b > 2 * threshold) {
            return {false, average};
        }


        for (int dx = 0; dx < blockX; dx++) {
            for (int dy = 0; dy < blockY; dy++) {
                int x = X + dx;
                int y = Y + dy;
                const RGB& pixel = original_pixels[x][y];
                const int 
                dr = abs(pixel.r - average.r),
                dg = abs(pixel.g - average.g),
                db = abs(pixel.b - average.b);

                difference += dr + dg + db;
                if (dr > threshold ||
                    dg > threshold ||
                    db > threshold) {
                    return {false, {average}};
                }
            }
        }
        return {true, average};
    }

    
    void add_in(vector<MRGB>& linked_to, vector<vector<CRGB>>& linked_image,
                const RGB& ave, const int& X, const int& Y, const int& blockX, const int& blockY) {
        linked_to.push_back({ave, {X, Y}, {}});
        int idx = linked_to.size() - 1;
        MRGB* current_mrgb = &linked_to[idx]; // For connected_RGBs only
        for (int bx = 0; bx < blockX; bx++) {
            for (int by = 0; by < blockY; by++) {
                const int dx = X + bx;
                const int dy = Y + by;
                linked_image[dx][dy].main_RGB_index = idx;
                linked_image[dx][dy].value = ave;
                current_mrgb->connected_RGBs.push_back({dx, dy});
            }
        }
    }


    bool process_2x2_block(const vector<vector<RGB>>& original_pixels, 
                          vector<MRGB>& linked_to, 
                          vector<vector<CRGB>>& linked_image,
                          int x, int y, float threshold) {

        // Get averages and differences for different block configurations
        auto [block2x2_valid, block2x2_avg] = give_average_block(original_pixels, x, y, 2, 2, threshold);
        float diff2x2 = difference;

        auto [block1x2_left_valid, block1x2_left_avg] = give_average_block(original_pixels, x, y, 1, 2, threshold);
        float diff1x2_left = difference;
                        
        auto [block1x2_right_valid, block1x2_right_avg] = give_average_block(original_pixels, x + 1, y, 1, 2, threshold);
        float diff1x2_right = difference;

        auto [block2x1_top_valid, block2x1_top_avg] = give_average_block(original_pixels, x, y, 2, 1, threshold);
        float diff2x1_top = difference;

        auto [block2x1_bottom_valid, block2x1_bottom_avg] = give_average_block(original_pixels, x, y + 1, 2, 1, threshold);
        float diff2x1_bottom = difference;

        // Calculate average differences
        float avg_diff_1x2 = (diff1x2_left + diff1x2_right) / 2;
        float avg_diff_2x1 = (diff2x1_top + diff2x1_bottom) / 2;

        // Check if any configuration is valid
        bool any_valid = block2x2_valid || 
                        (block1x2_left_valid || block1x2_right_valid) || 
                        (block2x1_top_valid || block2x1_bottom_valid);

        if (!any_valid) return true;
                        
        // Adjust differences based on validity
        if (!block2x2_valid) diff2x2 = max(avg_diff_1x2, avg_diff_2x1);
        if (!(block1x2_left_valid || block1x2_right_valid)) avg_diff_1x2 = max(avg_diff_2x1, diff2x2);
        if (!(block2x1_top_valid || block2x1_bottom_valid)) avg_diff_2x1 = max(avg_diff_1x2, diff2x2);

        // Choose best configuration based on minimum difference
        float min_diff = min({diff2x2, avg_diff_2x1, avg_diff_1x2});

        if (diff2x2 == min_diff) {
            add_in(linked_to, linked_image, block2x2_avg, x, y, 2, 2);
        }

        else if (avg_diff_1x2 == min_diff) {
            add_in(linked_to, linked_image, block1x2_left_avg, x, y, 1, 2);
            add_in(linked_to, linked_image, block1x2_right_avg, x + 1, y, 1, 2);
        }

        else if (avg_diff_2x1 == min_diff) {
            add_in(linked_to, linked_image, block2x1_top_avg, x, y, 2, 1);
            add_in(linked_to, linked_image, block2x1_bottom_avg, x, y + 1, 2, 1);
        }

        return false;
    }

    // Recursively process a block of the image and assign colors based on similarity.
    void process_block(const vector<vector<RGB>>& original_pixels, 
                       vector<MRGB>& linked_to, 
                       vector<vector<CRGB>>& linked_image, 
                       int x, int y, int size, float threshold) {
                    
        if (x >= original_pixels.size() || y >= original_pixels[0].size()) return;

        // Base case: if the block is a single pixel.
        if (size == 1) {
            RGB pixel = original_pixels[x][y];

            // connecting MRGB and CRGB.
            linked_to.push_back({pixel, {x, y}, {{x, y}}});
            linked_image[x][y].main_RGB_index = linked_to.size() - 1;
            linked_image[x][y].value = pixel;
        } 
        else {
            // In process_block function:
            bool flag = true;
            if (size == 2) {
                flag = process_2x2_block(original_pixels, linked_to, linked_image, x, y, threshold);
            }
            else {  
                const pair<bool, RGB> result = give_average_block(original_pixels, x, y, size,size, threshold);
                if (result.first) {
                    add_in(linked_to, linked_image, result.second, x, y, size, size);
                    flag = false;
                }
            }
            if (!flag)return;

            // If not, subdivide the block into four smaller blocks (quadrants).
            const int half_size = size / 2;
            process_block(original_pixels, linked_to, linked_image, x, y, half_size, threshold);

            if (x + half_size < original_pixels.size()) 
            process_block(original_pixels, linked_to, linked_image, x + half_size, y, half_size, threshold);

            if (y + half_size < original_pixels[0].size()) 
            process_block(original_pixels, linked_to, linked_image, x, y + half_size, half_size, threshold);

            if (x + half_size < original_pixels.size() && y + half_size < original_pixels[0].size()) 
            process_block(original_pixels, linked_to, linked_image, x + half_size, y + half_size, half_size, threshold);

        }
    }

    public:

    // Process the entire image into blocks by iterating through it in fixed block steps.
    void make_block(vector<vector<RGB>>& original_image, 
                    vector<MRGB>& linked_to, 
                    vector<vector<CRGB>>& linked_image) {


        // Get the image dimensions.
        const int height = original_image.size();
        const int width = original_image[0].size();

        // Set the initial block size.
        constexpr int block_size = 16;

        // Loop over the image in steps defined by block_size.
        for (int x = 0; x < height; x += block_size) {
            for (int y = 0; y < width; y += block_size) {
                const int current_size = min(block_size, min(height - x, width - y));
                process_block(original_image, linked_to, linked_image, x, y, current_size, 20.0);
            }
        }
    }
};



#endif 