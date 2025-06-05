#ifndef SMOOTH_TOOLS2_H
#define SMOOTH_TOOLS2_H

#include "basic_setup_tools.h"

class smooth {
private:
    vector<RGB> calculate_smooth_pixels(int BlockLength, const RGB& mainRGB, const vector<RGB>& surroundRGB) {
        vector<RGB> output;
        const double epsilon = 1.0;
        const int c = BlockLength / 2;

        // Representative points and factors for main block and up to 8 surrounding blocks
        vector<pair<double, double>> points = {
            {c, c},                   // Main block (center)
            {-1, -1},                 // Top-left
            {c, -1},                  // Top
            {BlockLength, -1},        // Top-right
            {-1, c},                  // Left
            {BlockLength, c},         // Right
            {-1, BlockLength},        // Bottom-left
            {c, BlockLength},         // Bottom
            {BlockLength, BlockLength}// Bottom-right
        };
        double middle_f = 1.0, side_f = 1.0, corner_f = 1.0;

        vector<double> factors = {middle_f, corner_f, side_f, corner_f, side_f, side_f, corner_f, side_f, corner_f};
        // vector<double> factors = {middle_f, side_f, side_f, side_f, side_f};
        
        // Adjust based on actual number of surrounding blocks
        int num_blocks = 1 + surroundRGB.size(); // Main block + surrounding blocks
        points.resize(num_blocks);
        factors.resize(num_blocks);

        // Populate RGB values
        vector<RGB> rgb_value;
        rgb_value.push_back(mainRGB);
        for (const auto& rgb : surroundRGB) {
            rgb_value.push_back(rgb);
        }

        output.resize(BlockLength * BlockLength);

        // Iterate over each pixel in the block
        for (int y = 0; y < BlockLength; ++y) {
            for (int x = 0; x < BlockLength; ++x) {
                double sumWeights = 0.0;
                double sumR = 0.0, sumG = 0.0, sumB = 0.0;
                for (int i = 0; i < num_blocks; ++i) {

                    double dx = x - points[i].first;
                    double dy = y - points[i].second;
                    // cout << dx << " " << dy << " " << i << "|";
                    if (i == 2) {
                        dx = 0;
                    }
                    else if (i == 4) {
                        dy = 0;
                    }
                    else if (i == 5) {
                        dy = 0;
                    }
                    else if (i == 7) {
                        dx = 0;
                    }

                    double distSquared = dx * dx + dy * dy;
                    double weight = factors[i] / (distSquared + epsilon);
                    sumWeights += weight;
                    sumR += weight * rgb_value[i].r;
                    sumG += weight * rgb_value[i].g;
                    sumB += weight * rgb_value[i].b;
                }

                double invSumWeights = 1.0 / sumWeights;
                output[y * BlockLength + x].r = static_cast<float>(sumR * invSumWeights);
                output[y * BlockLength + x].g = static_cast<float>(sumG * invSumWeights);
                output[y * BlockLength + x].b = static_cast<float>(sumB * invSumWeights);
            }
        }
        return output;
    }

    void take_the_block(pair<int, int> p, int BlockLength, vector<vector<CRGB>>& linked_image, vector<MRGB>& linked_to) {
        int x = p.first;
        int y = p.second;

        // Correct surrounding block positions (top-left corners)
        const vector<pair<int, int>> position_search = {
            {x - BlockLength, y - BlockLength}, // Top-left
            {x, y - BlockLength},               // Top
            {x + BlockLength, y - BlockLength}, // Top-right
            {x - BlockLength, y},               // Left
            {x + BlockLength, y},               // Right
            {x - BlockLength, y + BlockLength}, // Bottom-left
            {x, y + BlockLength},               // Bottom
            {x + BlockLength, y + BlockLength}  // Bottom-right
        };

        int i = 0;
        vector<RGB> surroundRGB;
        for (const auto& pos : position_search) {
            int px = pos.first;
            int py = pos.second;
            if (px >= 0 && px < linked_image.size() && py >= 0 && py < linked_image[0].size()) {
                int idx = linked_image[px][py].main_RGB_index;
                if (idx != -1) {
                    if (i == 0 || i == 2 || i == 5 || i == 7) {
                        surroundRGB.push_back(average_clamped(linked_to[idx].value, linked_image[x][y].value, 0.7));
                    }
                    else
                    surroundRGB.push_back(average_clamped(linked_to[idx].value, linked_image[x][y].value, 0.6));
                }
            }
            else {
                surroundRGB.push_back(linked_image[x][y].value);
            }
            i++;
        }

        int main_idx = linked_image[x][y].main_RGB_index;
        vector<RGB> result = calculate_smooth_pixels(BlockLength, linked_to[main_idx].value, surroundRGB);

        int index = 0;
        for (size_t X = 0; X < BlockLength; X++) {
            for (size_t Y = 0; Y < BlockLength; Y++) {
                linked_image[x + Y][y + X].value = result[index];
                index++;
            }
        }
    }

public:
    void smooth_the_image(vector<vector<CRGB>>& linked_image, vector<MRGB>& linked_to) {
        for (size_t i = 0; i < linked_to.size(); i++) {
            if (linked_to[i].connected_RGBs.size() < 4) continue;
            
            int BlockLength = static_cast<int>(sqrt(linked_to[i].connected_RGBs.size()));
            // Ensure the block is square
            if (BlockLength * BlockLength == linked_to[i].connected_RGBs.size()) {
                take_the_block(linked_to[i].coordinate, BlockLength, linked_image, linked_to);
            }
        }
    }
};

#endif