#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "basic_setup_tools.h"
#include "line_algorithm.h"
#include "image_tools.h"
#include "smooth_tools.h"

using namespace std;

void processSettedBlocks(vector<BlockAverage>& setted_blocks,float line_threshold, int bit_number) {
    vector<RGB> pixelRGB;

    for (size_t i = 0; i < setted_blocks.size(); i++) {
        pixelRGB.push_back(setted_blocks[i].average);
    }

    startTimer();
    auto calculated_result = calculate_lines(pixelRGB, line_threshold);
    auto t3 = stopTimer();
    cout << "time taken to calculate " << calculated_result.size() * 6 << " lines" << t3 << '\n';
    startTimer();

    vector<RGB> average_resultRGB;

    average_resultRGB.reserve(pixelRGB.size());

    for (size_t x = 0; x < calculated_result.size(); x++) {
        auto r = find_end_points(calculated_result[x]);
        for (size_t y = 0; y < calculated_result[x].size(); y++) {
            RGB indexed_result = indexed_value(r.first, r.second, bit_number, 
                nearest_index(calculated_result[x][y], r.first, r.second, bit_number));
            average_resultRGB.push_back({indexed_result.r, indexed_result.g, indexed_result.b, 
                calculated_result[x][y].position, calculated_result[x][y].is_setted});
        }
    }

    sort(average_resultRGB.begin(), average_resultRGB.end(), [](const RGB& a, const RGB& b) {
        return a.position < b.position;
    });

    for (size_t i = 0; i < setted_blocks.size(); i++){
        setted_blocks[i].average = average_resultRGB[i];
    }

    auto t4 = stopTimer();
    cout << "time taken to indexed -- " << t4 << '\n';
}
 

int main() {

    vector<vector<rgb>> pixels;
    cout << "program is running." << '\n';
    startTimer();
    try {
        std::string imagePath = "test_img.jpg";

        pixels = openImage(imagePath);
        std::cout << "Image loaded successfully with size: " << pixels.size() << "x" << pixels[0].size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }
    auto t1 = stopTimer();
    cout << "time taken in loading image -- "<< t1 << " miliseconds"<< '\n';


    startTimer();
    auto setted_blocks = make_block(pixels);
    auto t2 = stopTimer();
    cout << "time taken in making sets -- "<< t2 << " miliseconds"<< '\n';

    cout << setted_blocks.size() << '\n';

    cout << "total data of the sets in KB -- " << ((total16x16 + total8x8 + total4x4 + total2x2 + total1x2 + total2x1)/8)/1000 << '\n';
    cout << "total data in KB -- " << (count16x16 + count8x8 + count4x4 + count2x2 + count1x2*2 + count2x1*2)/1000/2 << " orignal data in KB -- " << (pixels.size() * pixels[0].size() * 3)/1000 << '\n';


    processSettedBlocks(setted_blocks,10,32);

    apply_averages_second(pixels,setted_blocks);

    startTimer();
    // smooth(pixels,setted_blocks);
    auto t = stopTimer();
    saveImage(pixels, "");
    cout << "saved - " << t << endl;
    return 0;
}