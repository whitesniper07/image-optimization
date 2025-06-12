
#include <math.h>
#include "header_files/basic_setup_tools.h"
#include "header_files/image_tools.h"
#include "header_files/smooth_tools.h"
#include "header_files/line_algorithm.h"

using namespace std;


void extract_main_values(vector<vector<CRGB>>& linked_image,vector<MRGB>& linked_to) {
    vector<MRGB> arranged_values;
    arranged_values.reserve(linked_to.size());

    vector<vector<int>> index_list(linked_image.size(), vector<int>(linked_image[0].size(), -1));

    for (size_t i = 0; i < linked_to.size(); i++) {
        index_list[linked_to[i].coordinate.first][linked_to[i].coordinate.second] = i;
    }

    for (size_t x = 0; x < index_list.size(); x++) {
        for (size_t y = 0; y < index_list[0].size(); y++) {
            if (index_list[x][y] == -1)continue;
            const int index = index_list[x][y];
            arranged_values.push_back({
                linked_to[index].value,
                linked_to[index].coordinate,
                linked_to[index].connected_RGBs,
            });

            for (size_t i = 0; i < linked_to[index].connected_RGBs.size(); i++) {
                auto [fst, snd] = linked_to[index].connected_RGBs[i];

                linked_image[fst][snd].main_RGB_index = arranged_values.size() - 1;
            }
        }
    }
    linked_to = arranged_values;
};



int main(int argc, char const *argv[]) {
    const vector<string> locations = {"images/jesko 3 low.jpg", "images/darth vader.jpg"};
    const auto start_image = openImage(locations[0]);

    vector<vector<RGB>> RGBpixels;
    for (const auto& row : start_image) {
        RGBpixels.push_back({});
        for (const auto&[r, g, b] : row) {
            RGBpixels.back().push_back({static_cast<float>(r), static_cast<float>(g), static_cast<float>(b)});
        }
    }

    block b;
    const int height = RGBpixels.size();
    const int width = RGBpixels[0].size();
    vector<MRGB> linked_to;
    vector<vector<CRGB>> linked_image(height, vector<CRGB>(width, {{0, 0, 0}, -1}));
    cout << height << " " << width << '\n';

    startTimer();
    b.make_block(RGBpixels, linked_to, linked_image);
    
    extract_main_values(linked_image, linked_to);
    const auto t = stopTimer();
    cout << "time taken = " << t << '\n';



    const float p = (static_cast<float>(linked_to.size()) / static_cast<float>(linked_image.size() * linked_image[0].size())) * 100.0f;
    cout << linked_to.size() << " " << linked_image.size() * linked_image[0].size() << " " << p << '\n';
    cout << (linked_to.size() / 2)/1000 << " KB. old - " << (linked_image.size() * linked_image[0].size() * 3)/1000 << " KB" << '\n';

    cout << "timer is started\n";
    startTimer();
    auto lines = calculate_lines(linked_to, 20);
    const auto t2 = stopTimer();
    cout << "time taken = " << t2 << " lines size - " << lines.size() << '\n';


    
    for (size_t x = 0; x < lines.size(); x++) {
        for (size_t y = 0; y < lines[x].size(); y++) {
            printRGB(linked_to[lines[x][y]].value);
        }
        cout << "|\n";
    }
    
    
    // smooth s;
    // s.smooth_the_image(linked_image, linked_to);

    vector<vector<rgb>> image(linked_image.size(),vector<rgb>(linked_image[0].size()));

    for (int x = 0; x < linked_image.size(); x++) {
        for (int y = 0; y < linked_image[x].size(); y++) {
            if (linked_image[x][y].main_RGB_index != -1) {
                image[x][y] = to_rgb(linked_image[x][y].value);
            } else {
                image[x][y] = {0, 0, 255};
            }
        }
    }
    
    // saveImage(image, "images/image.png");
    cout << "done\n";
    return 0;
}

