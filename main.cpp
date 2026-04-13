#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

#include <sycl/sycl.hpp>
#include "processImageData.h"
#include "medianFilter.h"
#include "medianFilterGPU.h"
#include "medianFilterSIMD.h"
//using namespace sycl;



//сравнение результатов 2-ух фильтров
bool compare_data(const uint8_t* A, const uint8_t* B, size_t size) {
    for (size_t i = 0; i < size; ++i)
        if (A[i] != B[i]) return false;
    return true;
}

//функция для "разогрева" GPU (инициализация очереди, выделение ресурсов, компиляция)
void warmupGPU(sycl::queue q) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1> idx) {});
        });
    q.wait();
}



int main() {
    //-------------------------- ЗАШУМЛЕННОЕ ИЗОБРАЖЕНИЕ --------------------------

    BMP inputBMP;
    const int ITERATIONS = 1000;//количество повторений для нагрузки

    std::string inputfilepath = "img/noise/";
    std::string filename = "impulse_5%_1.bmp";
    //std::string filename = "gaussian_50.bmp";
    inputBMP.ReadFromFile((inputfilepath + filename).c_str());

    const int w = inputBMP.TellWidth();
    const int h = inputBMP.TellHeight();
    uint8_t* inputPixels = new uint8_t[w * h];
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ebmpBYTE pixelID = inputBMP.GetPixel(x, y).Red;
            inputPixels[y * w + x] = static_cast<uint8_t>(pixelID);
        }
    }

    //-------------------------- ФИЛЬТРАЦИЯ + ЗАМЕРЫ --------------------------



    //ОДНОПОТОЧНАЯ ВЕРСИЯ
    uint8_t* outputPixels = new uint8_t[w * h];

    auto start1 = std::chrono::high_resolution_clock::now();
    MedianFilter::median_filter_3x3(inputPixels, outputPixels, w, h, w);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Single thread version: " << duration1.count() << " ms" << std::endl;


    //GPU ВЕРСИЯ

    //warmup
    sycl::queue q;
    warmupGPU(q);

    //base version (GPU)
    uint8_t* outputPixels_gpu = new uint8_t[w * h];

    auto start2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        MedianFilterGPU::median_filter_3x3_v1(inputPixels, outputPixels_gpu, w, h, w, q);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "GPU version v1: " << duration2.count() << " ms" << std::endl;

    //tiled version (GPU)
    uint8_t* outputPixels_gpu_v2 = new uint8_t[w * h];

    auto start3 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        MedianFilterGPU::median_filter_3x3_v2(inputPixels, outputPixels_gpu_v2, w, h, w, q);
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "GPU version v2: " << duration3.count() << " ms" << std::endl;

    //-------------------------- ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ --------------------------

    //создаем отфильтрованное изображение (GPU)
    //BMP outputBMP;
    //create_BMP_grayscale(inputBMP, outputBMP, outputPixels_gpu);
    //std::string outputfilepath = "img/filtered/";
    //std::string prefix = "filtered_";
    //outputBMP.WriteToFile((outputfilepath + prefix + filename).c_str());

    //-------------------------- ПРОВЕРКА --------------------------

    assert(compare_data(outputPixels, outputPixels_gpu, w * h));
    assert(compare_data(outputPixels, outputPixels_gpu_v2, w * h));

    std::cout << "Processing complete!" << std::endl;
    return 0;
}