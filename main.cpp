#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cstdint>
#include <string>
#include <sycl/sycl.hpp>
#include "processImageData.h"
#include "medianFilter.h"
#include "medianFilterGPU.h"

struct RgbChannels {
    std::vector<uint8_t> red;
    std::vector<uint8_t> green;
    std::vector<uint8_t> blue;

    explicit RgbChannels(size_t pixelCount)
        : red(pixelCount), green(pixelCount), blue(pixelCount) {}
};

bool buffers_equal(const uint8_t* left, const uint8_t* right, size_t size) {
    for (size_t index = 0; index < size; ++index)
        if (left[index] != right[index]) return false;
    return true;
}

bool rgb_channels_equal(const RgbChannels& left, const RgbChannels& right) {
    size_t pixelCount = left.red.size();
    return buffers_equal(left.red.data(), right.red.data(), pixelCount)
        && buffers_equal(left.green.data(), right.green.data(), pixelCount)
        && buffers_equal(left.blue.data(), right.blue.data(), pixelCount);
}

void warmup_gpu(sycl::queue& queue) {
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {});
    });
    queue.wait();
}

long elapsed_ms(std::chrono::high_resolution_clock::time_point startTime,
                std::chrono::high_resolution_clock::time_point endTime) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}

long run_cpu_filter(const RgbChannels& inputChannels, RgbChannels& cpuOutput,
                    int width, int height, int stride) {
    auto startTime = std::chrono::high_resolution_clock::now();
    MedianFilter::median_filter_3x3_rgb(
        inputChannels.red.data(), inputChannels.green.data(), inputChannels.blue.data(),
        cpuOutput.red.data(), cpuOutput.green.data(), cpuOutput.blue.data(),
        width, height, stride);
    auto endTime = std::chrono::high_resolution_clock::now();
    return elapsed_ms(startTime, endTime);
}

long run_gpu_v1_filter(const RgbChannels& inputChannels, RgbChannels& gpuOutput,
                       int width, int height, int stride, int iterations,
                       sycl::queue& queue) {
    MedianFilterGPU::median_filter_3x3_rgb_v1(
        inputChannels.red.data(), inputChannels.green.data(), inputChannels.blue.data(),
        gpuOutput.red.data(), gpuOutput.green.data(), gpuOutput.blue.data(),
        width, height, stride, queue);

    auto startTime = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < iterations; iteration++) {
        MedianFilterGPU::median_filter_3x3_rgb_v1(
            inputChannels.red.data(), inputChannels.green.data(), inputChannels.blue.data(),
            gpuOutput.red.data(), gpuOutput.green.data(), gpuOutput.blue.data(),
            width, height, stride, queue);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    return elapsed_ms(startTime, endTime) / iterations;
}

long run_gpu_v2_filter(const RgbChannels& inputChannels, RgbChannels& gpuOutput,
                       int width, int height, int stride, int iterations,
                       sycl::queue& queue) {
    MedianFilterGPU::median_filter_3x3_rgb_v2(
        inputChannels.red.data(), inputChannels.green.data(), inputChannels.blue.data(),
        gpuOutput.red.data(), gpuOutput.green.data(), gpuOutput.blue.data(),
        width, height, stride, queue);

    auto startTime = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < iterations; iteration++) {
        MedianFilterGPU::median_filter_3x3_rgb_v2(
            inputChannels.red.data(), inputChannels.green.data(), inputChannels.blue.data(),
            gpuOutput.red.data(), gpuOutput.green.data(), gpuOutput.blue.data(),
            width, height, stride, queue);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    return elapsed_ms(startTime, endTime) / iterations;
}

void save_rgb_image(const std::string& outputFilename, int width, int height,
                    const RgbChannels& outputChannels) {
    BMP outputBMP;
    create_BMP_rgb(outputBMP, width, height,
                   outputChannels.red.data(),
                   outputChannels.green.data(),
                   outputChannels.blue.data());
    outputBMP.WriteToFile(outputFilename.c_str());
}

int main() {
    const int iterations = 100;
    const std::string inputDirectory = "img/noise/";
    const std::string outputDirectory = "img/filtered/";
    const std::string imageFilename = "noisy_image.bmp";

    BMP inputBMP;
    std::string inputFilename = inputDirectory + imageFilename;

    if (!inputBMP.ReadFromFile(inputFilename.c_str())) {
        std::cerr << "Не удалось открыть файл: " << inputFilename << std::endl;
        return 1;
    }

    const int width = inputBMP.TellWidth();
    const int height = inputBMP.TellHeight();
    const int stride = width;
    const size_t pixelCount = (size_t)width * height;

    std::cout << "Изображение: " << width << " x " << height << " px" << std::endl;

    RgbChannels inputChannels(pixelCount);
    RgbChannels cpuOutput(pixelCount);
    RgbChannels gpuV1Output(pixelCount);
    RgbChannels gpuV2Output(pixelCount);

    load_rgb_from_bmp(inputBMP, inputChannels.red.data(), inputChannels.green.data(), inputChannels.blue.data());

    long cpuTimeMs = run_cpu_filter(inputChannels, cpuOutput, width, height, stride);
    std::cout << "Single thread:  " << cpuTimeMs << " ms" << std::endl;

    sycl::queue queue;
    std::cout << "GPU: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    warmup_gpu(queue);

    long gpuV1TimeMs = run_gpu_v1_filter(inputChannels, gpuV1Output,
                                         width, height, stride, iterations, queue);
    std::cout << "GPU v1 (naive): " << gpuV1TimeMs
              << " ms (среднее за " << iterations << " итераций)" << std::endl;

    long gpuV2TimeMs = run_gpu_v2_filter(inputChannels, gpuV2Output,
                                         width, height, stride, iterations, queue);
    std::cout << "GPU v2 (shared): " << gpuV2TimeMs
              << " ms (среднее за " << iterations << " итераций)" << std::endl;

    bool gpuV1IsCorrect = rgb_channels_equal(cpuOutput, gpuV1Output);
    bool gpuV2IsCorrect = rgb_channels_equal(cpuOutput, gpuV2Output);

    std::cout << "\nПроверка корректности:" << std::endl;
    std::cout << "  CPU == GPU v1: " << (gpuV1IsCorrect ? "OK" : "FAILED") << std::endl;
    std::cout << "  CPU == GPU v2: " << (gpuV2IsCorrect ? "OK" : "FAILED") << std::endl;

    assert(gpuV1IsCorrect && "GPU v1 дал неверный результат!");
    assert(gpuV2IsCorrect && "GPU v2 дал неверный результат!");

    std::string outputFilename = outputDirectory + "filtered_" + imageFilename;
    save_rgb_image(outputFilename, width, height, gpuV2Output);
    std::cout << "\nОтфильтрованное изображение сохранено: " << outputFilename << std::endl;

    return 0;
}
