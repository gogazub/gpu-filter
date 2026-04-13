#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include "utils.h"


class MedianFilterGPU {
private:
    static uint8_t median_9(uint8_t window[9]);
public:
    static void median_filter_3x3_v1(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride, sycl::queue& q);
    static void median_filter_3x3_v2(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride, sycl::queue &q);
};



uint8_t MedianFilterGPU::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);

    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);

    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);

    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);

    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);

    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);

    window[4] = get_max(window[3], window[4]);

    return window[4];
}


//base version (GPU)
void MedianFilterGPU::median_filter_3x3_v1(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride, sycl::queue& q) {
    uint8_t* d_input = sycl::malloc_shared<uint8_t>(height * stride, q);
    uint8_t* d_output = sycl::malloc_shared<uint8_t>(height * stride, q);

    q.memcpy(d_input, input, height * stride * sizeof(uint8_t)).wait();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
            size_t y = idx[0];//строка
            size_t x = idx[1];//столбец

            size_t y0 = (y > 0) ? y - 1 : 0;
            size_t y1 = y;
            size_t y2 = (y < height - 1) ? y + 1 : y;

            size_t x0 = (x > 0) ? x - 1 : 0;
            size_t x1 = x;
            size_t x2 = (x < width - 1) ? x + 1 : x;

            uint8_t window[9];

            window[0] = d_input[y0 * stride + x0];
            window[1] = d_input[y0 * stride + x1];
            window[2] = d_input[y0 * stride + x2];

            window[3] = d_input[y1 * stride + x0];
            window[4] = d_input[y1 * stride + x1];
            window[5] = d_input[y1 * stride + x2];

            window[6] = d_input[y2 * stride + x0];
            window[7] = d_input[y2 * stride + x1];
            window[8] = d_input[y2 * stride + x2];

            uint8_t median = median_9(window);

            d_output[y * stride + x] = median;
        });
    });
    q.wait();

    q.memcpy(output, d_output, height * stride * sizeof(uint8_t)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}


//tiled version (GPU)
void MedianFilterGPU::median_filter_3x3_v2(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride, sycl::queue& q) {
    uint8_t* d_input = sycl::malloc_shared<uint8_t>(height * stride, q);
    uint8_t* d_output = sycl::malloc_shared<uint8_t>(height * stride, q);

    q.memcpy(d_input, input, height * stride * sizeof(uint8_t)).wait();

    const size_t BLOCK_SIZE = 16;//размер рабочей группы
    const size_t SHARED_SIZE = BLOCK_SIZE + 2;//размер блока + края для окна

    //кол-во блоков с округлением вверх
    size_t blocks_y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t blocks_x = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;

    q.submit([&](sycl::handler& h) {
        //выделяем память внутри shared memory (для рабочей группы)
        sycl::local_accessor<uint8_t, 2> shared(sycl::range<2>(SHARED_SIZE, SHARED_SIZE), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(blocks_y * BLOCK_SIZE, blocks_x * BLOCK_SIZE),
                sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
            ),
            [=](sycl::nd_item<2> item) {
                size_t global_y = item.get_global_id(0);
                size_t global_x = item.get_global_id(1);
                size_t local_y = item.get_local_id(0);
                size_t local_x = item.get_local_id(1);
                size_t block_y = item.get_group(0);
                size_t block_x = item.get_group(1);

                //коорд. начала блока в глобальном пространстве
                int block_start_y = block_y * BLOCK_SIZE;
                int block_start_x = block_x * BLOCK_SIZE;

                //загрузка окна 3х3 в shared memory
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int src_y = block_start_y + local_y + dy;//коорд. окна по оси y в глобальной памяти
                        int src_x = block_start_x + local_x + dx;//коорд. окна по оси x в глобальной памяти
                        int dst_y = local_y + 1 + dy;//коорд. окна по оси y в shared памяти
                        int dst_x = local_x + 1 + dx;//коорд. окна по оси x в shared памяти

                        //обработка границ (std::clamp ограничивает значение заданным диапазоном)
                        src_y = std::clamp(src_y, 0, (int)height - 1);
                        src_x = std::clamp(src_x, 0, (int)width - 1);

                        //записываем пиксель окна из глобальной памяти в shared
                        shared[dst_y][dst_x] = d_input[src_y * stride + src_x];
                    }
                }

                //синхронизация всех потоков внутри рабочей группы
                item.barrier();

                //проверка выхода за границы изображения, если размер изображения не кратен размеру блоков рабочей группы
                if (global_y < height && global_x < width) {
                    uint8_t window[9] = {
                        shared[local_y][local_x],     shared[local_y][local_x + 1],     shared[local_y][local_x + 2],
                        shared[local_y + 1][local_x], shared[local_y + 1][local_x + 1], shared[local_y + 1][local_x + 2],
                        shared[local_y + 2][local_x], shared[local_y + 2][local_x + 1], shared[local_y + 2][local_x + 2]
                    };
                    d_output[global_y * stride + global_x] = median_9(window);
                }
            }
        );
    });
    q.wait();

    q.memcpy(output, d_output, height * stride * sizeof(uint8_t)).wait();
    sycl::free(d_input, q);
    sycl::free(d_output, q);
}