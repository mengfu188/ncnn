//
// Created by mengf on 2020/3/29.
//

#include "mylayer.h"

namespace ncnn{
    DEFINE_LAYER_CREATOR(MyLayer)
    // 如何加载参数
    int MyLayer::load_param(const ncnn::ParamDict &pd){
        channels = pd.get(0, 0);
        eps = pd.get(1, 0.001f);
        return 0;
    }
    // 如何加载权重
    int MyLayer::load_model(const ncnn::ModelBin &mb) {
        gamma_data = mb.load(channels, 1); // 0代表自动推倒，1代表32为格式
        if(gamma_data.empty()){

            fprintf(stderr, "mylayer load_model is empty\n");

//            return -100; // return non-zero on error, -100 indicate OOM
            gamma_data.fill(2.0);
        }

        return 0; // return zero if success
    }
    // 设置推理行为，应该是决定调用什么forword方法
    MyLayer::MyLayer(){
        one_blob_only=true;
        support_inplace=true;
    }

    int MyLayer::forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const {
        fprintf(stdout, "mylayer forward");
        if (bottom_blob.c != channels)
            return -1;

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int size = w * h;

        if (top_blob.empty())
            return -100;

        for (int q = 0; q < channels; q++){
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            const float gamma = gamma_data[q];

            for ( int i = 0; i < size; i++){
                outptr[i] = (ptr[i] + eps) * gamma;
            }
        }

        return 0;
    }

    int MyLayer::forward_inplace(ncnn::Mat &bottom_top_blob, const ncnn::Option &opt) const {
        fprintf(stdout, "mylayer forward_inplace\n");
        // check input dims
        if (bottom_top_blob.c != channels)
            return -1;

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int size = w*h;

        for(int q = 0; q < channels; q++){
            float* ptr = bottom_top_blob.channel(q);
            const float gamma = gamma_data[q];
            for (int i = 0; i < size; i++){
                ptr[i] = (ptr[i] + eps) * gamma;
            }
        }
        return 0;
    }

};