//
// Created by mengf on 2020/3/29.
//

#ifndef NCNN_MYLAYER_H
#define NCNN_MYLAYER_H

#include "layer.h"

namespace ncnn{

class MyLayer : public Layer{
public:
    MyLayer();
    virtual int load_param(const ParamDict & pd);
    virtual int load_model(const ModelBin& mb);


    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;// new code, optional
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;// new code
private:
    int channels;
    float gamma;
    Mat weight;
    float eps;
    Mat gamma_data;

};

};

#endif //NCNN_MYLAYER_H
