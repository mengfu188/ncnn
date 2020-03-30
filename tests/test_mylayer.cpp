//
// Created by mengf on 2020/3/29.
//

#include "layer/mylayer.h"
#include "datareader.h"
#include "net.h"
#include "testutil.h"

using namespace ncnn;

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const { return 0; }
    virtual size_t read(void* /*buf*/, size_t size) const { return size; }
};

void pretty_print(const Mat& m)
{
    printf("width %d, height %d, channel %d \n", m.w, m.h, m.c);
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
    printf("#################\n");
}

int main()
{
    SRAND(7767517);
    DataReaderFromEmpty dr;

    Net net;
    net.load_param("model-mylayer.param");
    net.load_model(dr);

    Mat random = RandomMat(5, 2, 3);

    Extractor ex = net.create_extractor();
    ex.input(0, random);

    Mat out;
    ex.extract("mylayer0", out);
    pretty_print(random);
    pretty_print(out);

    return 0;
}