# Post Training Quantization Tools

To support int8 model deployment on mobile devices,we provide the universal post training quantization tools which can convert the float32 model to int8 model.

The old convert tool and technical detail is in [Caffe-Int8-Convert-Tools](https://github.com/BUG1989/caffe-int8-convert-tools)

## User Guide

Example with mobilenet,just need three steps.

### 1. Optimization graphic 

```
./ncnnoptimize mobilenet-fp32.param mobilenet-fp32.bin mobilenet-nobn-fp32.param mobilenet-nobn-fp32.bin 0
```

### 2. Create the calibration table file

We suggest that using the verification dataset for calibration, which is more than 5000 images.

```
./ncnn2table --param=mobilenet-nobn-fp32.param --bin=mobilenet-nobn-fp32.bin --images=images/ --output=mobilenet-nobn.table --mean=104,117,123 --norm=0.017,0.017,0.017 --size=224,224 --thread=2
```

### 3. Quantization

```
./ncnn2int8 mobilenet-nobn-fp32.param mobilenet-nobn-fp32.bin mobilenet-int8.param mobilenet-int8.bin mobilenet-nobn.table
```

## (option) validate quantization result

### validate classification

Example with mnist, need two steps.

Suppose data structure like that:

```
mnist
©À©¤0
©¦  ©À©¤0_0.jpg
©¦  ©À©¤0_1.jpg
©¦   ...
©À©¤1
©À©¤2
©À©¤3
©À©¤4
©À©¤5
©À©¤6
 ...
```

#### 1. generate image annotation and label

in single label of one image, it has a script to generate annotation.txt and label.txt

```
python generate_annotation.py mnist/ output/
```

Example of output/annotaion.txt

```
file1,label1
file2,label2
file3,label1
...
```

Example of output/label.txt

```
label1
label2
...
```

#### 2. validate quantation result

```
./validate_classify \
--param=mobilenet-nobn-fp32.param \
--bin=mobilenet-nobn-fp32.bin \
--annotation=output/annotation.txt \
--label=output/label.txt \
--input_name=data \
--output_name=prob \
--mean=0,0,0 \
--norm=1,1,1 \
--size=224,224 
```


