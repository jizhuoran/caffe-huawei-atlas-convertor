# caffe-huawei-atlas-convertor

This is a converter to transfer Caffe-CPU prototxt to HUAWEI Atlas prototxt. It generates the Atlas AICORE kernel codes, compiler the code, and append the kernel info to prototxt.

## Basic Logic

It relies on Caffe-CPU to figure out what AICORE kernel codes are needed. The "name" of these AICORE kernel codes is automatically collected into a python file.

The python file also contains the necessary code to compile the code. It compiles the code first and then collects the kernels' information, such as their names and num_cores. It then appends the information to the original caffe prototxt, which is used by caffe-atlas.

## Compile

HUAWEI only provides binary files for its compiler, which is for Ubuntu 18.04. This converter also runs on Ubuntu 18.04 only.

Caffe can be compiled follow [link](https://caffe.berkeleyvision.org/install_apt.html), but you must compiler the one in this repo!


## Usage Example (lenet on mnist)

```
cd REPO_HOME
cd caffe
./data/mnist/get_mnist.sh
./examples/mnist/create_mnist.sh
./examples/mnist/train_lenet.sh
cd ..
```

It generates a python file `gen_compile_net.py`. Some path needs to be specified, but you can just run
```
sh run.sh
```

A directory named 'kernel_meta' is generated. It contains the `*.o` kernel codes and the `aicore_YOU_PROTOTXT_NAME.prototxt` file.

You can then use caffe-atlas and these files for training. 




