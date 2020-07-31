import caffe_pb2 as caffe_pb2
import google.protobuf.text_format
import json


def gen_net_prototxt(prototxt_path):
    net = caffe_pb2.NetParameter()
    with open('prototxt_path', 'r') as f:
        net = google.protobuf.text_format.Merge(str(f.read()), net)

    for i in range(0, len(net.layer)):
        if(net.layer[i].type == "Convolution"):

            for _ in range(len(net.layer[i].aicorekernel)):
                del net.layer[i].aicorekernel[0]

            with open('kernel_meta/{}.json'.format(net.layer[i].name), 'r') as f:
                kernel_infos = json.load(f)
                
            for info in kernel_infos:    
                kernel_info = caffe_pb2.AicoreKerel()
                with open('kernel_meta/{}.json'.format(net.layer[i].name)) as f:
                    kernel_info.kernelfile = info['kernelfile']
                    kernel_info.kernelname = info['kernelname']
                    kernel_info.block_num = info['block_num']
                net.layer[i].aicorekernel.append(kernel_info)

    new_prototxt_name = 'aicore_' + prototxt_path.split('/')[-1]
    with open('kernel_meta/new_prototxt_name', 'w') as f:
        f.write(str(net) + '\n')

# if True:
#     prototxt_path = '/home/zrji/ascend_generator/examples/mnist/lenet_train_test.prototxt'