import convertor as caffeaicore
caffeaicore.process_kernels(caffeaicore.conv_fw_bw_op(64, 1, 20, 28, 28, 24, 24, True, 5, 5, 0, 0, 1, 1, 'float16'), 'conv1')
caffeaicore.process_kernels(caffeaicore.conv_fw_bw_op(64, 20, 50, 12, 12, 8, 8, True, 5, 5, 0, 0, 1, 1, 'float16'), 'conv2')

caffeaicore.gen_net_prototxt('caffe/examples/mnist/lenet_train_test.prototxt')
