import onnx
import tensorflow as tf
import numpy as np
from onnx import numpy_helper
#from backbone_pnet import inference 
from FASDensenet_tf import DenseNet
import cv2

onnx_model_file = 'weights/best_ubuntu.onnx'
onnx_model = onnx.load(onnx_model_file)

#load onnx model
onnx_graph = onnx_model.graph
o_inputs = onnx_graph.input
o_outputs = onnx_graph.output
o_nodes = onnx_graph.node
o_init = onnx_graph.initializer

#convert initializer to dict
o_params = {}
for item in o_init:
    param = numpy_helper.to_array(item)
    o_params[item.name] = param
    # print(item.name)

print("o_params", len(o_params))
#for i in range(len(o_nodes)):
#    print(o_nodes[i].name)

params_name = ['features.conv1.weight', 'features.conv1.bias', 'features.conv2.weight', 'features.conv2.bias', 'features.conv3.weight', 'features.conv3.bias', \
               'conv4_1.weight', 'conv4_1.bias', 'conv4_2.weight', 'conv4_2.bias', 'onnx::PRelu_30', 'onnx::PRelu_31', 'onnx::PRelu_32']

PRINT_NODES = False
#PRINT_NODES = True

def predict(pred, output_name):
    #tf.identity(pred, output_name)
    tf.identity(tf.nn.softmax(pred), output_name)
    #tf.identity(tf.nn.softmax(face), output_name[1])  # use identity op to name the last layer
    #return tf.nn.softmax(face)
    #return tf.nn.softmax(pred)
    return (pred)

"""
def predict(outputs, output_name):
    #tf.identity(output, output_name[0])
    tf.identity(tf.nn.softmax(outputs), output_name)  # use identity op to name the last layer
    return tf.nn.softmax(outputs)
"""
"""
def preprocess(images):
    return (2.0 * images) - 1.0
"""
def preprocess(images):
    return (1.0 * images) - 0.0
    
output_pb_path = 'weights/best_ubuntu.pb'
output_name = ['output_cls', 'output_box', 'output_kpts']

# prepare input tensor
images = tf.placeholder(dtype=tf.uint8, shape=[1, 640, 640, 3], name='image_tensor')
img = tf.to_float(images) * (1.0 / 255.0)
preprocess_img = preprocess(img)

# inference
embedding_size = 128
growth_rate = 16 #24 #16 #32
block_config = (4, 6, 2)
num_init_features = 32 #32 #64
num_classes = 2
img_channel = 3
bn_size = 4

#logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=False).model
logits = DenseNet(x=preprocess_img, embedding_size=embedding_size, num_classes=num_classes, img_channel=img_channel,
                  growth_rate=growth_rate, block_config=block_config, num_init_features=num_init_features,
                  bn_size=bn_size).model

# post process, usually c code
##predict(bbox, face, output_name)
#result = predict(face, output_name)
result = predict(logits, output_name)

# remove un relative nodes
graph = tf.get_default_graph()
config = tf.ConfigProto()
sess = tf.Session(graph=graph, config=config)

sess.run(tf.global_variables_initializer())  # initialize the weights for example, cause it did't be trained

if PRINT_NODES:
    for n in graph.as_graph_def().node:
        print(n.name)
        # print(n)
        #if n.name == 'conv5/weight':
        #    print("*******************")
        #print(n)
        # if n.name == "conv4/weight:0":
        #     type(n)
        #     dir(n)

# with tf.variable_scope('conv5', reuse=True):
#     conv5_weights = tf.get_variable(name='weight')
#     print(conv5_weights)
#     print('*************')
#     dir(conv5_weights)
#     print('*************')
#for op in sess.graph.get_operations():
#    print(op.name)

# for op in sess.graph.get_operations():
#     print(op.name)

for variable in tf.trainable_variables():
    #print(variable.name)
    
    v_name = variable.name
    onnx_variable_name = v_name.replace('/', '.')
    onnx_variable_name = onnx_variable_name[:-2]
    name_list = v_name.split('/')
    tf_variable_name = name_list[-1]
    pos = v_name.find(tf_variable_name)
    tf_variable_name = tf_variable_name[:-2]
    tf_scope = v_name[:pos-1]
    #print(onnx_variable_name)
    #print("tf_scope: {} tf_variable_name: {}".format(tf_scope, tf_variable_name))
    with tf.variable_scope(tf_scope, reuse=True):
        tf_variable = tf.get_variable(tf_variable_name)
        # ndim = tf.rank(tf_variable)
        # print(ndim)
        #print(tf_variable.shape, tf.rank(tf_variable))
        #print(len(tf_variable.shape))
        if(len(tf_variable.shape) == 4):
            #print('============')
            o_param = o_params[onnx_variable_name]
            #print(tf_variable.shape)
            #print(o_param.shape)
            param_tensor = tf.convert_to_tensor(o_param.transpose(2, 3, 1, 0))
            update_op = tf.assign(tf_variable, param_tensor)
            sess.run(update_op)
        elif(len(tf_variable.shape) == 1):
            #print('*********')
            o_param = o_params[onnx_variable_name]
            param_tensor = tf.convert_to_tensor(o_param)
            update_op = tf.assign(tf_variable, param_tensor)
            sess.run(update_op)  

#image_sample = np.ones((1, 12, 12, 3), dtype=np.uint8)
#image_sample = cv2.imread('img.jpg')
#image_sample = cv2.cvtColor(image_sample, cv2.COLOR_BGR2RGB)
image_sample = np.fromfile('onnx_input.bin', dtype=np.uint8)
# image_sample = np.fromfile('img.rgb', dtype=np.uint8)
image_sample = image_sample.reshape(128, 128, 3)
image_sample = np.expand_dims(image_sample, axis=0)
#print(sess.run(result, feed_dict={images:image_sample}))
print(sess.run(logits, feed_dict={images:image_sample}))

# out = sess.graph.get_tensor_by_name('model/conv1/conv/Conv2D:0') 
# #feed_dict = {inp: test}
# output = sess.run(out, feed_dict={images:image_sample})
# print('output', output)
# print(output.shape)
    
keep_nodes = [output_name]
input_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, graph.as_graph_def(),
    output_node_names=keep_nodes
)

output_graph_def = tf.graph_util.remove_training_nodes(
    input_graph_def,
    protected_nodes=keep_nodes
)

"""
keep_nodes = output_name
input_graph_def = tf.graph_util.convert_variables_to_constants(
    sess, graph.as_graph_def(),
    output_node_names=keep_nodes
 )

output_graph_def = tf.graph_util.remove_training_nodes(
    input_graph_def,
    protected_nodes=keep_nodes
)
"""

# write out pb
with tf.gfile.GFile(output_pb_path, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
print('%d ops in the final graph.' % len(output_graph_def.node))
print(output_pb_path, 'is created!')  

# for n in output_graph_def.node:
#     print(n.name, n.op)
#     #print(n)
#     #if n.name == 'conv5/weight':
#     #    print("*******************")
#     #print(n)
#     # if n.name == "conv4/weight:0":
#     #     type(n)
#     #     dir(n)  
