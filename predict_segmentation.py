import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess
from scipy.misc import imsave
import numpy as np
slim = tf.contrib.slim
import argparse

parser = argparse.ArgumentParser(description="Run ENet inference")

parser.add_argument("--weights", type=str, help="Model to use.")
parser.add_argument("--num-classes", type=int, required=True)
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=False, default=None)

args = parser.parse_args()

num_classes = args.num_classes
image_dir = args.input_dir
images_list = sorted(
    [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png') or file.endswith(".jpg") or file.endswith(".jpeg")])

checkpoint_dir = os.path.basename(args.weights)
checkpoint = args.weights

num_initial_blocks = 1
skip_connections = False
stage_two_repeat = 2
'''
#Labels to colours are obtained from here:
https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py

However, the road_marking class is collapsed into the road class in the dataset provided.

Classes:
------------
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
'''
label_to_colours =    {0: [128,128,128],
                     1: [128,0,0],
                     2: [192,192,128],
                     3: [128,64,128],
                     4: [60,40,222],
                     5: [128,128,0],
                     6: [192,128,128],
                     7: [64,64,128],
                     8: [64,0,128],
                     9: [64,64,0],
                     10: [0,128,192],
                     11: [0,0,0]}

label_to_colours[num_classes - 1] = [0, 0, 0]

#Create the photo directory
photo_dir = args.output_dir if args.output_dir is not None else os.path.join(checkpoint_dir, "test_images")
if not os.path.exists(photo_dir):
    os.mkdir(photo_dir)

#Create a function to convert each pixel label to colour.
def grayscale_to_colour(image):
    print 'Converting image...'
    image = image.reshape((360, 480, 1))
    image = np.repeat(image, 3, axis=-1)
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            label = int(image[i][j][0])
            image[i][j] = np.array(label_to_colours[label])

    return image


with tf.Graph().as_default() as graph:
    images_tensor = tf.train.string_input_producer(images_list, shuffle=False)
    reader = tf.WholeFileReader()
    key, image_tensor = reader.read(images_tensor)
    image = tf.image.decode_png(image_tensor, channels=3)
    # image = tf.image.resize_image_with_crop_or_pad(image, 360, 480)
    # image = tf.cast(image, tf.float32)
    image = preprocess(image)
    images = tf.train.batch([image], batch_size = 10, allow_smaller_final_batch=True)

    #Create the model inference
    with slim.arg_scope(ENet_arg_scope()):
        logits, probabilities = ENet(images,
                                     num_classes=num_classes,
                                     batch_size=10,
                                     is_training=True,
                                     reuse=None,
                                     num_initial_blocks=num_initial_blocks,
                                     stage_two_repeat=stage_two_repeat,
                                     skip_connections=skip_connections)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint)

    predictions = tf.argmax(probabilities, -1)
    predictions = tf.cast(predictions, tf.float32)
    print 'HERE', predictions.get_shape()

    sv = tf.train.Supervisor(logdir=None, init_fn=restore_fn)
    
    with sv.managed_session() as sess:

        for i in xrange(len(images_list) / 10 + 1):
            segmentations = sess.run(predictions)
            # print segmentations.shape

            for j in xrange(segmentations.shape[0]):
                #Stop at the 233rd image as it's repeated
                if i*10 + j == 223:
                    break

                converted_image = grayscale_to_colour(segmentations[j])
                print 'Saving image %s/%s' %(i*10 + j, len(images_list))
                try:
                    plt.axis('off')
                    plt.imshow(converted_image)
                except:
                    # pass error for cmd line usage without display
                    pass
                input_img = cv2.imread(images_list[i * 10 + j])
                input_img = cv2.resize(input_img, tuple(list(converted_image.shape[:2])[::-1]), interpolation=cv2.INTER_LINEAR)
                imsave(os.path.join(photo_dir, "image_%s.png" %(i*10 + j)), np.hstack([input_img, converted_image]))
                # plt.show()