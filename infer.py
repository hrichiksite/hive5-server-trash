import tensorflow as tf
import numpy as np
from PIL import Image
from six import BytesIO
import cv2 


def normalize_image(imagea,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
    """Normalizes the image to zero mean and unit variance."""
    with tf.name_scope('normalize_image'):
        imageb = tf.image.convert_image_dtype(imagea, dtype=tf.float32)
        offset = tf.constant(offset)
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        imageb -= offset

        scale = tf.constant(scale)
        scale = tf.expand_dims(scale, axis=0)
        scale = tf.expand_dims(scale, axis=0)
        imageb /= scale
        return imageb


model = tf.saved_model.load(
    "/workspaces/codespaces-flask/saved_model/saved_model")


# Get the model's input and output signature
model_fn = model.signatures["serving_default"]
print(model_fn.inputs)

# Prepare the input data
# Resize the image to the input size
height = model_fn.structured_input_signature[1]['inputs'].shape[1]
width = model_fn.structured_input_signature[1]['inputs'].shape[2]
input_size = (height, width)


def load_image_into_numpy_array(path):
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    imagec = Image.open(BytesIO(image_data))

    (im_width, im_height) = imagec.size
    return np.array(imagec.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)

# make a function to use the model for inference with the input image path as parameter


def predict_image(image_path):

    # convert image to np array with color channels natively
    image = load_image_into_numpy_array(image_path)
    # Load the model

    # apply pre-processing functions which were applied during training the model
    image = cv2.resize(image[0], input_size[::-1],
                       interpolation=cv2.INTER_AREA)
    # resize the image with resizebilenear
    # image = tf.image.resize(image, input_size, method=tf.image.ResizeMethod.BILINEAR)

    image = normalize_image(image)
    # covert tensor to image for saving the image
    imagesave = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    # imagesave = tf.squeeze(imagesave, axis=0)
    imagesave = tf.image.encode_jpeg(imagesave)
    tf.io.write_file('normalpy.jpg', imagesave)

    # Expand the dimensions of the image
    image = tf.expand_dims(image, axis=0)

    # save the tensor as image file

    # get the shape of the input
    print(image.shape)

    # image = tf.squeeze(image, axis=0)

    # Run inference
    results = model_fn(inputs=image)

    # Get the output
    result = {key: value.numpy() for key, value in results.items()}
    print(result.keys())

    # get the class names from the output
    class_names = (result['detection_classes'][0]).astype(int)
    print(class_names)

    print(result)

    return result

    # save the output to a file
    # with open('output.txt', 'w') as f:
    # f.write(str(result))
