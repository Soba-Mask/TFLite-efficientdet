import glob
import cv2
import tensorflow as tf
import sys

import numpy as np
#from tflite_runtime.interpreter import Interpreter



def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    resized_img = cv2.resize(original_image, input_size)

    return resized_img.astype(np.uint8), original_image


def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    # Feed the input image to the model
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all outputs from the model
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    
    return results


def run_odt_and_draw_results(image_path, interpreter, threshold=0.5):
    """Run object detection on the input image and draw the detection results"""
    # Load the input shape required by the model
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Load the input image and preprocess it
    preprocessed_image, original_image = preprocess_image(
        image_path,
        (input_height, input_width)
    )

    # Run object detection on the input image
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    return results


def main():
    model_path = './efficientdet-lite2_reg.tflite'

    DETECTION_THRESHOLD = 0.3
    num_threads = int(sys.argv[1])
   
    image_file = './bed2.jpg'
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path,  num_threads=num_threads)
    #interpreter = Interpreter(model_path=model_path, num_threads=1)
    interpreter.allocate_tensors()

    
    # Run inference
    results = run_odt_and_draw_results(
        image_file,
        interpreter,
        threshold=DETECTION_THRESHOLD
    )
    print(results)

if __name__ == '__main__':
    main()