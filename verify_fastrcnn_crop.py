import cv2
import numpy as np
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils_crop as vis_util


class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = 'D:\\tensorflow-model\\research\\object_detection\\ssd_model_ctn\\graph_fastrcnn\\frozen_inference_graph.pb'
        self.PATH_TO_LABELS = 'D:\\tensorflow-model\\research\\object_detection\\ssd_model_ctn\\ctn_label_map.pbtxt'
        self.NUM_CLASSES = 2
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image, filefullname):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8, 
                    filename=filefullname)
        
        # cropped_image = tf.image.crop_and_resize(image, boxes, [300, 300], [2])
        # cropped_image.save('D:\\tensorflow-model\\research\\object_detection\\ssd_model_ctn\\JPEGImages_test\\test.jpg')

        # cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        # cv2.imshow("detection", image)
        # cv2.waitKey(0)

if __name__ == '__main__':

    detecotr = TOD()
    for dirpath, dirnames, filenames in os.walk("D:\\tensorflow-model\\research\\object_detection\\ssd_model_ctn\\JPEGImages"): 
        for filename in filenames: 
            print(os.path.join(dirpath, filename))
            image = cv2.imread(os.path.join(dirpath, filename))
            detecotr.detect(image, filename)
    # image = cv2.imread('D:\\tensorflow-model\\research\\object_detection\\ssd_model_ctn\\JPEGImages\\ACSU7010422_45G1.JPG')
    # filefullname = (os.path.splitext('D:\\tensorflow-model\\research\\object_detection\\ssd_model_ctn\\JPEGImages\\ACSU7010422_45G1.JPG')[0])
    # image = cv2.imread('D:\\tensorflow-model\\research\object_detection\\ssd_model_yaban\\IMG_1108.JPG')
    # detecotr = TOD()
    # detecotr.detect(image, filefullname)