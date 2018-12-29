from scipy import misc
import tensorflow as tf
import detect_face
import cv2
import matplotlib.pyplot as plt
import numpy as np

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)


def detection(image):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # detect with RGB image
    h, w = image.shape[:2]
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1:
        print("can't detect face in the frame")
        return None
    print("num %d faces detected"% len(bounding_boxes))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(len(bounding_boxes)):
        det = np.squeeze(bounding_boxes[i, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        # x1, y1, x2, y2
        margin = 0
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, w)
        bb[3] = np.minimum(det[3] + margin / 2, h)
        cv2.rectangle(bgr, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2, 8, 0)
    cv2.imshow("detected faces", bgr)
    return bgr


capture = cv2.VideoCapture(0)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
out = cv2.VideoWriter("D:/深度学习大作业-人脸检测/mtcnn_demo.mp4", cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 15,
                             (np.int(width), np.int(height)), True)
while True:
    ret, frame = capture.read()
    if ret is True:
        frame = cv2.flip(frame, 1)
       # cv2.imshow("frame", frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detection(rgb)

        out.write(result)
        c = cv2.waitKey(10)
        if c == 27:
            break
    else:
        break

cv2.destroyAllWindows()