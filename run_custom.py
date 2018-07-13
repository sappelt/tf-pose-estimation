import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
import math
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def debug_parts(humans):
    centers = {}
    image_w = 432
    image_h = 368

    for human in humans:
        # draw point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            logger.info('Body Part: ' + str(common.CocoPart(i)))
            coco_part = common.CocoPart(i)
            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            #cv2.circle(npimg, center, 3, common.CocoColors[i], thickness=3, lineType=8, shift=0)

            logger.info(body_part)

        # draw line
        for pair_order, pair in enumerate(common.CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            body_part1 = human.body_parts[pair[0]]
            body_part2 = human.body_parts[pair[1]]

            logger.info(pair)
            logger.info("CocoPair" + str(common.CocoPart(pair[0])) + " - " + str(common.CocoPart(pair[1])))
            logger.info("Points: " + str(body_part1) + " - " + str(body_part2))

            angle_radians = math.atan2(body_part1.y-body_part2.y, body_part1.x-body_part2.x)
            angle_degrees = math.degrees(angle_radians)
            logger.info("Angle: " + str(angle_degrees))

            # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
            # cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/arms/minus45.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    debug_parts(humans)
    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    #plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    #plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    plt.show()
