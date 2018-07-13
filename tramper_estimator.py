import math
import logging
from tf_pose import common

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class TramperEstimator:

    @staticmethod
    def is_indicating_stop(humans):
        for human in humans:

            # check pairs
            for pair_order, pair in enumerate(common.CocoPairsRender):

                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                
                # Right arm indicating stop
                if pair[0] == common.CocoPart.RElbow.value and pair[1] == common.CocoPart.RWrist.value:
                    body_part1 = human.body_parts[pair[0]]
                    body_part2 = human.body_parts[pair[1]]

                    angle_radians = math.atan2(body_part1.y-body_part2.y, body_part1.x-body_part2.x)
                    angle_degrees = math.degrees(angle_radians)
                    logger.info("Angle: " + str(angle_degrees))

                    if angle_degrees >= -70 and angle_degrees <= 50:
                        return True

                # Left arm indicating stop
                if pair[0] == common.CocoPart.LElbow.value and pair[1] == common.CocoPart.LWrist.value:
                    body_part1 = human.body_parts[pair[0]]
                    body_part2 = human.body_parts[pair[1]]

                    angle_radians = math.atan2(body_part2.y-body_part1.y, body_part2.x-body_part1.x)
                    angle_degrees = math.degrees(angle_radians)
                    logger.info("Angle: " + str(angle_degrees))

                    if angle_degrees >= -70 and angle_degrees <= 50:
                        return True
        return False
