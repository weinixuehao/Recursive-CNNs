''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import cv2
import numpy as np
import glob
import evaluation
import os
import shutil
import time


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", default="example_imgs", help="Document image folder")
    parser.add_argument('--model-type', default="resnet",
                    help='model type to be used. Example : resnet32, resnet20, densenet, test')
    parser.add_argument("-o", "--output", default="example_imgs/output", help="The folder to store results")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.85", type=float)
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        default="../cornerModelWell")
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        default="../documentModelWell")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_processor()

    corners_extractor = evaluation.corner_extractor.GetCorners(args.documentModel, args.model_type)
    corner_refiner = evaluation.corner_refiner.corner_finder(args.cornerModel, args.model_type)
    now_date = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
    output_dir = f"{args.output}_{now_date}"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    imgPaths = glob.glob(f"{args.images}/*.jpg")
    for imgPath in imgPaths:
        img = cv2.imread(imgPath)
        oImg = img
        e1 = cv2.getTickCount()
        extracted_corners = corners_extractor.get(oImg)
        corner_address = []
        # Refine the detected corners using corner refiner
        image_name = 0
        for corner in extracted_corners:
            image_name += 1
            corner_img = corner[0]
            refined_corner = np.array(corner_refiner.get_location(corner_img, args.retainFactor))

            # Converting from local co-ordinate to global co-ordinates of the image
            refined_corner[0] += corner[1]
            refined_corner[1] += corner[2]

            # Final results
            corner_address.append(refined_corner)
        e2 = cv2.getTickCount()
        print(f"Took time:{(e2 - e1)/ cv2.getTickFrequency()}")

        for a in range(0, len(extracted_corners)):
            cv2.line(oImg, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]), (255, 0, 0), 4)
        filename = os.path.basename(imgPath)
        cv2.imwrite(f"{output_dir}/{filename}", oImg)
