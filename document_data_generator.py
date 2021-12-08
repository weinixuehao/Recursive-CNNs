import os
from tqdm import tqdm

import cv2
import numpy as np
import utils
import dataprocessor

import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_processor():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="Path to data files (Extract images using video_to_image.py first")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    parser.add_argument("-v", "--visualize", help="Draw the point on the corner", default=False, type=bool)
    parser.add_argument("-a", "--augment", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Augment image dataset")
    parser.add_argument("--dataset", default="smartdoc", help="'smartdoc' or 'selfcollected' dataset")
    return parser.parse_args()


if __name__ == '__main__':
    if __name__ == '__main__':
        args = args_processor()
        input_directory = args.input_dir
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        import csv


        # Dataset iterator
        if args.dataset == "smartdoc":
            dataset_test = dataprocessor.dataset.SmartDocDirectories(input_directory)
        elif args.dataset == "selfcollected":
            dataset_test = dataprocessor.dataset.SelfCollectedDataset(input_directory)
        else:
            print("Incorrect dataset type; please choose between smartdoc or selfcollected")
            assert (False)
        with open(os.path.join(args.output_dir, 'gt.csv'), 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # Counter for file naming
            counter = 0
            for data_elem in tqdm(dataset_test.myData):

                img_path = data_elem[0]
                target = data_elem[1].reshape((4, 2))
                img = cv2.imread(img_path)

                if args.dataset == "selfcollected":
                    target = target / (img.shape[1], img.shape[0])
                    target = target * (1920, 1920)
                    img = cv2.resize(img, (1920, 1920))

                corner_cords = target
                angles = [0, 271, 90] if args.augment else [0]
                random_crops = [0, 16] if args.augment else [0]
                for angle in angles:
                    img_rotate, gt_rotate = utils.utils.rotate(img, corner_cords, angle)
                    for random_crop in random_crops:
                        counter += 1
                        f_name = str(counter).zfill(8)

                        img_crop, gt_crop = utils.utils.random_crop(img_rotate, gt_rotate)
                        mah_size = img_crop.shape
                        img_crop = cv2.resize(img_crop, (64, 64))
                        gt_crop = np.array(gt_crop)

                        if (args.visualize):
                            no=0
                            for a in range(0,4):
                                no+=1
                                cv2.circle(img_crop, tuple(((gt_crop[a]*64).astype(int))), 2,(255-no*60,no*60,0),9)
                        # # cv2.imwrite("asda.jpg", img)

                        cv2.imwrite(os.path.join(args.output_dir, f_name+".jpg"), img_crop)
                        spamwriter.writerow((f_name+".jpg", tuple(list(gt_crop))))
