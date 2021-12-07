import glob
import cv2
import os
import shutil
import time
import random


def args_processor():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="dataput")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    return parser.parse_args()


def process(imgpaths, out):
    for imgpath in imgpaths:
        csv_path = imgpath.split(".")[0] + ".csv"
        if os.path.isfile(csv_path) == False:
            continue
        img = cv2.imread(imgpath)
        now = time.time()
        # dt_string = now.strftime("%Y%m%d_%H%M%S")
        out_imgpath = f"{out}/IMG_{now}.jpg"
        shutil.copy(csv_path, f"{out_imgpath}.csv")
        cv2.imwrite(out_imgpath, img)


if __name__ == "__main__":
    args = args_processor()
    imgpaths = glob.glob(f"{args.input_dir}/*.jpg") + glob.glob(
        f"{args.input_dir}/*.png"
    )
    train_dataset_out = f"{args.output_dir}/train"
    test_dataset_out = f"{args.output_dir}/test"
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.mkdir(args.output_dir)
    os.mkdir(train_dataset_out)
    os.mkdir(test_dataset_out)

    imgpaths_num = len(imgpaths)
    test_num = int(imgpaths_num * 0.2)
    test_imgpaths = imgpaths[0:test_num]
    train_imgpaths = imgpaths[test_num:imgpaths_num]
    process(train_imgpaths, train_dataset_out)
    process(test_imgpaths, test_dataset_out)
