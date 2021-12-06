import glob
import cv2
import os
import shutil
import time

def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="dataput")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    return parser.parse_args()

if __name__ == "__main__":
    args = args_processor()
    imgpaths = glob.glob(f"{args.input_dir}/*.jpg") + glob.glob(f"{args.input_dir}/*.png")
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.mkdir(args.output_dir)
    for imgpath in imgpaths:
        csv_path = imgpath.split(".")[0]+".csv"
        if os.path.isfile(csv_path) == False:
            continue
        img = cv2.imread(imgpath)
        now = time.time()
        # dt_string = now.strftime("%Y%m%d_%H%M%S")
        out_imgpath = f"{args.output_dir}/IMG_{now}.jpg"
        shutil.copy(csv_path, f"{out_imgpath}.csv")
        cv2.imwrite(out_imgpath, img)
