import os
import glob


def list_file_numbers(dir_path="data/avec_2014_freeform/Development"):
    dirs = sorted(os.listdir(dir_path))
    max = 0
    sum = 0
    min = 100000000
    median_list = []
    for d in dirs:
        sub_dir_path = os.path.join(dir_path, d)
        files = glob.glob(f"{sub_dir_path}/*.jpg")
        files = sorted(files, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
        frame_num = files[-1].split("/")[-1].split(".")[0].split("_")[-1]
        num = len(files)
        median_list.append(num)
        if num > max:
            max = num
        if num < min:
            min = num
        sum += num
        # print(f"{d}: {num} | {frame_num}")
    print(f"dataset: {dir_path}")
    print(f"Max: {max}: Min: {min}: Avg: {sum/len(dirs)} Total: {sum} Count: {len(dirs)}")
    median_list = sorted(median_list)
    print(f"Median List: {median_list}")
    print(f"Median: {median_list[len(median_list)//2]}")
    print()

if __name__ == "__main__":
    list_file_numbers("data/avec_2014_northwind/Training")
    list_file_numbers("data/avec_2014_northwind/Development")
    list_file_numbers("data/avec_2014_northwind/Testing")