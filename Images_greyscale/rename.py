import os

dir_path = os.path.dirname(os.path.realpath(__file__))

for filename in os.listdir(dir_path):
    name_list = filename.split('.')
    if name_list[1] == "py":
        continue
    os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, f"{name_list[0]}-grey.{name_list[1]}"))