import numpy as np
from os import listdir
from os.path import isfile, join
import os
import json

# 데이터 경로 설정
cwd = os.getcwd()
data_path = join(cwd, 'sea')  # 'sea' 폴더 경로
savedir = './'  # JSON 저장 디렉터리
dataset_list = ['base', 'val', 'novel']  # 데이터셋 이름들

# 모든 클래스(디렉터리) 불러오기
classes = [d for d in listdir(data_path) if not isfile(join(data_path, d))]

if len(classes) == 0:
    print(f"Error: '{data_path}' 경로에 클래스 폴더가 존재하지 않습니다.")
    exit()

# 디버깅: 클래스 이름 확인
print("클래스 이름 리스트:", classes)

# JSON 생성
for dataset in dataset_list:
    image_names = []
    image_labels = []

    for label, class_name in enumerate(classes):
        class_path = join(data_path, class_name)
        file_list = [f for f in listdir(class_path) if isfile(join(class_path, f)) and f.endswith('.jpg')]
        file_list.sort()

        # 데이터셋별로 나누기
        if dataset == 'base':
            subset = file_list[:len(file_list) // 2]
        elif dataset == 'val':
            subset = file_list[len(file_list) // 2:3 * len(file_list) // 4]
        elif dataset == 'novel':
            subset = file_list[3 * len(file_list) // 4:]
        else:
            subset = []

        for file_name in subset:
            image_names.append(join(class_path, file_name))
            image_labels.append(label)

    # JSON 저장
    json_data = {
        "label_names": classes,  # 클래스 이름 포함
        "image_names": image_names,
        "image_labels": image_labels
    }

    with open(join(savedir, f"{dataset}.json"), "w") as fo:
        json.dump(json_data, fo, indent=4)

    print(f"{dataset} - JSON 파일 생성 완료")
