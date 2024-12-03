import os
import shutil

# 이미지가 들어있는 폴더 경로
data_path = './sea'
if not os.path.exists(data_path):
    print(f"Error: '{data_path}' 경로가 존재하지 않습니다.")
    exit()

# 모든 이미지 파일 읽기
file_list = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.jpg')]

if len(file_list) == 0:
    print(f"Error: '{data_path}' 경로에 이미지 파일이 존재하지 않습니다.")
    exit()

# 클래스별 폴더로 정리
for file_name in file_list:
    # 클래스 이름 추출 (첫 단어만 가져오기)
    class_name = file_name.split('_')[0]  # 첫 번째 단어를 클래스 이름으로 사용

    # 클래스 폴더 경로
    class_path = os.path.join(data_path, class_name)

    # 클래스 폴더가 없으면 생성
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    # 이미지 파일 이동
    src_path = os.path.join(data_path, file_name)
    dst_path = os.path.join(class_path, file_name)
    shutil.move(src_path, dst_path)

print("이미지를 클래스별로 정리 완료!")
