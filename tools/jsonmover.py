import os
import shutil
import argparse

def move_json_files(src_root, dst_folder, start_idx, end_idx):
    os.makedirs(dst_folder, exist_ok=True)
    moved_count = 0

    for i in range(start_idx, end_idx + 1):
        folder_name = f"sav_{i:03d}"
        src_path = os.path.join(src_root, folder_name, "sav_train", folder_name)
        
        if not os.path.exists(src_path):
            print(f"경로 없음: {src_path}")
            continue
        
        for file_name in os.listdir(src_path):
            if file_name.endswith(".json"):
                src_file = os.path.join(src_path, file_name)
                dst_file = os.path.join(dst_folder, file_name)

                # 덮어쓰기 방지: 같은 이름 있으면 뒤에 번호 붙임
                base, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(dst_file):
                    dst_file = os.path.join(dst_folder, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.move(src_file, dst_file)
                moved_count += 1

    print(f"총 {moved_count} 개 JSON 파일을 {dst_folder} 로 이동 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move JSON files from sav_XXX folders to a flat folder")
    parser.add_argument("--src_root", type=str, default="/data/SA-V", help="원본 루트 폴더 (default: /data/SA-V)")
    parser.add_argument("--dst_folder", type=str, default="/data/SA-V-json-flat_val", help="대상 폴더 (default: /data/SA-V-json-flat_val)")
    parser.add_argument("--start", type=int, default=49, help="시작 인덱스 ")
    parser.add_argument("--end", type=int, default=53, help="끝 인덱스 ")

    args = parser.parse_args()
    move_json_files(args.src_root, args.dst_folder, args.start, args.end)
