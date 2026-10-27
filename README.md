# Motion–Part FSL

## 1. Cài đặt

```bash
git clone --recurse-submodules https://github.com/trantrongnb/CSP_Thu6.git
cd CSP_Thu6

conda create -n motion_part_fsl python=3.12 -y
conda activate motion_part_fsl
python -m pip install -e '.[dev,landmarks]'
```

Nếu đã clone trước đó, chạy `git submodule update --init --recursive`.
Các lệnh bên dưới chạy từ gốc repository; trong workspace hiện tại là
`~/CSP/motion_part_fsl`. Đường dẫn YAML được tính từ thư mục chạy lệnh.
Preprocessing cần `mediapipe==0.10.21`, được cài cùng extra `landmarks`.

## 2. Kiểm tra nhanh

```bash
python -m pytest -q
mpfsl-smoke --output outputs/smoke_new --steps 40
```

Smoke dùng dữ liệu tổng hợp để kiểm tra train, checkpoint và evaluate.
Chọn thư mục `--output` mới cho mỗi lần chạy.

## 3. Chuẩn bị dữ liệu

Đặt metadata theo đường dẫn trong cấu hình:

| Dữ liệu | Metadata | Cache |
|---|---|---|
| WLASL | `data/wlasl_60_20_20/metadata.csv` | `data/wlasl_rgb_landmarks_t8` |
| VSL | `data/vsl/metadata.csv` | `data/vsl_rgb_landmarks_t8` |

Định dạng CSV:

```csv
video_id,path,gloss,class_id,language,signer_id,split,is_mirrored
w001,/data/videos/a.mp4,HELLO,1,ASL,,train,false
w002,/data/videos/b.mp4,THANKS,2,ASL,,val,false
w003,/data/videos/c.mp4,PLEASE,3,ASL,,test,false
```

Sửa `path` theo vị trí video; đường dẫn tương đối tính từ thư mục chứa metadata.
`split` nhận `train`, `val`, `test`; các lớp không được trùng giữa các split.
`video_id` và đường dẫn video phải duy nhất. Để trống `signer_id` nếu chưa biết;
đặt `is_mirrored=true` nếu cần unmirror video trước preprocessing.

Mỗi split cần ít nhất 5 lớp. Với 5 query/lớp, mỗi lớp cần ít nhất 6 video cho
1-shot hoặc 10 video cho 5-shot. Chuẩn bị metadata riêng cho WLASL và VSL.
Video, metadata và cache cần được chuẩn bị cục bộ sau khi clone.

## 4. Preprocessing

Xử lý thử 20 video WLASL:

```bash
mpfsl-preprocess \
  --metadata data/wlasl_60_20_20/metadata.csv \
  --output data/wlasl_rgb_landmarks_t8 \
  --frames 8 --image-size 224 --limit 20
```

Xem landmark; thay `VIDEO_ID` bằng ID trong metadata:

```bash
python scripts/visualize_cache.py \
  --cache data/wlasl_rgb_landmarks_t8/VIDEO_ID.npz \
  --output outputs/debug/VIDEO_ID.mp4
```

Xử lý toàn bộ và kiểm tra cache:

```bash
mpfsl-preprocess \
  --metadata data/wlasl_60_20_20/metadata.csv \
  --output data/wlasl_rgb_landmarks_t8 \
  --frames 8 --image-size 224

python scripts/audit_data.py \
  --metadata data/wlasl_60_20_20/metadata.csv \
  --cache-dir data/wlasl_rgb_landmarks_t8 --verify-source \
  --output outputs/wlasl_cache_audit.json
```

VSL dùng metadata và cache tương ứng trong bảng trên. Khi đổi số frame, kích
thước ảnh hoặc detector, chọn cache mới và cập nhật YAML.

## 5. Huấn luyện

```bash
# WLASL
mpfsl-train --config configs/wlasl_5w1s.yaml --device cuda
mpfsl-train --config configs/wlasl_5w5s.yaml --device cuda

# VSL
mpfsl-train --config configs/vsl_5w1s.yaml --device cuda
mpfsl-train --config configs/vsl_5w5s.yaml --device cuda
```

Kiểm tra `data.metadata`, `data.cache_dir`, `output_dir` trong YAML trước khi
chạy. Thay `--device cuda` bằng `--device cpu` khi dùng môi trường PyTorch CPU.
Checkpoint `best.pt` và `last.pt` được lưu trong `output_dir`.

Tiếp tục huấn luyện với cùng cấu hình và output:

```bash
mpfsl-train --config configs/wlasl_5w1s.yaml --device cuda \
  --resume outputs/WLASL_A5_5w1s_seed42/last.pt
```

Chạy thí nghiệm mới bằng thư mục output khác:

```bash
mpfsl-train --config configs/wlasl_5w1s.yaml --device cuda \
  --output outputs/WLASL_A5_5w1s_run2
```

## 6. Đánh giá

Tạo test episodes một lần và dùng lại cho các thí nghiệm:

```bash
mpfsl-episodes \
  --metadata data/wlasl_60_20_20/metadata.csv \
  --cache-dir data/wlasl_rgb_landmarks_t8 --split test \
  --n-way 5 --k-shot 1 --queries-per-class 5 --count 1000 --seed 123 \
  --output data/episodes/wlasl_test_5w1s_seed123.json

mpfsl-evaluate \
  --checkpoint outputs/WLASL_A5_5w1s_seed42/best.pt \
  --episodes data/episodes/wlasl_test_5w1s_seed123.json \
  --output outputs/WLASL_A5_5w1s_seed42/test.json --device cuda
```

Với 5-shot, đặt `--k-shot 5` và dùng tên file episode, checkpoint, output tương
ứng. VSL dùng metadata và cache riêng. Kết quả được ghi vào JSON tại `--output`.

## 7. Chạy ablation

Chạy lần lượt A0–A5 trên cùng test episodes:

```bash
python scripts/ablate.py \
  --config configs/motion_part_full.yaml \
  --episodes data/episodes/wlasl_test_5w1s_seed123.json \
  --output outputs/ablations_wlasl_1shot \
  --variants A0 A1 A2 A3 A4 A5 --device cuda
```

Thêm `A6 A7` vào `--variants` nếu cần. Bảng kết quả nằm tại
`outputs/ablations_wlasl_1shot/results.csv`. Dùng output mới cho mỗi lượt chạy.

## 8. Launcher trên server

Với conda `signclip_5090` có sẵn, có thể chạy mà không cài package vào môi
trường chung:

```bash
bash scripts/run_conda.sh check
bash scripts/run_conda.sh smoke --output outputs/smoke_server --steps 40
bash scripts/run_conda.sh train --config configs/wlasl_5w1s.yaml
```

Launcher cũng hỗ trợ `preprocess`, `episodes`, `evaluate`. Preprocessing vẫn cần
môi trường có MediaPipe 0.10.21. Đặt `MPFSL_PYTHON=/path/to/python` để dùng
interpreter khác hoặc `MPFSL_CONDA_ROOT=/path/to/conda` để đổi vị trí conda.

Tra cứu tham số bằng `mpfsl-train --help`, `mpfsl-evaluate --help` hoặc
`mpfsl-preprocess --help`.
