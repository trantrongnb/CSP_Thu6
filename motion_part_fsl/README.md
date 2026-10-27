# Motion–Part FSL

Triển khai PyTorch theo [đặc tả của bạn](../Motion_Part_FSL_TEAM_CVPR2025_Chi_Tiet_Cai_Tien.md):
full-frame RGB + landmarks → Motion-Guided Pattern Representation (MGPR) → TEAM
support adaptation → Episode-Adaptive Part Matching (EAPM). Đây là mã nghiên cứu;
chưa xác nhận cải thiện accuracy trên WLASL/VSL.

Thư mục `TEAM/` trước đây được chuyển vào `references/team_original/`, giữ nguyên
mã và lịch sử Git gốc. `../mfpa_fslr/` là baseline flow/prototype cũ, có tên package
và A0–A5 khác với phương pháp này. Cache cũ không chứa RGB/torso/timestamp đầy đủ
cho MGPR nên cần preprocessing mới; không chuyển flow/mouth thành relation tay.

TEAM tham chiếu được theo dõi bằng Git submodule tại commit cố định. Khi clone
repository, dùng `git clone --recurse-submodules` hoặc chạy
`git submodule update --init --recursive` từ thư mục gốc trước khi chạy test.

## Chạy trong workspace

Tất cả lệnh dưới đây chạy từ `CSP/motion_part_fsl`. Đường dẫn YAML tính theo
working directory. Launcher luôn chuyển vào thư mục này và dùng Python conda
`signclip_5090`, không cài đè môi trường chung.
Đặt `MPFSL_PYTHON=/path/to/python` nếu muốn dùng interpreter của môi trường riêng.

```bash
bash scripts/run_conda.sh check
bash scripts/run_conda.sh smoke --output outputs/smoke_new --steps 40
bash scripts/run_conda.sh train --help
```

Smoke dùng RGB/landmark **tổng hợp**, tiny CNN không pretrained, 3-way 1-shot;
overfit một episode và chạy train → validation → checkpoint → test. Chọn output
mới cho mỗi lần chạy. Smoke không đánh giá chất lượng nhận dạng ký hiệu.

Cài trong môi trường riêng đã có PyTorch/torchvision phù hợp:

```bash
python -m pip install -e '.[dev,landmarks]'
python -m pytest -q
mpfsl-smoke --output outputs/smoke_installed
```

Holistic được pin `mediapipe==0.10.21`. Conda `signclip_5090` hiện có 0.10.32, thiếu
API `solutions.holistic`: train dùng cache được, preprocessing Holistic cần môi
trường riêng theo pin trên. `--landmarks none` chỉ dành cho kiểm tra RGB/fallback,
không sinh dữ liệu motion để kiểm chứng MGPR. Launcher hỗ trợ cả `--env manta`;
không mặc định rằng mọi dependency trong môi trường đó đáp ứng pyproject.
Trong lần kiểm tra 27/10/2026, CUDA MPS không phản hồi; kiểm thử đã dùng bản
PyTorch CPU riêng. Chọn môi trường PyTorch CPU nếu cần chạy độc lập với MPS;
chỉ đặt `--device cpu` không khắc phục được mọi lỗi khởi tạo của CUDA runtime.

## Dữ liệu và preprocessing

Metadata CSV cho WLASL hoặc VSL độc lập:

```csv
video_id,path,gloss,class_id,language,signer_id,split,is_mirrored
w001,/data/wlasl/a.mp4,HELLO,1,ASL,,train,false
w002,/data/wlasl/b.mp4,THANKS,2,ASL,,val,false
w003,/data/wlasl/c.mp4,PLEASE,3,ASL,,test,false
```

Ví dụ trên chỉ mô tả schema. Cần ít nhất N lớp mỗi split và K+Q video khác nhau
mỗi lớp. `video_id` và source path không trùng; class ID/gloss không giao nhau
giữa train/val/test. Đường dẫn video tương đối tính từ metadata. Không sampling
with replacement. Signer chưa biết để trống; sampler hiện chưa cưỡng chế signer
separation, nên không gọi kết quả là cross-signer. Một experiment chỉ có một ngôn
ngữ trong metadata; WLASL/VSL dùng metadata, checkpoint và episode files riêng.

Trong workspace đã có metadata WLASL lọc trùng nội dung, split lớp 60/20/20 ở
`../mfpa_fslr/data/wlasl_60_20_20/metadata.csv`; đây là protocol tự xây dựng,
không phải few-shot split chính thức. Các config WLASL dùng lại metadata này.

```bash
# Chạy trong môi trường có Holistic tương thích, sau khi cài package:
mpfsl-preprocess \
  --metadata ../mfpa_fslr/data/wlasl_60_20_20/metadata.csv \
  --output data/wlasl_rgb_landmarks_t8 --frames 8 --image-size 224 --limit 20
python scripts/visualize_cache.py --cache data/wlasl_rgb_landmarks_t8/VIDEO_ID.npz \
  --output outputs/debug/VIDEO_ID.mp4
# Kiểm tra tay trái/phải và ROI trên clip thực, rồi chạy toàn bộ:
mpfsl-preprocess \
  --metadata ../mfpa_fslr/data/wlasl_60_20_20/metadata.csv \
  --output data/wlasl_rgb_landmarks_t8 --frames 8 --image-size 224
python scripts/audit_data.py \
  --metadata ../mfpa_fslr/data/wlasl_60_20_20/metadata.csv \
  --cache-dir data/wlasl_rgb_landmarks_t8 --verify-source \
  --output outputs/wlasl_cache_audit.json
```

Video được unmirror nếu metadata chỉ định, letterbox toàn khung hình (không cắt
mất tay), rồi lấy 8 frame đều. RGB và detector dùng cùng frames/timestamps.
Timestamp ưu tiên thời gian decode, fallback FPS thật; không giả định FPS chung.
Clip ngắn lặp frame cuối với padding mask 0. Cache lưu RGB uint8, hand XY,
detected mask nhị phân, torso reference, frame IDs, timestamp, source/config hash.
Không phát sinh confidence liên tục từ detector chỉ có detected/not-detected.

Origin và scale dùng median midpoint/width hai vai của chính video, giữ reference
ổn định theo thời gian. Nếu không có vai hợp lệ, ghi rõ fallback image-center,
scale=1. Landmarks 2D giữ vị trí tay so với thân; handshape còn có wrist-relative
coordinates. Velocity/acceleration chia timestamp thật, mask mọi cặp/gộp ba frame
không hợp lệ. Relation gồm vector giữa tâm hai tay, distance, vận tốc tương đối,
dot/cross của hướng palm 2D. Không suy diễn contact 3D từ distance 2D.

Cache ghi atomically, tiếp tục được sau gián đoạn và từ chối source/config/mirror
đã đổi. Khi đổi T/image size/detector phải chọn cache mới. Dataset không loại video
chỉ vì mất landmarks: RGB vẫn dùng được, giúp giữ cùng video pool giữa ablations.

## Huấn luyện và đánh giá

```bash
bash scripts/run_conda.sh train --config configs/wlasl_5w1s.yaml
bash scripts/run_conda.sh train --config configs/wlasl_5w5s.yaml
# Sau khi cung cấp và preprocessing VSL độc lập:
bash scripts/run_conda.sh train --config configs/vsl_5w1s.yaml
```

Config mặc định ResNet-50 ImageNet V1, D=256, M=16, T=8, AdamW, 10.000 training
episodes. Đây là điểm khởi tạo, chưa tuning. Backbone fine-tune nhưng BatchNorm
giữ statistics ImageNet, tránh query tác động lên support qua batch statistics.
Frame chunking/checkpointing trong config giúp giảm memory khi train 5-shot.
ViT-B/16 là lựa chọn thêm, dùng torchvision; chưa tuyên bố tương đương weights
timm của TEAM. Tiny CNN chỉ dùng smoke. Weights ImageNet tải ở lần train đầu nếu
chưa có; đánh giá/resume nạp weights từ checkpoint, không tải lại ImageNet.

Loss chính là **CE(logits_pos) + CE(logits_neg)**; dự đoán cộng hai nhánh.
A7 chỉ CE_pos. A6 thêm hard-pair margin loss sau warmup; mining negatives dùng
RGB stop-gradient của support khác nhãn, loss học representation motion–part.
Mặc định A5 không dùng hard loss hoặc KL. Trong inference, forward chỉ nhận
support/query inputs, không nhận query labels. Prototype/adaptation/part weights
được tính lại từ support novel, không gradient update tại test.

Validation episodes được tạo một lần và khóa hash; checkpoint tốt nhất chọn theo
val accuracy. Checkpoint lưu optimizer/scaler/RNG/config/data fingerprints.
Không ghi đè run đã có checkpoint khi thiếu `--resume`:

```bash
bash scripts/run_conda.sh train --config configs/wlasl_5w1s.yaml \
  --resume outputs/WLASL_A5_5w1s_seed42/last.pt
bash scripts/run_conda.sh episodes \
  --metadata ../mfpa_fslr/data/wlasl_60_20_20/metadata.csv \
  --cache-dir data/wlasl_rgb_landmarks_t8 --split test \
  --n-way 5 --k-shot 1 --queries-per-class 5 --count 1000 --seed 123 \
  --output data/episodes/wlasl_test_5w1s_seed123.json
bash scripts/run_conda.sh evaluate \
  --checkpoint outputs/WLASL_A5_5w1s_seed42/best.pt \
  --episodes data/episodes/wlasl_test_5w1s_seed123.json \
  --output outputs/WLASL_A5_5w1s_seed42/test.json
```

Dùng thêm file episodes riêng cho 5-shot, seed khác và VSL. JSON có accuracy,
CI95 theo episode, confusion, part weights/entropy, hashes, parameters, thời gian
model gồm cache I/O, và offline preprocessing cost riêng. Chưa có phép đo FLOPs
hay latency streaming end-to-end. Test không tham gia lựa chọn checkpoint.

## Ablation và tính kế thừa TEAM

| ID | Triển khai |
|---|---|
| A0 | TEAM RGB-only, một part |
| A1 | RGB + concat các motion descriptors, một part |
| A2 | MGPR + uniform weights |
| A3 | Part values có motion, keys RGB-only, uniform weights |
| A4 | Như A3 + EAPM |
| A5 | MGPR đầy đủ + EAPM |
| A6 | A5 + hard-pair loss (mặc định λ=0.05 sau 2 epoch) |
| A7 | A5 bỏ exclusive branch |

```bash
PYTHONPATH=src python scripts/ablate.py --config configs/motion_part_full.yaml \
  --episodes data/episodes/wlasl_test_5w1s_seed123.json \
  --output outputs/ablations_wlasl_1shot --variants A0 A1 A2 A3 A4 A5
```

Runner dùng cùng backbone/D/M/budget/split/episodes. Parameter count khác nhau;
chưa có concat baseline parameter-matched, vì vậy chưa đủ cơ sở kết luận phần
chênh lệch chắc chắn đến từ guidance. `configs/team_rgb.yaml` giữ D=2048/M=60,
sum-token/temperature=1 để kiểm tra head gần cấu hình gốc; không dùng nó thay
cho A0 D=256/M=16 trong bảng so sánh cùng capacity.

TEAM adaptation dùng đúng similarity pos/neg nhân hiệu context của đối thủ,
shared MLP + ReLU **trước** khi trung bình các đối thủ. Mặc định support mean
trước DPM (`support_aggregation: frames`). `tokens` là ablation mean token/context
sau extraction, rồi adaptation dùng context trung bình và token similarities;
đây là thiết kế khác đường reproduce. Context attention được tái sử dụng khi
adaptation: tương đương số học ở eval, không lặp các dropout draws như code gốc.

`negative_reduction: separate` giữ hai phép min độc lập đúng code TEAM phát hành;
`joint` dùng min sau khi cộng hai nhánh theo §9.2 trong đề xuất. Matching so token
cùng chỉ số, không tạo ma trận frame–frame. EAPM dùng nearest-competitor cosine,
quality support, temperature và uniform shrinkage; part không có đối thủ hợp lệ
được mask. Khi không còn part eligible của class hoặc query mất toàn bộ landmark,
score dùng nhánh RGB TEAM chung backbone/pattern bank. Query mất riêng một part
thì part đó tạo token bằng RGB attention như §7.2; không thay support weights.

Có thể đặt `guidance_mode: bias|key|value|motion_off|full`, `lambda_motion: 0`,
`support_adaptation: false`, `parts: [0,1]`, `negative_weights:
opponent|candidate|uniform`, `part_weights.mode: uniform|learned|episode|adaptive`
để làm ablation. λ_motion=0 tắt mọi đóng góp motion; part embeddings vẫn tồn tại.
Không có episode-conditioned routing hoặc optical-flow branch bắt buộc.

## Cấu trúc và xác nhận

`src/motion_part_fsl/models/` chứa RGB/motion encoder, MGPR, adaptation, EAPM,
matching và `MotionPartFSL`. `datasets/` kiểm tra metadata/cache và episodes;
`preprocessing/` sinh RGB/landmark; `losses/` chứa CE/hard-pair; `engine.py` quản lý
train/resume/evaluate; `scripts/` chứa launcher/audit/visualize/ablate.

Xem [VALIDATION.md](VALIDATION.md) cho các kiểm tra thực sự đã chạy. Kiểm tra số
học đọc trực tiếp DPM và matching trong checkout gốc, không dùng lại công thức
chỉ từ tài liệu tóm tắt. Nguồn: [TEAM official repository](https://github.com/leesb7426/TEAM).
Chưa tái lập accuracy TEAM, chưa chạy training lớn trên WLASL/VSL; cần audit
landmark thực, khóa protocol và chạy ablation nhiều seeds trước khi kết luận.
