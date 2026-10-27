# Kiểm chứng triển khai — 27/10/2026

Đã thực hiện trong workspace này. Chưa có thí nghiệm accuracy trên WLASL/VSL.

| Kiểm tra | Kết quả |
|---|---|
| Unit/integration tests | **28 passed** |
| RGB pattern tokens so với TEAM gốc | Khớp số học ở eval trong tolerance 2–4e-6 |
| Support adaptation + logits pos/neg | Khớp code TEAM ở cả 1-shot và 5-shot |
| A0–A7 | Forward, loss, backward finite; runner train/evaluate xuất CSV |
| Support-only adaptation | Đổi query không đổi support weights; các query độc lập ở eval |
| Permutation | Đổi thứ tự class/query cho logits tương ứng sau relabel |
| Missing landmarks | Mất một tay hoặc toàn bộ vẫn finite; all-missing khớp RGB fallback |
| Motion scale = 0 | Thay motion inputs không làm đổi logits |
| EAPM | Nearest competitor, shrinkage, invalid-part masking, composition sensitivity |
| Kinematics | Velocity dùng dt thật; thiếu landmark không tạo vận tốc giả |
| Backbone checkpointing | Chunked/checkpointed tiny backbone giữ logits và gradient trong tolerance |
| Resume | Trọng số cuối bằng chính xác run không gián đoạn |
| Fixed episodes | Seed tái lập, không trùng support/query, phát hiện file episode bị sửa |
| Video preprocessing | Decode, letterbox, padding/timestamps, resume cache, source-change rejection |
| Python/shell syntax | Compile Python và `bash -n` qua |

Checkout TEAM tham chiếu được giữ nguyên trong `references/team_original/`,
revision `8bd6b6e07cbdf992649fc758ba16f13b674f8b8f`. Test đọc trực tiếp DPM
và compile các matching methods từ `model.py`, không tự viết lại reference formulas.
Đây là kiểm chứng head ở eval, chưa phải reproduction accuracy hay training
trajectory gốc. Training dùng lại attention context khi adaptation, BN statistics
đóng băng và preprocessing local; các lựa chọn đó được ghi trong README.

Smoke 40 bước, 3-way 1-shot, tiny CNN, dữ liệu RGB/landmark tổng hợp:

- Loss: **1.57572055 → 0.07467562**; overfit episode accuracy 100%.
- Train → validation → `best.pt`/`last.pt` → test chạy qua engine thật.
- Runner A0–A7 dùng cùng episode SHA256
  `46346187ec1bd5b5a312b15e67a215b34818ba20aa51a9f2c3c58dbbe39843d0`.
- Số accuracy tổng hợp không chứng minh giá trị của MGPR/EAPM trên ký hiệu thật.

Artifacts:

- [Smoke summary](outputs/smoke_cpu/smoke_summary.json)
- [Synthetic A0–A7 CSV](outputs/smoke_cpu/ablations/results.csv)
- [Synthetic checkpoint](outputs/smoke_cpu/training/best.pt)

Preprocessing thực trên 4 video WLASL, hai gloss `learn`/`leave`:

- MediaPipe Holistic **0.10.21**, 8 uniformly sampled frames, letterbox 224.
- Audit cache và SHA256 video nguồn: **0 failures**, không trùng source/config.
- Cả bốn video có body reference từ vai; mean hand detection coverage từng video:
  0.9375, 0.375, 0.5, 0.5. Coverage là detected mask, không phải calibrated confidence.
- Đã xuất và xem overlay RGB/L/R/torso trên clip `wlasl_1005_32598`.
- Episode thực 2-way 1-shot: ResNet-50 **không pretrained**, D=256, M=16,
  gradient checkpointing; logits `[2,2]`, loss finite, gradients tới RGB và motion.
  Chỉ kiểm tra forward/backward, không dùng số dự đoán làm metric nghiên cứu.

Artifacts:

- [Audit 4 video](outputs/real_video_check/audit.json)
- [ResNet-50 forward/backward](outputs/real_video_check/forward_backward.json)
- [Overlay PNG](outputs/real_video_check/learn_overlay.png)
- [Overlay video](outputs/real_video_check/learn_overlay.mp4)

Môi trường kiểm thử: Python 3.12.12, **PyTorch 2.5.1+cpu**, torchvision 0.20.1+cpu,
NumPy 1.26.4, OpenCV 4.13.0, pytest 8.4.2. Torch/torchvision CPU và pytest được cài
riêng vào `/tmp/mpfsl_cpu_deps` và `/tmp/mpfsl_test_deps`, không đổi môi trường
conda dùng chung. Holistic/protobuf 4.25.8 nằm trong `/tmp/mpfsl_landmark_deps`.
OpenCV hiện tại mới hơn ràng buộc `<4.12` trong pyproject; lần chạy này không phải
kiểm tra sạch toàn bộ dependency lock của một môi trường cài mới.

Lệnh đã dùng cho bộ tests:

```bash
CUDA_VISIBLE_DEVICES=-1 \
PYTHONPATH=/tmp/mpfsl_cpu_deps:/tmp/mpfsl_test_deps:src \
/home/iec/disk2/TrongTV/miniconda3/envs/signclip_5090/bin/python -m pytest -q
```

`signclip_5090` có PyTorch CUDA 12.8 nhưng CUDA MPS không phản hồi, check hết hạn
30 giây. Chưa kiểm chứng CUDA/AMP, chưa preprocessing toàn WLASL, chưa có VSL,
chưa benchmark ImageNet-pretrained accuracy, FLOPs hoặc nhiều seeds. Cần audit
landmarks trên nhiều clip, chuẩn bị cache đầy đủ và chạy ablation trên novel
classes trước khi kết luận phương pháp hiệu quả.
