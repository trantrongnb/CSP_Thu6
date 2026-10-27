# CSP_Thu6 — Motion–Part FSL

Repository theo dõi: [trantrongnb/CSP_Thu6](https://github.com/trantrongnb/CSP_Thu6),
nhánh `main`.

Triển khai chính: [motion_part_fsl](motion_part_fsl/README.md), theo
[tài liệu cải tiến TEAM](Motion_Part_FSL_TEAM_CVPR2025_Chi_Tiet_Cai_Tien.md).

MGPR điều khiển pattern attention bằng chuyển động tay; EAPM tính trọng số
tay trái/tay phải/quan hệ hai tay từ support. Giữ adaptation, instance/exclusive
matching của TEAM và loss hai nhánh. Chưa có kết quả accuracy trên dữ liệu thật.

```bash
git clone --recurse-submodules https://github.com/trantrongnb/CSP_Thu6.git
cd CSP_Thu6/motion_part_fsl
bash scripts/run_conda.sh smoke --output outputs/smoke_new --steps 40
```

Nếu đã clone mà chưa lấy TEAM tham chiếu, chạy
`git submodule update --init --recursive` từ thư mục repository. Dữ liệu video,
cache và checkpoint nằm ngoài Git; xem README triển khai để chuẩn bị chúng.

Thư mục `TEAM/` đã chuyển thành `motion_part_fsl/references/team_original/`,
giữ nguyên checkout gốc để kiểm tra số học. Baseline optical-flow trước đây vẫn
ở [mfpa_fslr](mfpa_fslr/README.md). Hai phương pháp dùng cache khác nhau;
metadata/split WLASL cũ có thể tái sử dụng. Xem README triển khai mới để chuẩn bị
RGB + landmark cache, huấn luyện 1/5-shot và đánh giá WLASL/VSL độc lập.
