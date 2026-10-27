# Motion–Part FSL: Thiết kế chi tiết các cải tiến từ TEAM (CVPR 2025) cho Few-Shot Isolated Sign Language Recognition

> **Phiên bản:** Research design v1.0 — 27/10/2026  
> **Mục tiêu:** Đặc tả phương pháp đủ chi tiết để bắt đầu hiện thực bằng PyTorch, thiết kế thí nghiệm, và chuyển thành mục *Method* trong bài báo.  
> **Phạm vi:** Isolated few-shot sign language recognition; thực nghiệm trên **WLASL** và **VSL** **độc lập** (within-dataset).  
> **Cơ sở:** TEAM, *Temporal Alignment-Free Video Matching for Few-shot Action Recognition*, SuBeen Lee et al., CVPR 2025.  
> **Trạng thái:** **Đề xuất nghiên cứu, chưa được xác thực bằng thực nghiệm.** Các phương trình ở phần “Proposed” là thiết kế để kiểm chứng, không phải công thức TEAM nguyên bản.

---

## Mục lục

1. [Tóm tắt và quyết định thiết kế](#1-tóm-tắt-và-quyết-định-thiết-kế)
2. [Bài toán và protocol](#2-bài-toán-và-protocol)
3. [TEAM gốc: giữ lại gì, thay đổi gì?](#3-team-gốc-giữ-lại-gì-thay-đổi-gì)
4. [Motivation: vì sao sign language khác action thông thường](#4-motivation-vì-sao-sign-language-khác-action-thông-thường)
5. [Tổng quan kiến trúc Motion–Part FSL](#5-tổng-quan-kiến-trúc-motionpart-fsl)
6. [Input preprocessing và đặc trưng động học](#6-input-preprocessing-và-đặc-trưng-động-học)
7. [Cải tiến 1 — Motion-Guided Pattern Representation](#7-cải-tiến-1--motion-guided-pattern-representation)
8. [Cải tiến 2 — Episode-Adaptive Part Matching](#8-cải-tiến-2--episode-adaptive-part-matching)
9. [Giữ và mở rộng instance/exclusive matching của TEAM](#9-giữ-và-mở-rộng-instanceexclusive-matching-của-team)
10. [Cải tiến tùy chọn — Hard-Pair Motion Loss](#10-cải-tiến-tùy-chọn--hard-pair-motion-loss)
11. [Hàm mất mát, tối ưu, và quy trình huấn luyện](#11-hàm-mất-mát-tối-ưu-và-quy-trình-huấn-luyện)
12. [Suy luận với novel classes: ví dụ đầy đủ](#12-suy-luận-với-novel-classes-ví-dụ-đầy-đủ)
13. [Thuật toán và tensor shapes](#13-thuật-toán-và-tensor-shapes)
14. [Gợi ý triển khai bằng PyTorch](#14-gợi-ý-triển-khai-bằng-pytorch)
15. [Thiết kế tập dữ liệu, chia lớp và chống leakage](#15-thiết-kế-tập-dữ-liệu-chia-lớp-và-chống-leakage)
16. [Ablation có đối chứng](#16-ablation-có-đối-chứng)
17. [Phân tích sâu, kiểm chứng giả thuyết](#17-phân-tích-sâu-kiểm-chứng-giả-thuyết)
18. [Rủi ro kỹ thuật và cách xử lý](#18-rủi-ro-kỹ-thuật-và-cách-xử-lý)
19. [Các phiên bản khả thi và lộ trình](#19-các-phiên-bản-khả-thi-và-lộ-trình)
20. [Đóng góp, novelty, và cách trình bày bài báo](#20-đóng-góp-novelty-và-cách-trình-bày-bài-báo)
21. [Nguồn tham khảo](#21-nguồn-tham-khảo)
22. [Checklist trước khi kết luận phương pháp hiệu quả](#22-checklist-trước-khi-kết-luận-phương-pháp-hiệu-quả)

---

## 1. Tóm tắt và quyết định thiết kế

### 1.1. Câu hỏi nghiên cứu

> **Trong một episode gồm các lớp ký hiệu mới, bằng chứng chuyển động xuất hiện vào lúc nào, nằm ở bộ phận nào, và làm thế nào dùng nó để phân biệt các lớp với chỉ 1–5 video support mỗi lớp mà không cần temporal alignment giữa từng cặp video?**

Ý tưởng là kế thừa ưu thế **alignment-free, token-wise matching** của TEAM và bổ sung các quy nạp (*inductive biases*) phù hợp với ngôn ngữ ký hiệu.

### 1.2. Hai đóng góp chính đã chọn

| Thành phần | Cơ chế trong TEAM | Cải tiến đề xuất | Điều phải chứng minh |
|---|---|---|---|
| **C1. Motion-Guided Pattern Representation (MGPR)** | Learnable pattern tokens cross-attend tới đặc trưng video toàn cục | Motion từ hand landmarks điều khiển key/bias của token-to-frame attention, đồng thời tạo descriptor riêng cho tay trái, tay phải, quan hệ hai tay | Hơn TEAM RGB-only và hơn **RGB + landmark concatenation** trong điều kiện công bằng |
| **C2. Episode-Adaptive Part Matching (EAPM)** | Các pattern tokens được so khớp theo chỉ số; adaptation loại bỏ thông tin chung giữa classes | Tính trọng số **theo lớp và theo bộ phận** bằng mức độ phân biệt giữa các support classes, có cân nhắc độ tin cậy landmark | Hơn trọng số uniform / learned global, đặc biệt với các cặp sign gần nhau |

### 1.3. Thành phần mở rộng, không phải đóng góp bắt buộc

- **C3. Hard-Pair Motion Loss:** nhấn mạnh negative có appearance giống nhau nhưng nhãn ký hiệu khác nhau. Chỉ giữ nếu có cải thiện trong ablation.
- **Reliability calibration / shrinkage:** xem là thiết kế để ổn định C2, không quảng bá thành novelty độc lập nếu chỉ là kỹ thuật thông thường.
- **Episode-Adaptive Pattern Routing:** từng được nghiên cứu thành **một hướng TEAM khác** (episode-conditioned pattern semantics, dynamic routing, class-conditioned query parsing). Không được tự động gộp ba module này vào Motion–Part FSL; sẽ làm mất tập trung hai đóng góp trên.
- **Optical flow:** *không* là nhánh bắt buộc của phiên bản 1; có thể dùng trong ablation mở rộng.

### 1.4. Phân biệt điều đã chốt và điều cần thử

**Đã thống nhất ở mức đề xuất:** RGB + landmarks; hai module C1/C2; support-only adaptation; token-wise matching; WLASL và VSL riêng biệt; 5-way 1/5-shot; hard-pair tùy chọn.  
**Chưa cố định:** số token M, backbone cuối cùng, tỉ lệ pha trộn negative score, chính xác công thức reliability, tập keypoints và bộ VSL cụ thể.  
**Chưa có chứng cứ:** độ tăng accuracy, FLOPs, tính ổn định trên signer chưa gặp, hoặc khẳng định TEAM thất bại trên sign language.

---

## 2. Bài toán và protocol

Ta xét tập dữ liệu video isolated signs với các nhãn từ vựng rời rạc:

\[
\mathcal D=\{(v_i,y_i)\}_{i=1}^{L},\qquad y_i\in\mathcal C.
\]

Tách tập lớp không trùng nhau:

\[
\mathcal C_{\mathrm{train}}\cap\mathcal C_{\mathrm{val}}=
\mathcal C_{\mathrm{train}}\cap\mathcal C_{\mathrm{test}}=
\mathcal C_{\mathrm{val}}\cap\mathcal C_{\mathrm{test}}=\varnothing.
\]

Một episode \(\mathcal E\) gồm \(N\) lớp, mỗi lớp có \(K\) support:

\[
\mathcal S=\{(v_{c,k},c)\mid c=1,\ldots,N;\ k=1,\ldots,K\},
\quad
\mathcal Q=\{(q_j,y_j)\}_{j=1}^{U}.
\]

- **N-way**: số lớp ứng viên trong episode; mặc định \(N=5\).
- **K-shot**: số support video **mỗi lớp**; bắt đầu với \(K\in\{1,5\}\). Khảo sát 2/3/4-shot về sau nếu chi phí cho phép.
- **Meta-train**: học encoder, pattern bank, motion attention và các tham số khác trên lớp base.
- **Meta-val**: chọn hyperparameter và checkpoint theo các lớp validation.
- **Meta-test**: giữ nguyên trọng số đã học; chỉ tính prototype, adaptation và trọng số part từ **support set novel**. Không gradient-update bằng nhãn query.

### 2.1. Phạm vi dữ liệu

1. **WLASL:** train/val/test trong WLASL.
2. **VSL:** train/val/test trong VSL.
3. Không coi **WLASL → VSL** là thí nghiệm chính. Nếu có cross-lingual transfer thì tách riêng và chỉ thực hiện sau khi mô hình trong từng dataset được kiểm định tốt.

> Lưu ý về protocol: tên WLASL/VSL không tự xác định một split few-shot chuẩn. Phải công bố chính xác phiên bản dữ liệu, tập class ID, số video khả dụng/lớp và quy tắc signer separation.

---

## 3. TEAM gốc: giữ lại gì, thay đổi gì?

### 3.1. Cơ chế cốt lõi (phần **TEAM**, không phải đóng góp của chúng ta)

TEAM (Lee et al., CVPR 2025) lấy \(T\) frame mỗi video và trích xuất feature:

\[
F(v)=[f_1,\ldots,f_T]\in\mathbb R^{T\times D},
\qquad f_t=g_\theta(x_t).
\]

TEAM dùng một *pattern bank* học được:

\[
P=[p_1,\ldots,p_M]\in\mathbb R^{M\times D}.
\]

**Instance pattern tokens** (TEAM Eq. 2–3):

\[
\bar p_m^+=p_m+\operatorname{CA}(p_m,F,F),\qquad
p_m^+=\bar p_m^++\operatorname{MLP}(\bar p_m^+).
\]

**Exclusive pattern tokens** (TEAM Eq. 6–7):

\[
\bar p_m^-=p_m-\operatorname{CA}(p_m,F,F),\qquad
p_m^-=\bar p_m^-+\operatorname{MLP}(\bar p_m^-).
\]

Hai loại token phục vụ hai mục đích bổ sung:

- **Instance (+):** bằng chứng hiện diện, đặc trưng cho class/instance.
- **Exclusive (−):** bằng chứng “otherness”, giúp diễn giải sự không thuộc về một class và tạo ranh giới với các class khác.

TEAM **đã có episode adaptation trên support tokens** để giảm thông tin chung giữa các lớp. Vì vậy, không thể tuyên bố “cải tiến lần đầu đưa episode adaptation vào TEAM”. Điểm cải tiến của chúng ta phải là **part-conditioned, motion-aware và support-driven discriminative weighting**, khác adaptation gốc về đối tượng và cách tính.

### 3.2. So khớp token-wise (TEAM)

Với cosine distance \(d\), positive matching dạng:

\[
\operatorname{PD}(c,q)=\sum_{m=1}^{M}-d(\hat p^+_{c,m},p^+_{q,m}),
\]

với \(\hat p^+_{c,m}\) là support token sau adaptation. Nhánh exclusive còn xét các lớp cạnh tranh khác với class đang kiểm tra; xem TEAM Eq. 8, 10–18 để tái lập **chính xác**. Kết quả tích hợp hai nhánh instance/exclusive.

Quan trọng: TEAM so token \(m\) ở support với token \(m\) ở query, không tính ma trận similarity frame–frame và không chạy DTW/OTAM. Tính toán attention để *tổng hợp video* vẫn phụ thuộc \(T\), nhưng matching giữa hai video có độ phức tạp chủ yếu phụ thuộc \(M\), không phải \(T^2\).

### 3.3. Thông tin từ code/paper chính thức

- Paper thực nghiệm dùng **ResNet-50** và **ViT-B** pre-trained trên ImageNet, **8 uniformly sampled frames/video**, input **224×224**; train 10.000 iterations theo cấu hình được báo cáo. Đây là cấu hình **TEAM gốc**, không mặc định là cấu hình tối ưu cho sign.
- Trong repo, `model.py` khai báo `CNN_FSHead`, `TEAM` và các biến thể; `model_util.py` có `Discriminative_Pattern_Matching`.
- `TEAM` dùng `logits_pos` và `logits_neg`, với loss phân loại cho cả hai nhánh. Việc thay score nên theo sát triển khai thực tế, không chỉ dựa công thức tóm tắt.
- Code có ví dụ `--backbone ResNet --shot 1 --agg_num 60`; số token này là **cấu hình ví dụ TEAM**, không nên xem là cố định cho mọi dataset/sign.

### 3.4. Những điều **giữ lại**

1. Backbone hình ảnh/chia episode làm baseline đối chứng.
2. Khái niệm shared learnable pattern bank.
3. Instance **và** exclusive pattern tokens.
4. Support-token adaptation của TEAM, trước khi áp dụng mở rộng hoặc so sánh biến thể.
5. Token-wise comparison, không explicit temporal alignment support–query.

### 3.5. Những điều **thay đổi**

1. Dùng hand motion và quan hệ hai tay để điều hướng **pattern aggregation**.
2. Phát sinh đặc trưng theo vùng \(r\in\{L,R,B\}\): left hand \(L\), right hand \(R\), bimanual relation \(B\).
3. Tính cách đóng góp từng vùng **từ support class composition của episode**.
4. Tổng hợp score part-wise thay vì giả định mọi vùng đóng góp như nhau.
5. Tùy chọn hard-pair contrast trong meta-training.

---

## 4. Motivation: vì sao sign language khác action thông thường

### 4.1. Các yếu tố cấu thành ký hiệu

Nhận dạng isolated sign không chỉ dựa trên “đối tượng nào xuất hiện” mà còn nhạy với:

- **Handshape:** cấu hình các ngón tay và hình dạng bàn tay.
- **Location:** vị trí bàn tay so với thân, đầu hoặc vùng mặt.
- **Motion:** hướng, quỹ đạo, tốc độ, lặp lại và thay đổi vị trí theo thời gian.
- **Orientation:** hướng lòng bàn tay và ngón tay.
- **Two-hand relation:** phối hợp tương đối giữa hai tay, độ lệch/đối xứng/tiếp xúc.
- **Non-manual cues:** khuôn mặt, đầu, nét biểu cảm có thể mang nghĩa. Phiên bản đầu không chọn face làm core part nhưng **cần kiểm tra tác động của việc bỏ qua**.

### 4.2. Giả thuyết H1 — motion có thể chưa được nhấn đúng

Một encoder RGB có thể học thông tin chuyển động gián tiếp qua các frame, nhưng hai ký hiệu cùng handshape/position có thể chỉ khác *quỹ đạo*, *hướng*, hoặc *thời điểm bắt đầu và dừng*. Vì vậy, giả thuyết cần kiểm chứng là motion guidance giúp tập trung bằng chứng đúng hơn so với chỉ đưa RGB vào TEAM.

**Không khẳng định rằng TEAM gốc không học được motion**. Điều cần chứng minh là guidance tạo giá trị **bổ sung có ý nghĩa** so với baseline được kiểm soát.

### 4.3. Giả thuyết H2 — bộ phận quan trọng thay đổi theo episode

- Episode A: các class có handshape phải giống nhau nhưng quỹ đạo **tay trái** khác nhau.
- Episode B: mỗi class khác ở **tay phải**.
- Episode C: phân biệt chủ yếu ở **khoảng cách và tương tác giữa hai tay**.

Nếu mô hình dùng một tập trọng số cố định cho mọi episode, tín hiệu không liên quan có thể làm giảm discriminability. Tuy nhiên trọng số học từ 1-shot cũng có thể nhiễu, vì vậy phải có **regularization, reliability và ablation**.

### 4.4. Ba kịch bản kiểm thử

| Kịch bản | Bằng chứng quyết định | Kỳ vọng hợp lý của phương pháp |
|---|---|---|
| Hai ký hiệu cùng handshape nhưng chuyển động trái/ngược nhau | Motion direction | C1 khiến attention và descriptor thể hiện rõ hướng/quỹ đạo |
| Các ký hiệu chỉ khác phối hợp hai tay | Relative hand position và synchronization | C2 tăng trọng số `B` nếu support episode có độ phân biệt cao |
| Một tay bị che khuất hoặc landmark kém | RGB/fallback và độ tin cậy từng vùng | C2 không tăng nhầm trọng số part nhiễu; giữ độ ổn định |

---

## 5. Tổng quan kiến trúc Motion–Part FSL

```text
             N-way K-shot SUPPORT                  QUERY video
              RGB + landmarks                    RGB + landmarks
                     |                                 |
                Shared RGB encoder + motion/landmark encoder
                     |                                 |
              F^a, F^m, confidence              F^a, F^m, confidence
                     |                                 |
            [C1] Motion-Guided Pattern Representation
                     |                                 |
          part-wise P+ / P- tokens             part-wise P+ / P- tokens
             L | R | Relation                   L | R | Relation
                     |
               TEAM support adaptation
                     |
             Class-wise support references
                     |
           [C2] Episode-Adaptive Part Weights
            w_{c,L}, w_{c,R}, w_{c,B}
                     |                                 |
                     +--------------+------------------+
                                    |
                   Part-wise token matching (+ / -)
                                    |
                       Weighted class logits
                                    |
                                argmax(c)
```

**Quy tắc chống data leakage:** support và query chạy **cùng encoder**. Query tự sinh feature/tokens bằng shared weights, nhưng chỉ support **quyết định class prototypes, support adaptation và part weights**. Query không cung cấp nhãn hoặc tham gia cập nhật support weight. Không transductive query statistics ở setting chính.

### 5.1. Ký hiệu tensor

| Ký hiệu | Ý nghĩa | Shape một video |
|---|---|---|
| \(X\) | RGB frames | \(T\times3\times H\times W\) |
| \(H\) | hand landmarks với validity/confidence | \(T\times J\times(2/3+1)\) |
| \(F^a\) | appearance features | \(T\times D_a\) |
| \(b_t^r\) | kinematic descriptor vùng \(r\) | \(D_m\) |
| \(v_t^r\) | fused part/frame descriptor | \(D\) |
| \(A_{m,t}^r\) | motion-guided attention | \(M\times T\) mỗi vùng |
| \(Z^{r,+}, Z^{r,-}\) | instance/exclusive pattern tokens | \(M\times D\) mỗi vùng |
| \(\hat Z_c^{r,+},\hat Z_c^{r,-}\) | support references sau adaptation | \(M\times D\) mỗi class/vùng |
| \(w_{c,r}\) | part weight dựa support | \(N\times R\) toàn episode |

Ở phiên bản đầu, \(R=3\) (`L`, `R`, `B`). Có thể đặt \(D=256\) hoặc \(512\) trong phiên bản lightweight, nhưng **nếu so sánh trực tiếp với TEAM gốc phải kiểm soát backbone/dimension/params**.

---

## 6. Input preprocessing và đặc trưng động học

### 6.1. Dữ liệu RGB

- Decode video theo FPS và duration, bỏ clip lỗi.
- Lấy \(T\in\{8,16\}\) frame bằng uniformly sampling làm baseline. **8** là lựa chọn dễ đối chiếu TEAM; **16** là ablation temporal resolution.
- Dùng cùng frame timestamps cho RGB và landmarks. Không được để landmark index lệch frame index sau sampling hoặc temporal augmentation.
- Resize/crop có kiểm soát, giữ tay không bị cắt: sign language có thể mất thông tin nếu random crop chặt.
- Nên giữ **full-frame RGB** để không làm mất body/face; tùy ablation thêm crop tay (ROI), nhưng crop là một **biến số riêng**.

### 6.2. Landmarks

Có thể trích từ MediaPipe Holistic/Hands (hoặc bộ detector tương đương), mỗi tay chứa \(J_h\) khớp; với 21 khớp/tay, hai tay có 42 khớp. Ghi rõ phiên bản extractor, tọa độ 2D/3D, confidence và policy cho keypoints không tìm thấy.

Đặt \(h_{t,j}^{(r)}\) là tọa độ khớp \(j\) tại frame \(t\), tay \(r\in\{L,R\}\). Chọn một gốc quy chiếu dựa trên midpoint vai hoặc torso center \(o_t\), và scale \(s_t\) như shoulder width ổn định:

\[
\tilde h_{t,j}^{(r)}=\frac{h_{t,j}^{(r)}-o_t}{\max(s_t,\varepsilon)}.
\]

**Không dùng tự thân bounding box của từng tay làm chuẩn hóa duy nhất** nếu muốn giữ vị trí tay so với cơ thể, vì điều này có thể triệt tiêu thông tin location. Có thể song song giữ cả coordinates body-relative và hand-relative.

### 6.3. Vận tốc và gia tốc

\[
\Delta \tilde h_{t,j}^{(r)}=\frac{\tilde h_{t,j}^{(r)}-\tilde h_{t-1,j}^{(r)}}{\Delta t},
\qquad
\Delta^2\tilde h_{t,j}^{(r)}=
\frac{\Delta\tilde h_{t,j}^{(r)}-\Delta\tilde h_{t-1,j}^{(r)}}{\Delta t}.
\]

Nếu \(T\) frame được subsample từ video dài, \(\Delta t\) phải tương ứng **timestamp thực** hoặc thời gian đã chuẩn hóa, không giả sử các video đều có FPS giống nhau. Mất landmark ở t hoặc t−1 phải dẫn tới **invalid velocity mask**, không tạo vận tốc giả từ giá trị zero.

### 6.4. Quan hệ giữa hai tay

Đặt tâm tay \(\bar h_t^L,\bar h_t^R\) là average của các khớp hợp lệ hoặc tâm palm. Đặc trưng quan hệ:

\[
\delta_t^{LR}=\bar h_t^L-\bar h_t^R,
\quad d_t^{LR}=\|\delta_t^{LR}\|_2,
\quad \Delta\delta_t^{LR}=\delta_t^{LR}-\delta_{t-1}^{LR}.
\]

Có thể thêm hướng tương đối, độ tương đồng vận tốc và chênh lệch độ mở bàn tay. Không suy ra **hand contact** chắc chắn chỉ từ distance 2D khi không có chứng cứ 3D.

### 6.5. Motion descriptor \(b_t^r\)

Với mỗi vùng:

\[
\begin{aligned}
u_t^L &= [\tilde H_t^L,\Delta\tilde H_t^L,\Delta^2\tilde H_t^L,\operatorname{handshape}(H_t^L)],\\
\nu_t^R &= [\tilde H_t^R,\Delta\tilde H_t^R,\Delta^2\tilde H_t^R,\operatorname{handshape}(H_t^R)],\\
\nu_t^B &= [\delta_t^{LR},d_t^{LR},\Delta\delta_t^{LR},\operatorname{relativeOrientation}_t].
\end{aligned}
\]

\[
b_t^r = E_m^r(\nu_t^r,\operatorname{mask}_t^r),
\qquad b_t^r\in\mathbb R^{D_m}.
\]

`handshape` có thể là các độ dài tương đối giữa khớp, góc các đốt tay hoặc tọa độ hand-relative; ban đầu không cần quá phức tạp. Motion encoder gọn: MLP 2 lớp + LayerNorm; ablation so với Temporal Conv/Transformer nhẹ. Nếu detector không đủ chất lượng, *fuse* ít thông tin hơn thay vì tạo dày đặc nhiều vector sai.

### 6.6. Reliability \(\rho\) và missing data

Đặt \(u_t^r\in[0,1]\) là độ tin cậy mỗi frame/vùng. Nếu detector chỉ trả `detected/not detected`, dùng mask nhị phân; không tự tạo confidence giả có vẻ liên tục. Video-level reliability:

\[
\rho^r(v)=\frac{1}{T}\sum_{t=1}^T u_t^r.
\]

Có thể bổ sung `valid coverage`, tỉ lệ velocity hợp lệ hoặc detector confidence đã hiệu chuẩn. Với quan hệ hai tay, ví dụ lấy \(u_t^B=\min(u_t^L,u_t^R)\) vì cần cả hai tay. **Reliability score không đồng nghĩa với discriminativeness**: một tay được thấy rõ có thể hoàn toàn không giúp phân biệt các lớp.

Chính sách mất keypoints:

1. `mask=0` cho tọa độ không xác thực.
2. Nội suy chỉ với gap ngắn và có cờ `interpolated`; không nội suy qua đoạn bị che dài.
3. Vận tốc tại các cặp frame không hợp lệ phải masked.
4. Khi toàn bộ landmark của video mất, cho phép **RGB-only fallback**, không cho softmax sinh `NaN`.
5. Test cả một tay mất hoàn toàn, cả hai tay mất, jitter 2D, và confidence thấp.

---

## 7. Cải tiến 1 — Motion-Guided Pattern Representation

### 7.1. Ý tưởng chính

Đầu vào của TEAM truyền thống là chuỗi appearance \(F^a\). Ta không chỉ `concat` vector motion vào video embedding ở cuối. Thay vào đó, **motion trực tiếp thay đổi phân bố temporal attention** dùng để hình thành các pattern tokens. Nhờ đó, một token có khả năng chú ý khác nhau khi đoạn video có chuyển động của tay trái, tay phải hoặc quan hệ hai tay nổi bật.

### 7.2. Cách đơn giản nhất (đề nghị làm MVP)

Đặt shared RGB features:

\[
f_t^a = g_\theta(x_t)\in\mathbb R^{D_a}.
\]

Project appearance và kinematics về cùng kích thước:

\[
a_t=W_a f_t^a,\qquad m_t^r=W_m^r b_t^r,
\qquad a_t,m_t^r\in\mathbb R^D.
\]

Tạo part-conditioned frame representation:

\[
v_t^r=\operatorname{LN}\big(a_t+\lambda_v m_t^r+e_r\big),
\]

với \(e_r\) là embedding xác định bộ phận (cùng shape D). Có thể dùng \(v_t^r=a_t\) lúc MVP để **isolate tác dụng motion attention**; sau đó mới bật fusion trong values.

Pattern query cho từng part:

\[
q_m^r=p_m+e_r.
\]

Key có motion:

\[
k_t^r=W_k v_t^r+W_b b_t^r.
\]

**Motion saliency bias** cho từng token:

\[
\beta_{m,t}^r=\lambda_g\tanh\!\left(
(u_m^r)^\top\operatorname{LN}(b_t^r)
\right).
\]

Với frame có motion thấp/bất động cần tránh tự động gán `bad`: một số sign phân biệt bằng handshape, nên attention vẫn có nhánh appearance và hệ số \(\lambda_g\) học được/được điều chỉnh. Bias có thể phụ thuộc token, không chỉ dựa vào chuẩn vận tốc.

Độ chú ý:

\[
e_{m,t}^r=
\frac{(W_q q_m^r)^\top k_t^r}{\sqrt D}
+\beta_{m,t}^r
+\lambda_u\log(\varepsilon+u_t^r).
\]

\[
\alpha_{m,t}^r=\frac{\exp(e_{m,t}^r)}{\sum_{\ell=1}^{T}\exp(e_{m,\ell}^r)}.
\]

Khi \(u_t^r=0\), có thể dùng large negative bias thay cho log nếu **còn ít nhất một valid frame**. Nếu không frame nào hợp lệ, bỏ term motion/quality và quay về RGB attention. Như vậy không gây all-`-inf`.

Tổng hợp video:

\[
A_m^r=\sum_{t=1}^{T}\alpha_{m,t}^r W_vv_t^r.
\]

Sinh hai loại tokens tương tự TEAM:

\[
\bar z_{m}^{r,+}=q_m^r+A_m^r,
\qquad z_{m}^{r,+}=\bar z_{m}^{r,+}+\mathrm{MLP}^{+}(\bar z_{m}^{r,+}),
\]

\[
\bar z_{m}^{r,-}=q_m^r-A_m^r,
\qquad z_{m}^{r,-}=\bar z_{m}^{r,-}+\mathrm{MLP}^{-}(\bar z_{m}^{r,-}).
\]

> **Đây là phương trình Proposed MGPR**, lấy cảm hứng từ `+CA`/`−CA` của TEAM. Các biến \(m_t^r,\beta^r,u_t^r\) không có trong TEAM gốc. Nếu muốn giữ faithful TEAM hơn, dùng **cùng MLP** cho hai nhánh như code gốc và so sánh phiên bản shared-vs-separate MLP trong ablation.

### 7.3. Đặc điểm quan trọng của thiết kế

**(i) Shared pattern index:** token \(m\) của query và support dùng cùng global pattern prior \(p_m\), giữ ý nghĩa matching theo token index như TEAM. `part embedding` \(e_r\) giúp phân biệt vùng nhưng không buộc mọi token mang ngữ nghĩa một phase cố định.

**(ii) Không cần explicit alignment:** mỗi video tự tính \(\alpha^r\) trên trục thời gian của chính video đó. Không có matching frame \(t\) của query với frame \(t'\) của support.

**(iii) Motion acts before pooling:** tác động của motion đến token có thể đo trực tiếp qua attention maps. Đây là đối chứng cần thiết với concatenation sau pooling.

**(iv) Motion không thay thế appearance:** RGB vẫn cần nhận biết handshape, orientation, location, face/body context.

**(v) Hard hand segmentation không bắt buộc:** ROI tay là biến thể thêm, còn baseline dùng full-frame RGB + motion-conditioned part embedding.

### 7.4. Các biến thể nên thử tuần tự

| Biến thể | Key | Value | Motion bias | Vai trò |
|---|---|---|---|---|
| TEAM gốc | RGB | RGB | Không | Baseline |
| Early concat | RGB+motion | RGB+motion | Không | Kiểm tra liệu chỉ thêm dữ liệu đã đủ |
| MGPR-A | RGB | RGB | Có | Tách riêng tác dụng temporal guidance |
| MGPR-B | RGB+motion | RGB | Có | Motion đổi vị trí attention nhưng giữ visual value |
| **MGPR-C (đề xuất full)** | RGB+motion | RGB+motion | Có | Full motion-conditioned representation |
| Motion-only | motion | motion | Có | Kiểm tra phần đóng góp của RGB |

**Quy tắc:** tránh so full MGPR có thêm parameters với TEAM nhỏ hơn rồi kết luận “guidance tốt hơn”. Thêm baseline `concat+MLP` được **parameter-matched** nếu có thể.

### 7.5. Ví dụ trực giác

Giả sử có hai ký hiệu với khung đầu đều cùng handshape, nhưng ở giữa clip một sign di chuyển tay trái sang ngang, sign còn lại di chuyển theo đường chéo. TEAM có thể chú ý nhiều đến các khung có hình rõ. MGPR kỳ vọng giúp một số token tăng attention vào đoạn chứa hướng vận động khác biệt. **Ví dụ chỉ để giải thích, không đại diện cho cặp sign cụ thể đã xác minh trong WLASL/VSL.**

### 7.6. Điểm không nên làm ở phiên bản đầu

- Không tuyên bố pattern tokens tương đương `phase alignment` cố định; TEAM cố ý alignment-free.
- Không dùng soft/hard query-label supervision để điều khiển attention ở meta-test.
- Không buộc mỗi token chỉ được nhìn một đoạn không chồng lắp; vậy sẽ làm mất linh hoạt TEAM.
- Không tăng quá nhiều bộ phận (face, body, head, từng ngón) ngay từ đầu trước khi chứng minh 3 vùng cơ bản.

---

## 8. Cải tiến 2 — Episode-Adaptive Part Matching

### 8.1. Ý tưởng chính

Sau C1, mỗi video có representation cho từng vùng:

\[
Z^r(v)=\{Z^{r,+}(v),Z^{r,-}(v)\},\quad r\in\{L,R,B\}.
\]

Cần quyết định: trong **episode hiện tại**, vùng nào có thể phân biệt class \(c\) với các đối thủ? Trọng số **không phải** một parameter \(w_r\) cố định được học trên base classes; chúng được **tính động từ support classes** và có thể khác nhau theo class:

\[
\boldsymbol w_c^{\mathcal E}=(w_{c,L},w_{c,R},w_{c,B}),
\qquad \sum_r w_{c,r}=1.
\]

### 8.2. Support class references

Với \(K\) support video của class \(c\), chọn một trong hai cách:

**A. Mean trước pattern aggregation:** trung bình frame features trong class trước `DPM`, bám sát một số nhánh triển khai gốc của TEAM (cần tái lập code chính xác).  
**B. Mean sau token extraction:**

\[
\mu_{c,m}^{r,\pm}=\frac1K\sum_{k=1}^{K} z_{m}^{r,\pm}(v_{c,k}).
\]

Khuyến nghị làm **A** để reproduce TEAM trước, sau đó **B** là ablation thiết kế. Nếu \(K=1\), hai cách trùng nhau chỉ khi aggregation không có lớp phi tuyến/augmentation làm khác quy trình.

### 8.3. TEAM support adaptation theo từng part

TEAM gốc thích nghi support tokens theo các lớp của episode. Trong phương pháp đề xuất, áp dụng cùng nguyên tắc **độc lập theo part**, trước khi tính class distance:

\[
\hat Z_{c}^{r,\pm}=
\mathcal A_{\mathrm{TEAM}}
\left(
Z_{c}^{r,\pm},\left\{Z_o^{r,\pm}:o\ne c\right\}
\right).
\]

Ở đây \(\mathcal A_{\mathrm{TEAM}}\) là support adaptation gốc; khi cài đặt phải dùng toán tử và hệ số được tác giả dùng, không tự thay thế bằng `subtract mean` rồi gọi là reproduce TEAM. Lấy \(\hat Z_c\) để tính prototype similarity của từng part.

### 8.4. Độ phân biệt support-only

Cách đơn giản, trực giác và có thể triển khai:

\[
\mathrm{sim}_{c,o}^{r}
=\frac1M\sum_{m=1}^{M}
\cos(\hat z_{c,m}^{r,+},\hat z_{o,m}^{r,+}).
\]

Để đánh giá **đối thủ gần nhất** của class \(c\):

\[
D_{c,r}
=\min_{o\ne c}\left(1-\mathrm{sim}_{c,o}^{r}\right)
=1-\max_{o\ne c}\mathrm{sim}_{c,o}^{r}.
\]

- \(D_{c,r}\) lớn: class \(c\) tách tốt khỏi **đối thủ khó nhất** tại part \(r\).
- \(D_{c,r}\) nhỏ: có đối thủ gần giống ở part đó.
- Dựa **hoàn toàn trên support**; không dùng nhãn query, nên hợp lệ trong inductive few-shot meta-test.

Có thể thử `mean competitor margin`, `top-2 nearest`, `max inter-class gap` thay vì min; tuy nhiên nên ưu tiên nearest competitor vì phù hợp nghiên cứu hard-class discrimination.

### 8.5. Kết hợp reliability

Tính độ tin cậy support của class \(c\), part \(r\):

\[
\rho_{c,r}=\frac1K\sum_{k=1}^{K}\rho^r(v_{c,k}).
\]

Một thiết kế đề xuất:

\[
\xi_{c,r}
=\frac{D_{c,r}}{\tau_d}
+\lambda_\rho\log(\rho_{c,r}+\varepsilon).
\]

Sau đó:

\[
\tilde w_{c,r}=\operatorname{softmax}_{r}(\xi_{c,r}).
\]

Để tránh 1-shot bị quá chắc chắn vì một prototype nhiễu, áp dụng **uniform shrinkage**:

\[
w_{c,r}=(1-\eta)\tilde w_{c,r}+\eta\frac1R,
\qquad 0\le\eta\le1.
\]

Ở đây \(\eta\) lớn → gần uniform hơn; \(\eta=1\) → hoàn toàn uniform. Có thể học \(\eta\) trên meta-train nhưng phải tránh khiến nó lệ thuộc dataset test. Với reliability bằng 0, không nên ép part có landmark hỏng vẫn được điểm lớn chỉ vì uniform prior: hoặc thay \(s_{c,r}\) bằng score RGB fallback có định nghĩa, hoặc mask part rồi chuẩn hóa lại trọng số trên tập part hợp lệ.

**Phiên bản tối giản** giống ý tưởng ban đầu:

\[
w_{c,r}=\operatorname{softmax}_{r}\left(D_{c,r}/\tau_d\right).
\]

Reliability và shrinkage là **cải tiến ổn định** thêm sau khi đã thử công thức đơn giản này.

### 8.6. Vấn đề quan trọng ở 1-shot

Với \(K=1\), không có thống kê intra-class đáng tin. Do đó:

- Tránh dùng \(\operatorname{Var}\) trong class như thể variance ước lượng được từ một ví dụ.
- Không dùng `within-class covariance inverse` để tính Mahalanobis từ 1-shot.
- Có thể tính **inter-class separation** từ 5 support video nhưng phải regularize.
- Đánh giá \(\eta\), \(\tau_d\) trên meta-val riêng và báo cáo sensitivity.
- Theo dõi entropy \(-\sum_r w_{c,r}\log w_{c,r}\) để phát hiện collapse vào một part.

### 8.7. Ví dụ tính trọng số

Giả sử một 5-way episode có class C, độ phân biệt (giá trị **minh họa**, không phải thực nghiệm) như sau:

| Part | \(D_{C,r}\) | Reliability \(\rho_{C,r}\) | Nhận xét |
|---|---:|---:|---|
| Left | 0.45 | 0.95 | Khá phân biệt và đáng tin |
| Right | 0.12 | 0.99 | Dù thấy rõ nhưng dễ nhầm các class |
| Relation | 0.35 | 0.80 | Khá phân biệt nhưng dữ liệu kém chắc |

Nếu \(\lambda_\rho=0\), \(\tau_d=0.2\): \(\xi=(2.25,0.60,1.75)\); \(\tilde w\) xấp xỉ \((0.55,0.11,0.34)\). Với \(\eta=0.2\), trọng số cuối gần \((0.51,0.16,0.34)\). Như vậy `Left` quan trọng nhất **không phải** vì là tay thuận, mà vì nó phân biệt class C rõ nhất trong episode cụ thể.

Nếu composition thay đổi, \(D_{C,r}\) và \(w_{C,r}\) cũng thay đổi dù video support của C không đổi. Đây là yếu tố **episode-adaptive**.

### 8.8. Cách đánh giá EAPM có thật sự hoạt động

- So `uniform`, `global learned weights`, `support discriminativeness`, `support + reliability`, `support + reliability + shrinkage`.
- Cố định tập support của một class, thay các class khác trong episode và quan sát \(w\) thay đổi.
- Đánh giá correlation giữa \(w_{c,r}\) và hiệu quả nhận dạng nếu chỉ dùng part \(r\) (kiểm định trên test phân tích; không dùng để điều chỉnh inference).
- Báo cáo tỉ lệ collapse trọng số, phân phối entropy và quan hệ với landmark quality.

---

## 9. Giữ và mở rộng instance/exclusive matching của TEAM

Đây là phần rất quan trọng để phương pháp thực sự là **TEAM extension**, không phải một ProtoNet RGB+landmarks đứng riêng.

### 9.1. Positive score trên từng part

Với \(\hat Z_c^{r,+}\) (support sau TEAM adaptation) và \(Z_q^{r,+}\) (query):

\[
s_{c,r}^{+}(q)=\frac1M\sum_{m=1}^{M}
\cos(\hat z_{c,m}^{r,+},z_{q,m}^{r,+}).
\]

Lấy trọng số support-only \(w_{c,r}\):

\[
S_c^{+}(q)=\sum_{r=1}^{R}w_{c,r}s_{c,r}^{+}(q).
\]

### 9.2. Exclusive / otherness score: giữ logic TEAM

Vì TEAM negative branch định nghĩa theo **các lớp đối thủ khác với c**, một generalization phù hợp là:

\[
\begin{aligned}
S_c^{-}(q)=\min_{o\ne c}\sum_{r=1}^{R}\frac{w_{o,r}}{2M}
\sum_{m=1}^{M}\Big[
&\cos(\hat z_{o,m}^{r,-},z_{q,m}^{r,+})\\
+&\cos(\hat z_{o,m}^{r,+},z_{q,m}^{r,-})
\Big].
\end{aligned}
\]

Giải thích:

- \(o\ne c\): các *other classes* trong episode, nhất quán với TEAM Eq. 8.
- \(w_{o,r}\): trọng số tính từ support của đối thủ \(o\), vì mỗi nhánh score này dùng reference của lớp đối thủ. **Đây là lựa chọn thiết kế**, không phải duy nhất; cần ablation `w_o` so với `w_c` và unweighted.
- Hệ số \(1/(2M)\) làm score ổn định hơn khi thay số tokens và hai loại so khớp.
- Với cosine distance \(d=1-\cos\), việc chuyển sang cosine similarity chỉ khác **hằng số chung** nếu các trọng số được chuẩn hóa đúng, nhưng hãy kiểm chứng với implementation chính thức.

> **Quan trọng:** Giữ đúng `min`/đối thủ \(o\ne c\) theo định nghĩa original TEAM khi reproduce baseline. Nếu thấy score negative sai chiều hoặc kết quả bất thường, phải kiểm tra lại logit convention trong code thay vì tự đổi dấu.

### 9.3. Tổng hợp và training objectives

Một logit phân loại:

\[
\ell_c(q)=\gamma_+S_c^+(q)+\gamma_-S_c^-(q).
\]

Khởi tạo \(\gamma_+=\gamma_-=1\) để gần với TEAM; hiệu chỉnh trên meta-val nếu thang đo thay đổi do chuẩn hóa \(1/M\). Có thể dùng \(\ell/\tau_s\) cho softmax theo class.

\[
p(y=c\mid q,\mathcal S)=
\frac{\exp(\ell_c(q)/\tau_s)}{\sum_{o=1}^{N}\exp(\ell_o(q)/\tau_s)}.
\]

### 9.4. Lựa chọn đơn giản hơn khi cần debug

Trong vòng thử đầu tiên có thể đánh giá **positive-only**:

\[
\ell_c^{\mathrm{pos-only}}=S_c^+.
\]

Nhưng phiên bản này **không được báo là giữ nguyên toàn bộ TEAM**. Nó chỉ là debug/ablation. Kết quả paper phải có `+exclusive` và kiểm chứng phần cải tiến khi tích hợp cả hai nhánh TEAM.

### 9.5. Phân biệt với ProtoNet

Nếu chỉ tạo \(z_r\) rồi \(s_c=\sum_r w_{c,r}\cos(z^Q_r,\mu_{c,r})\), kiến trúc là **Part-Adaptive ProtoNet**, không còn thực sự tận dụng pattern tokens/adaptation/exclusive branch của TEAM. Nên thực nghiệm cả hai:

- **EAP-Proto**: baseline phụ độc lập để xem trọng số part có ích cho prototype matching nói chung không.
- **SignTEAM/Motion–Part TEAM**: method chính tích hợp C1/C2 trên instance + exclusive TEAM.

Tránh gọi hai phương án là cùng một architecture.

---

## 10. Cải tiến tùy chọn — Hard-Pair Motion Loss

### 10.1. Motivation

Nhiều episode có các cặp sign nhìn khá giống nhau theo RGB, nhưng chuyển động, vị trí hoặc phối hợp tay khác nhau. Standard episodic cross-entropy có thể chưa nhấn đủ vào các cặp “appearance-confusable”. Tuy nhiên đây là **giả thuyết**, không phải khẳng định chắc chắn.

### 10.2. Tìm hard negative

Với query \(q\) và nhãn \(y\), lấy global appearance embedding \(a(q)\), \(a(s)\) từ RGB-only encoder đã stop-gradient hoặc teacher EMA để chọn đối thủ:

\[
n^*(q)=\arg\max_{s\in\mathcal S,\,y_s\ne y}
\cos\big(a(q),a(s)\big).
\]

Positive \(p(q)\) lấy từ support cùng lớp; 5-shot có thể chọn nearest/mean support. **Tại meta-training**, được phép sử dụng nhãn query trong loss và mining; ở meta-test không dùng query labels.

### 10.3. Triplet/margin loss trên motion-aware representations

Định nghĩa \(s_{\mathrm{MP}}(q,s)\) là score bằng motion–part representation, không phải RGB score dùng để chọn negative. Loss:

\[
\mathcal L_{\mathrm{hard}}=
\frac1{|\mathcal Q|}\sum_{q\in\mathcal Q}
\max\left(0,\delta+s_{\mathrm{MP}}(q,n^*)-s_{\mathrm{MP}}(q,p)\right).
\]

Có thể dùng supervised contrastive thay thế, nhưng không nên thay loss cùng lúc với thêm module, vì khi đó không biết cải tiến tới từ đâu.

### 10.4. Ba rủi ro

1. **False hard pair:** hai sign labels khác nhau nhưng hình/chuyển động rất giống do từ đồng nghĩa, lỗi annotation hoặc ambiguities.
2. **Mining quá sớm:** appearance encoder chưa ổn định → negative selection gần như ngẫu nhiên. Warmup cross-entropy trước khi bật hard loss.
3. **Loss quá mạnh:** overfit 1-shot, giảm generalization. Quét \(\lambda_h\) rất nhỏ trước, báo cáo confidence intervals.

### 10.5. Điều kiện giữ module

Chỉ đưa C3 vào mô hình cuối nếu cải thiện **có thể tái lập** trên cả 1-shot và 5-shot, ít nhất trên các hard-pair subsets và không gây thiệt hại lớn trên overall accuracy. Nếu không, bỏ hẳn C3 và giữ paper gồm **hai đóng góp chính**.

---

## 11. Hàm mất mát, tối ưu, và quy trình huấn luyện

### 11.1. Bản cơ sở có thể tái lập

Bám TEAM, học positive và exclusive heads:

\[
\mathcal L_{+}=\operatorname{CE}(S^+/\tau_s,y),
\qquad
\mathcal L_{-}=\operatorname{CE}(S^-/\tau_s,y).
\]

\[
\boxed{\mathcal L_{\mathrm{base}}=\mathcal L_++\mathcal L_-}
\]

Đây là điểm bắt đầu quan trọng nhất; đừng thêm regularizer trước khi model tái lập ổn định.

### 11.2. Bản mở rộng

\[
\boxed{\mathcal L=\mathcal L_++\mathcal L_-
+\lambda_h\mathcal L_{\mathrm{hard}}
+\lambda_w\mathcal L_{\mathrm{reg}}.}
\]

Một regularizer chống part-weight collapse (tùy chọn):

\[
\mathcal L_{\mathrm{reg}}=\frac1N\sum_{c=1}^{N}
\mathrm{KL}(w_c\parallel \mathrm{Uniform}(R)),
\]

nhưng đây là một **bias** hướng uniform, có thể làm mất khả năng chọn part quan trọng. Nên ưu tiên shrinkage đã mô tả ở C2 và chỉ thêm KL nếu có lý do thực nghiệm.

### 11.3. Những tham số cần học

- Backbone RGB \(g_\theta\): freeze/partial-finetune/full-finetune là **biến thí nghiệm**, không thay đổi chỉ ở mô hình proposed.
- Global pattern bank \(P\) và module attention/MLP TEAM.
- Landmark motion encoder \(E_m\), \(W_m\), \(W_b\), part embeddings \(e_r\), motion bias \(u_m^r\).
- Part-score calibration (nếu học): temperature/gamma, nhưng nên tránh học quá nhiều scalar trên meta-test.

### 11.4. Chu trình training khuyến nghị

**Giai đoạn 0 — chuẩn bị:** xây episode sampler, split theo class, thống kê missing landmark/clip length/signer, tái lập TEAM RGB-only.

**Giai đoạn 1 — add motion:** thêm C1, giữ C2 uniform. Chạy `TEAM+RGB/landmark concat` làm đối chứng.

**Giai đoạn 2 — add part:** bật support-only part weights C2, train/evaluate; sau đó thêm reliability/shrinkage có kiểm soát.

**Giai đoạn 3 — optional:** thử C3 sau warmup hoặc fine-tune, không mặc định dùng.

**Giai đoạn 4 — cố định protocol:** chọn checkpoint bằng meta-val, đóng băng thiết kế/hyperparameter rồi meta-test nhiều seeds.

### 11.5. Hyperparameter khởi tạo (giả thuyết, không phải TEAM settings chính thức)

```yaml
experiment:
  way: 5
  shots: [1, 5]
  frames: 8                    # ablation 16
  image_size: 224
  episodes_eval: 10000         # nếu theo được giao thức
  random_seeds: [11, 22, 33, 44, 55]

model:
  rgb_backbone: resnet50      # reproduce TEAM; ViT-B là ablation
  feature_dim: 256           # cần adapter nếu backbone ra 2048/768
  pattern_tokens: 16         # placeholder của bản lightweight, CHƯA phải optimum
  parts: [left, right, relation]
  motion_encoder: mlp
  motion_guidance: true
  part_matching: support_adaptive
  exclusive_branch: true
  keypoints_confidence: true

part_weights:
  temperature: 0.20
  reliability_lambda: 0.5
  uniform_shrinkage: 0.20

optimization:
  optimizer: adamw            # khởi tạo cho version mới, so với SGD TEAM
  lr: 0.0001                 # quét riêng backbone/head
  weight_decay: 0.01
  hard_pair_lambda: 0.0      # bật sau ablation
  use_amp: true
  train_iterations: 10000   # đối chứng theo budget TEAM, không phải kết quả
```

Cảnh báo: **TEAM gốc dùng 50–80 pattern tokens trong nhiều cấu hình paper**, trong khi `pattern_tokens: 16` là ví dụ bản lightweight và có thể underperform. Phải sweep \(M\) (ví dụ 8/16/32/60) và so ở cùng hoặc có kiểm soát chi phí.

---

## 12. Suy luận với novel classes: ví dụ đầy đủ

Giả sử episode gồm 5 lớp ký hiệu A/B/C/D/E, mỗi lớp chỉ có 1 support video, và 1 query video chưa có nhãn.

### Bước 1 — Nhận support và query

- `S_A ... S_E`: 5 RGB clips và landmark tracks.
- `Q`: query clip.
- Không có gradient-update từ nhãn query.

### Bước 2 — Encode video

Cùng một RGB encoder và motion encoder sinh \(F^a\), \(b_t^L\), \(b_t^R\), \(b_t^B\), confidence cho từng clip. Motion-guided cross-attention tạo `L/R/B × instance/exclusive tokens`.

### Bước 3 — TEAM support adaptation

Thích nghi instance/exclusive token references của 5 support classes, vẫn phân biệt riêng từng part.

### Bước 4 — Ước lượng support-only part importance

Ví dụ minh họa, với class A:

\[
\boldsymbol w_A=(0.60,0.15,0.25).
\]

Class B có thể khác:

\[
\boldsymbol w_B=(0.12,0.65,0.23).
\]

Các giá trị chỉ là **minh họa**, không phải kết quả đạt được.

### Bước 5 — Score query

Tính similarity token-wise theo từng part, trên cả nhánh instance/exclusive. Mỗi class dùng trọng số tương ứng với class/support, rồi tổng hợp logits \(\ell_A,\ldots,\ell_E\).

### Bước 6 — Dự đoán

\[
\hat y(Q)=\arg\max_{c\in\{A,B,C,D,E\}}\ell_c(Q).
\]

**Không có bước 7 “fine-tune query”**, cũng không tối ưu lại backbone bằng test episode labels. Nếu muốn nghiên cứu transductive adaptation ở tương lai phải tách như một protocol khác.

### 12.1. Tại sao query không được tính part weights?

Trong proposed core method, giả thuyết là *class discriminativeness* được xác định bởi **cấu trúc tương phản giữa các support classes**. Để tránh thông tin query ảnh hưởng chọn trọng số, ta giữ class/part weights hỗ trợ-only. Query vẫn có **motion-guided attention riêng** từ clip của chính nó, nhưng attention này không sử dụng class label.

### 12.2. K=5 khác K=1 như thế nào?

- Có thể lấy trung bình nhiều support để ổn định prototype/reference.
- Độ tin cậy \(\rho\) trung bình qua 5 clip sẽ ổn định hơn.
- Có thể tính intra-class variation như ablation riêng; nhưng không trộn quy trình 1-shot và 5-shot không kiểm soát.

---

## 13. Thuật toán và tensor shapes

### 13.1. Algorithm 1 — Motion–Part FSL (pseudocode)

```text
INPUT:
    episode support S = {(v_c,k, c)} for c=1..N, k=1..K
    query clips Q
    pretrained RGB encoder, motion encoder,
    shared TEAM pattern bank P and adaptation operator A_TEAM

FOR each support/query video v:
    X, H, valid_mask <- decode_and_extract_synchronized_inputs(v)
    F_appearance <- RGBEncoder(X)                  # [T, Da]
    FOR r in {L, R, Relation}:
        B_r, confidence_r <- MotionFeatures(H,r)   # [T,Dm], [T]
        V_r <- FuseAppearanceMotion(F_appearance,B_r)
        attn_r <- MotionGuidedAttention(P,V_r,B_r,confidence_r)
        Z_plus_r, Z_minus_r <- TEAMStylePatternAggregation(P, V_r, attn_r)
    END
END

FOR each support class c and part r:
    class_reference(c,r) <- aggregate_K_support_tokens(S_c, r)
    adapted_reference(c,r) <- A_TEAM(class_reference(c,r), competitors, part=r)
END

FOR each c,r:
    D(c,r) <- 1 - max_{o != c} token_similarity(adapted_reference(c,r),
                                                adapted_reference(o,r))
    reliability(c,r) <- mean support valid confidence
    w(c,r) <- reliability_aware_shrunk_softmax(D(c,:), reliability(c,:))[r]
END

FOR each query q and class c:
    score_pos(c,q) <- sum_r w(c,r)*positive_pattern_similarity(c,q,r)
    score_neg(c,q) <- TEAM_otherness_score_with_parts(c,q, all_competitors)
    logit(c,q) <- gamma_pos*score_pos(c,q) + gamma_neg*score_neg(c,q)
END

RETURN argmax_c logit(c,q) for each query

TRAINING ONLY:
    compute episodic CE for positive and negative scores
    optionally add hard-pair margin loss and regularization
    backpropagate through encoders, attention and pattern bank
```

### 13.2. Tensor shapes trong batch episodic

Đặt \(B\)=số episode/batch, \(N\)=way, \(K\)=shot, \(U\)=queries/class, \(R=3\), \(T\)=frames, \(M\)=pattern tokens, \(D\)=latent dimension:

```text
support_rgb           [B, N, K, T, 3, H, W]
query_rgb             [B, N*U, T, 3, H, W]
support_landmarks     [B, N, K, T, J, C]
query_landmarks       [B, N*U, T, J, C]
support_appearance    [B, N, K, T, Da]
support_motion        [B, N, K, R, T, Dm]
support_patterns_pos  [B, N, K, R, M, D]
support_patterns_neg  [B, N, K, R, M, D]
class_refs_pos        [B, N, R, M, D]
class_refs_neg        [B, N, R, M, D]
query_patterns_pos    [B, N*U, R, M, D]
query_patterns_neg    [B, N*U, R, M, D]
part_weights          [B, N, R]
score_per_part        [B, N*U, N, R]
logits                [B, N*U, N]
```

Đầu vào thực tế của repo TEAM có thể flatten `T` vào batch, sau đó reshape trở lại. Khi sửa code nên giữ một adapter để chuyển tensor shape nhất quán; đặc biệt kiểm tra `spt_labels` không bị lệch thứ tự sau `view/reshape`.

### 13.3. Độ phức tạp khái quát

- Extract RGB features: \(O(T\cdot\operatorname{Cost}(g_\theta))\).
- Motion feature extraction: khoảng tuyến tính theo \(T\times J\) cho các phép hình học cơ bản.
- Motion-guided cross-attention: \(O(RMTD)\) nếu dùng linear projections đơn giản, bỏ qua hằng số/head details.
- Token-wise query–class matching: \(O(NRMD)\) cho positive; otherness branch có thêm so sánh với competitors và có thể là \(O(N^2RMD)\) khi triển khai ngây thơ, cần vectorization.
- **Không** cần \(O(T^2D)\) cho frame-to-frame alignment trong matching support–query, nhưng thời gian thực phải đo thực nghiệm vì vision encoder và ROI/detector có thể chiếm phần lớn chi phí.

---

## 14. Gợi ý triển khai bằng PyTorch

### 14.1. Các file gốc TEAM cần đọc

| File TEAM gốc | Ý nghĩa | Thay đổi dự kiến |
|---|---|---|
| `model.py` | Backbone, `TEAM`, forward, logits và loss | Tạo `SignTEAM` subclass/wrapper, nhận thêm landmarks và trả về part-wise logits |
| `model_util.py` | `Discriminative_Pattern_Matching`, attention, pattern bank, support adaptation | Tách aggregator có thể nhận attention bias/part features, giữ nguyên bản baseline để reproduce |
| `preprocess.py` / `video_reader.py` | Video preprocessing | Đồng bộ frame timestamps, landmark extraction, valid mask và cache |
| `run_train.py`, `run_eval.py` | Episodic training/evaluation | Thêm dataset WLASL/VSL, sampler, config, ablation flags, logging |
| `splits/` | Class splits | Tạo split độc lập cho WLASL/VSL; khóa file split + seed |

### 14.2. Cấu trúc dự án đề xuất

```text
signteam/
  configs/
    team_rgb.yaml
    concat.yaml
    mgpr_only.yaml
    eapm_only.yaml
    motion_part_full.yaml
  datasets/
    wlasl.py
    vsl.py
    episodic_sampler.py
    landmark_cache.py
  models/
    team_original_adapter.py
    rgb_encoder.py
    motion_encoder.py
    motion_guided_patterns.py
    support_adaptation.py
    part_weighting.py
    sign_team.py
  losses/
    episodic.py
    hard_pair.py
  scripts/
    train.py
    evaluate.py
    audit_data.py
    ablate.py
  tests/
    test_shapes.py
    test_support_only.py
    test_missing_landmarks.py
    test_invariance.py
  reports/
    results.csv
    seedwise_accuracy.csv
    complexity.csv
```

### 14.3. Pseudocode cho part weighting (có thể chuyển thẳng sang PyTorch)

```python
import torch
import torch.nn.functional as F


def support_part_weights(
    support_pos,        # [B, N, R, M, D] (đã adapt)
    reliability,        # [B, N, R] support-only in [0,1]
    temp=0.2,
    rho_coeff=0.5,
    shrink=0.2,
    eps=1e-6,
):
    B, N, R, M, D = support_pos.shape
    x = F.normalize(support_pos, dim=-1)

    # same pattern m, same part r between classes
    sim = torch.einsum('bnrmd,bormd->bnor', x, x) / M
    # mask chính class trước nearest competitor
    same = torch.eye(N, device=x.device, dtype=torch.bool)[None, :, :, None]
    sim = sim.masked_fill(same, float('-inf'))
    nearest = sim.max(dim=2).values     # [B,N,R]
    discr = 1.0 - nearest

    quality_logits = rho_coeff * torch.log(reliability.clamp_min(eps))
    logits = discr / temp + quality_logits
    weights = F.softmax(logits, dim=-1)
    weights = (1.0-shrink)*weights + shrink/R
    return weights, discr
```

**Ghi chú kỹ thuật:** code này giả sử \(N>1\); nếu một part mất toàn bộ landmark và reliability bằng 0 cần chính sách masking/fallback riêng trước khi áp dụng softmax. Cần kiểm tra `supports_pos` được adapt đúng như TEAM gốc và `cosine` không bị NaN. Đây là đoạn code mẫu để giải thích, **chưa kiểm thử end-to-end**.

### 14.4. Motion-guided attention (pseudocode rõ shape)

```python
# q_part:  [B, R, M, D]
# k_part:  [B, R, T, D]
# v_part:  [B, R, T, D]
# bias:    [B, R, M, T]
# valid:   [B, R, T]  -> policy fallback cho all-invalid

logits = torch.einsum('brmd,brtd->brmt', q_part, k_part) / (D ** 0.5)
logits = logits + bias
# chỉ masked_fill khi có ít nhất một frame hợp lệ mỗi [B,R]
attention = torch.softmax(logits, dim=-1)
aggregated = torch.einsum('brmt,brtd->brmd', attention, v_part)
positive = q_part + aggregated
negative = q_part - aggregated
# tiếp tục qua TEAM MLP + support adaptation + part-wise matching
```

**Kiểm thử bắt buộc**:

1. Tổng attention theo \(T\) bằng 1 cho mỗi token/part.
2. Với `lambda_motion=0`, MGPR có thể suy biến hợp lý về attention RGB để làm sanity check.
3. Support/query hoán đổi thứ tự trong batch nhưng cùng nhãn vẫn cho kết quả nhất quán.
4. Thay đổi class IDs nhưng giữ cấu trúc episode không làm đổi prediction sau relabel.
5. Mất toàn bộ landmarks vẫn chạy không NaN.
6. `no query labels` trong forward inference.

---

## 15. Thiết kế tập dữ liệu, chia lớp và chống leakage

### 15.1. Trước khi training

Xây script kiểm kê:

- Số class và valid videos/class.
- Clip duration, FPS, resolution, signer IDs nếu có.
- Tỉ lệ videos có phát hiện ít nhất một tay/hai tay.
- Phân bố valid-landmark ratio theo class và video length.
- Duplicates/near-duplicates: re-encode, resize, checksum/perceptual hash nếu có thể.
- Clips có thẻ label sai hoặc hand detections chất lượng thấp; lưu manifest để người khác reproduce.

### 15.2. Class-disjoint split

Không dùng các novel test classes để meta-train bất kỳ module học nào. Nếu class ít video, phải đảm bảo lớp đó có đủ video để tạo support/query **khác nhau**. Không tái sử dụng cùng clip vừa làm support vừa làm query trong một episode.

Ví dụ manifest:

```csv
video_id,dataset,class_id,signer_id,split,frame_count,landmark_coverage,sha256
v001,WLASL,C101,S01,meta_train,48,0.97,...
v002,WLASL,C101,S02,meta_train,52,0.94,...
v201,VSL,C004,unknown,meta_test,64,0.88,...
```

Đây là **minh họa định dạng**, không phải IDs có thật.

### 15.3. Signer leakage

Class-disjoint **khác** signer-disjoint. Nếu cùng signer xuất hiện ở train/test classes, mô hình có thể tận dụng identity/style. Cần ghi rõ:

- Official/new few-shot class split có signer overlap hay không.
- Nếu metadata cho phép, tạo **signer-disjoint robustness evaluation** bổ sung.
- Nếu VSL thiếu signer IDs, không tuyên bố signer-independent khi chưa xác minh.

### 15.4. Fair comparison

Giữa các phương pháp phải cố định:

- Split, episode IDs, support/query clips, resolution, frame timestamps và số frame.
- Backbone pretraining và trạng thái freeze/fine-tune.
- Data augmentation và training budget; số params bổ sung được báo cáo.
- Checkpoint selection/seed count.
- Landmark extractor và chi phí inference được tính riêng hoặc công bố rõ.

### 15.5. Chỉ số báo cáo

Accuracy trung bình và 95% confidence interval theo episode, nhưng phải cẩn thận vì các episode có thể phụ thuộc cùng tập videos/classes. Ưu tiên báo cáo **mean ± variation qua nhiều run/seeds**, đồng thời thêm episode-level bootstrap như phân tích bổ sung; không coi 10.000 episode là 10.000 mẫu hoàn toàn độc lập.

---

## 16. Ablation có đối chứng

### 16.1. Main ablation matrix

| ID | Model | Motion-conditioned attention | Part-adaptive weights | TEAM exclusive | Hard loss | WLASL 1/5 | VSL 1/5 |
|---|---|:---:|:---:|:---:|:---:|---|---|
| A0 | TEAM original reproduced (RGB-only) | — | — | ✓ | — | TBD | TBD |
| A1 | TEAM + landmarks concat | — | — | ✓ | — | TBD | TBD |
| A2 | TEAM + motion guidance, no part weighting | ✓ | — | ✓ | — | TBD | TBD |
| A3 | TEAM + uniform part-wise matching | — | Uniform | ✓ | — | TBD | TBD |
| A4 | TEAM + support-adaptive part matching | — | ✓ | ✓ | — | TBD | TBD |
| **A5** | **Motion–Part FSL (C1+C2)** | **✓** | **✓** | **✓** | — | **TBD** | **TBD** |
| A6 | Motion–Part FSL + hard-pair loss | ✓ | ✓ | ✓ | ✓ | TBD | TBD |
| A7 | A5 without exclusive branch | ✓ | ✓ | — | — | TBD | TBD |

`TBD` nghĩa là **chưa chạy**. Ký hiệu 1/5 là báo cáo riêng 1-shot và 5-shot, không phải một số gộp.

### 16.2. MGPR ablations

| Thí nghiệm | Biến số cô lập | Câu hỏi |
|---|---|---|
| Motion-off | Bỏ \(b_t^r\) hoàn toàn | Có thực sự cần motion không? |
| Early/late concat | Vẫn dùng cùng RGB + landmarks | Guidance hơn concatenation đơn giản? |
| Bias-only | Motion chỉ ảnh hưởng attention logits | Novelty nằm ở temporal attention hay feature richness? |
| Key-only | Motion ảnh hưởng keys | Motion định vị chứng cứ tốt hơn? |
| Value-only | Motion chỉ cộng vào values | Nếu bằng full model, attention guidance có đáng giữ không? |
| 2D vs 3D landmarks | Thay representation | Độ sâu có tăng đáng kể? |
| velocity only vs pose+velocity | Bỏ tọa độ tuyệt đối/body-relative | Có mất handshape/location? |
| with/without acceleration | Bổ sung đạo hàm bậc hai | Có nhạy với noise? |
| 8 vs 16 frames | Temporal resolution | Motion cần nhiều frames hơn? |

### 16.3. EAPM ablations

| Thí nghiệm | Trọng số | Vai trò |
|---|---|---|
| Uniform | \(1/R\) | Baseline tối thiểu |
| Global learned | \(w_r\) học từ train và cố định ở test | Xem support adaptation có thực sự cần |
| Episode-global | \(w_r^{\mathcal E}\) chung cho cả episode | Cần class-wise hay chỉ episode-wise? |
| Class-adaptive | \(w_{c,r}^{\mathcal E}\) | C2 đầy đủ |
| + confidence | Weight có \(\rho\) | Chống nhiễu landmark |
| + shrinkage | thêm \(\eta/R\) | Chống overfitting 1-shot |
| No relation part | Chỉ L/R | Vai trò của quan hệ hai tay |
| L/R/relation + face | \(R=4\) | Non-manual cues có ích không? |

### 16.4. TEAM fidelity ablations

- TEAM positive-only vs positive+exclusive.
- Support adaptation original on/off.
- `mean before DPM` vs `mean after DPM`.
- Giữ nguyên \(M\) vs giảm \(M\) khi nhân thêm 3 parts.
- Shared pattern bank giữa parts vs separate banks (đánh đổi params/semantic consistency).
- Positive/negative similarity: dùng \(w_c\), \(w_o\), hoặc uniform trong negative branch.
- Pattern count \(M\in\{8,16,32,60\}\) tùy khả năng GPU và tính công bằng.

### 16.5. Lưu ý về benchmark

Không nên lấy kết quả TEAM gốc trên HMDB/Kinetics rồi so trực tiếp với kết quả WLASL/VSL của mô hình proposed. Cần **reproduce TEAM trên chính WLASL và VSL** và cùng split/training protocol.

---

## 17. Phân tích sâu, kiểm chứng giả thuyết

### 17.1. Hard sign pairs

Chọn cặp sign dễ nhầm theo appearance-only feature trong **meta-test** để phân tích (không dùng chúng để tuning mô hình). Tách lỗi thành:

1. Cùng handshape, khác motion direction.
2. Cùng motion, khác handshape/location.
3. Tay trái/tay phải khác nhau.
4. Khác quan hệ hai tay.
5. Khác facial/non-manual cue (để thấy giới hạn mô hình 3 parts).

Báo cáo confusion matrix, Δ accuracy từng nhóm và qualitative clips/heatmaps có bản quyền sử dụng hợp lệ.

### 17.2. Temporal attention visualization

Vẽ \(\alpha_{m,t}^r\) cho một vài token quan trọng; đối chiếu với đồ thị vận tốc tay và frames. Tránh kết luận “token đã học semantic phase chuẩn” chỉ từ heatmap; muốn claim tương quan với chuyển động cần thiết kế metrics như overlap với vùng động học/phân biệt được annotation thủ công.

### 17.3. Part-weight visualization

Cùng một sign class trong hai episode khác nhau: giữ support sign đó không đổi, thay 4 class cạnh tranh và vẽ \(w_{c,L},w_{c,R},w_{c,B}\). Kiểm tra class-aware support adaptation có thật tạo sự thay đổi theo competitor.

### 17.4. Robustness

- Gaussian jitter cho landmark tọa độ đã chuẩn hóa với các mức nhỏ/vừa/lớn.
- Missing-hand: drop một tay, một số frame, liên tiếp một đoạn.
- Detector confidence miscalibration.
- Vary FPS/sampling stride: giữ quỹ đạo nhưng đổi tốc độ.
- Background/appearance shift: quần áo, camera, crop.
- One-hand vs two-hand signs (nếu có metadata hoặc gán nhóm đáng tin).

### 17.5. Complexity

Báo cáo tối thiểu:

- Trainable params: backbone và phần added head.
- Total params và memory peak.
- FLOPs ước lượng cho RGB encoder, motion encoder, token aggregation, matching.
- Milliseconds/video end-to-end trên GPU xác định, có/không tính landmark extraction (cần ghi rõ).
- Thời gian query khi tăng \(T\), \(M\), \(N\), \(R\).

Sẽ không đúng nếu nói “alignment-free nên không tốn thời gian hơn TEAM”: số parts nhân ~3 có thể làm aggregation và matching chậm hơn, dù không có frame-pair alignment.

---

## 18. Rủi ro kỹ thuật và cách xử lý

| Rủi ro | Tại sao nguy hiểm | Cách chẩn đoán | Phản ứng |
|---|---|---|---|
| Motion signal yếu/nhiễu | MediaPipe không ổn định ở chuyển động nhanh/che khuất | Coverage theo class, jitter tests | Mask, smoothing ngắn, confidence, RGB fallback |
| Motion chỉ duplicate RGB | Thêm params mà không tạo thông tin bổ sung | Concat vs MGPR, feature/attention ablation | Giảm/tắt motion branch nếu không có lợi |
| Trọng số 1-shot quá sắc | Một support video không đại diện lớp | Entropy weights, episode sensitivity | Shrinkage, temperature, class-independent fallback |
| 3 parts quá ít | Một số sign dựa vào face/body | Nhóm lỗi non-manual, face ablation | Thêm part sau, không lạm dụng trong v1 |
| Separate bank không học | Tokens giữa support/query mất ổn định | Token cosine/attention maps | Global shared bank + learned part embedding |
| Exclusive branch sai | TEAM score phụ thuộc sign convention và competitors | Sanity-test against original code | Giữ nguyên code baseline, unit tests |
| Fairness kém | Proposed dùng nhiều modalities/params hơn | Parameter-matched baselines | So `concat+MLP`, report cost, equal data |
| Leakage signer/data | Accuracy tăng giả | Class/signer audit, duplicate check | Công bố split và có bổ sung signer-disjoint |
| Too many components | Khó thuyết phục novelty và causal attribution | Main ablation A0–A6 | Chốt hai core modules, hard loss optional |
| Overclaim SOTA | WLASL/VSL nhiều giao thức | Protocol table | Không so các con số không cùng split |

---

## 19. Các phiên bản khả thi và lộ trình

### 19.1. Phiên bản P0 — Reproduction (bắt buộc)

- TEAM original RGB-only, ResNet-50/ViT-B chọn rõ, 8 frames, class-disjoint episodic protocol.
- Prototype simple baseline.
- Run trên WLASL rồi VSL với manifests đã audit.
- Đầu ra: reproducible metrics và config YAML, code checkpoint có thể load lại.

### 19.2. P1 — Motion guidance only

- RGB+landmark inputs.
- Motion feature encoder; MGPR-A (bias-only) hoặc MGPR-B (key+bias) trước.
- So với `TEAM+concat`, motion-only, missing-landmark fallback.
- Chỉ đi tiếp khi có xu hướng tốt từ nhiều seeds, không chỉ một episode/test split.

### 19.3. P2 — Part-aware + support weighting

- Part L/R/B, TEAM style positive/negative per part.
- Uniform weights → class-adaptive D → reliability → shrinkage.
- Ablate per-part vs global, visualise w changes across episodes.

### 19.4. P3 — Full TEAM extension

- TEAM support adaptation giữ nguyên về ý nghĩa.
- Motion-conditioned token aggregation + support-aware part matching.
- Positive/exclusive branch integrated và test invariance.
- Đo latency/params/peak memory.

### 19.5. P4 — Optional hard-pair

- Mining theo appearance similarity, loss margin.
- Cần gains độc lập từ hard-pairs và overall; nếu không đạt, bỏ module.

### 19.6. Mốc quyết định

| Gate | Điều kiện đi tiếp | Nếu không đạt |
|---|---|---|
| G0 | TEAM baseline tái lập, data sạch, labels/splits đúng | Sửa data và code trước |
| G1 | MGPR vượt concat baseline hợp lý, không phụ thuộc seed đơn lẻ | Dùng motion đơn giản hoặc không dùng |
| G2 | EAPM vượt uniform và learned global | Giảm complexity của weighting |
| G3 | C1+C2 hợp tác và không suy giảm lớn ở 5-shot | Khảo sát giao thoa/score scale |
| G4 | Gain rõ trên similar-sign subset + robustness | Thu hẹp research claim |
| G5 | C3 có ích rõ ràng | Bỏ C3 khỏi main model |

---

## 20. Đóng góp, novelty, và cách trình bày bài báo

### 20.1. Câu chuyện nghiên cứu nhất quán

**Problem:** Few-shot isolated sign recognition đặc biệt nhạy với motion và vùng cơ thể có tính phân biệt; cùng một class mới có thể cần các vùng quan sát khác nhau tùy tập lớp cạnh tranh.  
**Gap (hypothesis):** Generic global pattern aggregation và matching có thể không tận dụng hết fine-grained hand kinematics và episode-dependent part importance.  
**Method:** Motion-guided pattern representation + support-driven discriminative part weighting trong TEAM-style alignment-free matching.  
**Evidence required:** Gains hơn TEAM và concat khi cùng input, phân tích hard-pair/part switching, robustness, efficiency.

### 20.2. Câu diễn đạt đóng góp (chỉ dùng khi có số liệu xác nhận)

> *We introduce a motion-guided pattern representation that conditions alignment-free pattern aggregation on fine-grained hand kinematics, enabling sign-specific evidence extraction without pairwise temporal alignment.*

> *We develop a support-driven, class-specific part weighting strategy that adaptively emphasizes discriminative left-hand, right-hand, and inter-hand relation cues for each novel episode.*

> *Experiments on within-dataset few-shot WLASL and VSL benchmarks investigate motion guidance, episode-dependent part importance, robustness to missing landmarks, and matching efficiency.*

Không viết “consistently outperforms SOTA” hay “novel first” khi chưa có reproducible results và literature review cập nhật.

### 20.3. Nên viết Method thành mấy phần?

```text
3. Method
  3.1. Problem Formulation
  3.2. Overview and TEAM-based Pattern Matching
  3.3. Motion-Guided Pattern Representation
      3.3.1. Kinematic feature extraction
      3.3.2. Motion-conditioned token aggregation
  3.4. Episode-Adaptive Part Matching
      3.4.1. Part-wise class references and TEAM adaptation
      3.4.2. Support-driven part discriminativeness
      3.4.3. Reliability-aware weighting
  3.5. Objective and Inference
```

`Hard-Pair Loss` chỉ cần một đoạn nhỏ hoặc Supplementary nếu được giữ; không để nó che khuất hai đóng góp trung tâm.

### 20.4. Điểm phản biện rất có thể hỏi

**Q1. Tại sao không chỉ concat RGB và landmarks?**  
→ Phải chứng minh motion vào *attention/token formation* mang lợi ích riêng bằng ablation feature-matched/parameter-matched.

**Q2. TEAM đã có support adaptation, vậy thêm part weighting có gì mới?**  
→ TEAM adaptation tách shared information giữa classes cho token representations; C2 học **cấu trúc độ quan trọng theo bộ phận, phụ thuộc đối thủ trong episode**, trực tiếp thay class score. Cần nhấn mạnh hai loại adaptation khác nhau.

**Q3. Landmarks thêm annotation ngoài nhãn sign?**  
→ Keypoints do pretrained pose estimator sinh tự động; báo rõ mô hình external/dữ liệu pretraining và pipeline inference. Đừng nói “không dùng external data” nếu dựa vào pretrained pose estimator.

**Q4. Vì sao chỉ left/right/relation mà không có face/body?**  
→ Thiết kế v1 tập trung thủ thuật chủ đạo; để ablation face/body và chỉ ra giới hạn trên non-manual signs.

**Q5. Reliability có dùng query không?**  
→ Class weights lấy từ support; query validity chỉ được dùng nếu thực sự cần cho **query-side feature extraction/fallback**, không dùng để tái tối ưu class discriminativeness.

**Q6. Few-shot là training ít dữ liệu hay chỉ test ít support?**  
→ Meta-training dùng **nhiều base videos/classes**, meta-testing yêu cầu **ít support mỗi novel class**. Làm rõ ngay trong Problem Formulation.

**Q7. Có thể claim preserved alignment-free?**  
→ Có: per-video temporal attention không phải support–query temporal alignment. Tuy nhiên chứng minh complexity và ghi rõ token aggregation phụ thuộc \(T\).

### 20.5. Không gộp nhầm với hướng Episode-Adaptive Pattern Routing

Một đề xuất TEAM khác đã xuất hiện trước đây dùng:

1. Support-conditioned **pattern semantics specialization**.
2. Dynamic **pattern routing** theo mức phân biệt.
3. **Class-conditioned query parsing**.

Đây là **một hướng khác** về *episode-adaptive global pattern bank*, có thể làm phương pháp/paper riêng. Hướng Motion–Part FSL đang mô tả tập trung vào motion evidence và part-level matching; không yêu cầu cả ba thành phần routing trên. Có thể đưa vào phần future work hoặc additional experiments nếu sau này thực sự kết hợp và kiểm chứng.

---

## 21. Nguồn tham khảo

**[1]** SuBeen Lee, WonJun Moon, Hyun Seok Seong, Jae-Pil Heo. **Temporal Alignment-Free Video Matching for Few-shot Action Recognition**. *CVPR 2025*, pp. 5412–5421.  
- [CVF official paper page](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Temporal_Alignment-Free_Video_Matching_for_Few-shot_Action_Recognition_CVPR_2025_paper.html)  
- [Paper PDF on arXiv](https://arxiv.org/pdf/2504.05956)  
- [Official code — leesb7426/TEAM](https://github.com/leesb7426/TEAM)

**[2]** Tài liệu nghiên cứu cá nhân đã trao đổi ngày 16/10/2026:  
- `Motion_Part_FSL_WLASL_VSL_Research_Proposal.pptx`  
- `Motion_Part_FSL_WLASL_VSL_Updated_Pipeline_LaTeX.pptx`  
- `Motion_Part_FSL_Redesigned_Research_Deck.pptx`  
Nội dung chính: two-core-module design, RGB/hand landmarks, 3-part evidence, support-only weighting, WLASL/VSL independent protocols, optional hard pairs, risk-driven ablations.

**[3]** Hướng nghiên cứu TEAM khác (không phải lõi của tài liệu hiện tại):  
- `Episode_Adaptive_TEAM_Method_Deck.pptx`  
- `Episode_Adaptive_TEAM_Method_Deck_v2.pptx`  
Nội dung: support-conditioned pattern specialization, routing, class-conditioned query parsing.

> **Ranh giới dẫn nguồn:** phần mô tả TEAM ở §3 lấy theo paper/code gốc. Từ §6 đến §20 là bản thiết kế chi tiết hóa từ đề xuất Motion–Part FSL và các quyết định triển khai mới, **không phải phương pháp đã được TEAM/CVPR 2025 công bố**. Những con số ví dụ, hyperparameters chưa có thực nghiệm đều được ghi là minh họa/đề xuất.

---

## 22. Checklist trước khi kết luận phương pháp hiệu quả

- [ ] Đã tái lập TEAM RGB-only trên đúng split WLASL và VSL.
- [ ] WLASL và VSL chạy độc lập; không trộn train/test cross-language.
- [ ] Đã khóa class IDs và metadata support/query.
- [ ] Không có trùng clip/near-duplicate giữa support và query.
- [ ] Kiểm tra signer overlap và công bố limitations.
- [ ] RGB frames và landmark timestamp thực sự đồng bộ.
- [ ] Confidence/missing keypoints được xử lý mà không NaN.
- [ ] C1 vượt RGB+landmark concat baseline trong so sánh tương đương.
- [ ] C2 vượt uniform/global part weights; không chỉ cải thiện 1 seed.
- [ ] Exclusive branch và original TEAM support adaptation được tích hợp đúng.
- [ ] Không dùng nhãn query để tạo support class weights tại meta-test.
- [ ] Đã đo 5-way 1-shot **và** 5-way 5-shot.
- [ ] Đã đo similar-sign subset và case study motion directions.
- [ ] Đã thử noise/missing-hand robustness.
- [ ] Báo cáo total/trainable params, GPU, latency và FLOPs.
- [ ] Hard-pair loss có ablation riêng; không tự nhận là phần đóng góp nếu chưa hiệu quả.
- [ ] Không tuyên bố best/SOTA/first nếu chưa đối chiếu literature và protocol.

---

### Kết luận ngắn

**Motion–Part FSL không đơn giản là TEAM + MediaPipe landmarks.** Thiết kế tập trung hai thay đổi ở **cơ chế biểu diễn và suy luận**: (1) chuyển động tay tham gia điều khiển cách học các pattern tokens, (2) mức đóng góp của mỗi bộ phận do khả năng phân biệt novel classes trong **support episode** quyết định. Hai cải tiến này được kiểm tra trên nền TEAM với instance/exclusive tokens, support adaptation và token-wise, alignment-free matching. Phần còn lại (hard-pair loss, reliability calibration, face/body mở rộng) chỉ nên được giữ khi có bằng chứng ablation.
