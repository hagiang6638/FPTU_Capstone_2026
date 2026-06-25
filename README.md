<div align="center">

# Đồ án tốt nghiệp: Nhận dạng Ngôn ngữ Ký hiệu Việt Nam


</div>

## 📑 1. Abstract / Tóm tắt

Hệ thống nhận dạng ngôn ngữ ký hiệu (Sign Language Recognition - SLR) đóng vai trò làm cầu nối giao tiếp quan trọng giữa cộng đồng người khiếm thính và người nghe bình thường. Tại Việt Nam, nghiên cứu về nhận dạng ngôn ngữ ký hiệu (VSL - Vietnamese Sign Language) vẫn còn nhiều hạn chế, đặc biệt là ở mảng nhận dạng câu liên tục (Continuous Sign Language Recognition - CSLR) do sự phức tạp về cấu trúc ngữ pháp và sự thiếu hụt các tập dữ liệu chuẩn. 

Đồ án này đề xuất một giải pháp toàn diện cho cả hai bài toán: **Nhận dạng từ vựng riêng lẽ (ISLR)** và **Nhận dạng ngôn ngữ ký hiệu liên tục (CSLR)**. Chúng tôi triển khai và tinh chỉnh nhiều kiến trúc học sâu tiên tiến (Pipeline CNN-RNN, MSKA+, Lite TCN-BiGRU, LSTM Attention). Đặc biệt, đồ án đóng góp các phương pháp tiếp cận **Transfer Learning xuyên ngôn ngữ** (từ PHOENIX-14T sang VIECSL) và **Transfer Learning xuyên bài toán** (sử dụng CTC Loss để chuyển giao tri thức từ mô hình ISLR sang CSLR), mang lại cải thiện đáng kể về độ chính xác (Accuracy) cho ISLR và giảm tỷ lệ lỗi từ (Word Error Rate - WER) cho CSLR.

## 🎯 2. Giới thiệu

Ngôn ngữ ký hiệu không chỉ đơn thuần là việc ghép các cử chỉ tay, mà là một ngôn ngữ hình thể hoàn chỉnh với ngữ pháp, biểu cảm khuôn mặt và tư thế cơ thể. Bài toán nhận dạng tự động ngôn ngữ ký hiệu được chia làm hai mảng chính:
* **ISLR (Isolated Sign Language Recognition):** Bài toán phân loại một chuỗi video ngắn thành một từ vựng ký hiệu duy nhất (như phân loại ảnh, nhưng có yếu tố thời gian).
* **CSLR (Continuous Sign Language Recognition):** Bài toán dịch một chuỗi video dài chứa nhiều ký hiệu liên tiếp thành một câu hoàn chỉnh, không có sự phân định rõ ràng (boundary) giữa các từ. Đây là thách thức lớn và gần với thực tế giao tiếp nhất.

Việc phát triển một hệ thống SLR chính xác cho ngôn ngữ ký hiệu Việt Nam (VSL) có tầm quan trọng to lớn, giúp dỡ bỏ rào cản giao tiếp, hỗ trợ giáo dục, y tế và dịch vụ công cho cộng đồng người khiếm thính tại Việt Nam.

## 💡 3. Động lực & Đóng góp

### Động lực
* **Khoảng trống dữ liệu:** Thiếu hụt các bộ dataset quy mô lớn và chuẩn hóa cho ngôn ngữ ký hiệu Việt Nam, đặc biệt là dữ liệu CSLR.
* **Khoảng trống nghiên cứu:** Hầu hết các nghiên cứu hiện tại ở Việt Nam mới dừng lại ở việc nhận diện bảng chữ cái hoặc từ vựng cô lập (ISLR), chưa giải quyết triệt để bài toán nhận dạng câu liên tục (CSLR).
* **Giá trị xã hội:** Khát vọng xây dựng một công cụ hỗ trợ thiết thực, góp phần thúc đẩy sự hòa nhập xã hội của cộng đồng người khiếm thính.

### Đóng góp chính
1. **Phát triển & Đánh giá các mô hình Baseline cho VSL:** Xây dựng và đánh giá chi tiết các kiến trúc mạnh mẽ (MSKA+, LSTM-Attention) trên tập dữ liệu tiếng Việt.
2. **Kỹ thuật Transfer Learning đột phá:** 
   * Chứng minh hiệu quả của việc pretrain mô hình trên tập dữ liệu tiếng Đức (PHOENIX) để finetune cho tiếng Việt (VIECSL).
   * Đề xuất phương pháp luân chuyển trọng số từ mô hình học từ vựng (ISLR) để giải quyết bài toán dịch câu (CSLR) sử dụng CTC Loss.
3. **Thành lập Benchmark:** Đồ án cung cấp một trong những benchmark có tính hệ thống đầu tiên cho bài toán CSLR tiếng Việt.

## 🗂 4. Dataset

Đồ án sử dụng ba tập dữ liệu chính để phục vụ cho các phương pháp huấn luyện khác nhau:

| Dataset | Bài toán | Số lượng mẫu (Video) | Số Classes / Từ vựng | Nguồn gốc | Mô tả / Tiền xử lý |
|---------|----------|-----------------------|----------------------|-----------|--------------------|
| **VIEISL** | ISLR | *5,189* | *325* | Tự thu thập / Tổng hợp | Tập dữ liệu các từ vựng ngôn ngữ ký hiệu Việt Nam cô lập. Đã được crop, resize và chuẩn hóa số frame. |
| **VIECSL** | CSLR | *3,761* | *447* | Tự thu thập / Tổng hợp | Tập dữ liệu câu liên tục tiếng Việt. Các video chứa chuỗi ký hiệu tự nhiên, đi kèm nhãn (label) là các câu văn bản tương ứng. |
| **PHOENIX14T** | CSLR (Pretrain) | *8,257* | *1,066* | RWTH Aachen | Tập dữ liệu Benchmark ngôn ngữ ký hiệu tiếng Đức dùng để pre-train nhằm trích xuất đặc trưng không gian-thời gian mạnh mẽ. |

*(Split: Dữ liệu được chia theo tỷ lệ tiêu chuẩn Train/Val/Test là 70/15/15 hoặc 80/10/10 tùy tập dữ liệu)*


## 🛠 5. Tech Stack

<table align="center">
  <tr>
    <td align="center"><b>Deep Learning</b></td>
    <td align="center"><b>Computer Vision & Pose Estimation</b></td>
    <td align="center"><b>Data Processing & Visualization</b></td>
    <td align="center"><b>Deployment & Backend</b></td>
    <td align="center"><b>Experiment Tracking & Tools</b></td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/PyTorch_Lightning-792EE5?style=for-the-badge&logo=pytorch&logoColor=white"/>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/MediaPipe-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/Matplotlib-11577B?style=for-the-badge&logo=matplotlib&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/Uvicorn-4B8BBE?style=for-the-badge&logo=python&logoColor=white"/>
    </td>
    <td align="center">
      <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/><br>
      <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=white"/>
    </td>
  </tr>
</table>

## 📂 6. Cấu trúc dự án

```text
D:\CSLR\ĐỒ ÁN TỐT NGHIỆP\
├── app_demo/                       # Ứng dụng demo (Giao diện người dùng)
├── model/                          # Thư mục chính chứa source code huấn luyện mô hình
    ├── notebooks/                  # Tổng hợp tất cả các file notebook chạy thử nghiệm (.ipynb)
    ├── notebook1/                  # [CSLR] Train từ đầu Pipeline baseline trên VIECSL
    ├── notebook2/                  # [CSLR] Train từ đầu MSKA+ trên VIECSL
    ├── notebook3/                  # [CSLR] Pretrain Pipeline trên PHOENIX
    ├── notebook4/                  # [CSLR] Pretrain MSKA+ trên PHOENIX
    ├── notebook5/                  # [CSLR] Finetune: PHOENIX -> VIECSL (Pipeline)
    ├── notebook6/                  # [CSLR] Finetune: PHOENIX -> VIECSL (MSKA+)
    ├── notebook7/                  # [ISLR] Train Pipeline cho bài toán từ vựng trên VIEISL
    ├── notebook8/                  # [ISLR] Train MSKA+ cho bài toán từ vựng trên VIEISL
    ├── notebook9/                  # [ISLR] Train mô hình Lite TCN-BiGRU cho bài toán ISLR
    ├── notebook10/                 # [ISLR -> CSLR] Finetune mô hình Lite từ ISLR sang CSLR dùng CTC
    ├── notebook11/                 # [ISLR -> CSLR] Finetune mô hình Pipeline từ ISLR sang CSLR dùng CTC
    ├── notebook12/                 # [ISLR] Train LSTM-Attention cho bài toán từ vựng trên VIEISL
    ├── notebook13/                 # [ISLR -> CSLR] Finetune MSKA+ từ ISLR sang CSLR dùng CTC
    └── notebook14/                 # [ISLR -> CSLR] Finetune LSTM-Attention từ ISLR sang CSLR dùng CTC
```

## 🧠 7. Methodology

Đồ án ứng dụng các kiến trúc Học sâu Không gian - Thời gian (Spatio-Temporal Deep Learning) kết hợp với các kỹ thuật Alignment Sequence:

1. **Kiến trúc Pipeline (Baseline):** 
   * Backbone CNN 2D (ResNet/VGG) trích xuất đặc trưng không gian tĩnh từ mỗi frame.
   * Module Sequence (BiLSTM / BiGRU) học mối quan hệ thời gian giữa các frame.
2. **Kiến trúc MSKA+ (Multi-Scale Kernel Attention Plus):**
   * Sử dụng mạng Tích chập đa tỷ lệ (Multi-scale 1D-CNN) kết hợp cơ chế Attention để nắm bắt các cử động có tốc độ và biên độ khác nhau của tay.
3. **Mô hình Lite TCN-BiGRU:**
   * Một mô hình gọn nhẹ kết hợp Temporal Convolutional Network với Receptive Field lớn và BiGRU, phù hợp để deploy trên các thiết bị giới hạn tài nguyên.
4. **Mô hình LSTM kết hợp Attention:**
   * Cơ chế Attention giúp mô hình tập trung vào các frame chứa thông tin cốt lõi (key-frames), loại bỏ nhiễu từ các chuyển động thừa.

**Giải quyết bài toán CSLR bằng CTC Loss:**
Đối với tập dữ liệu liên tục không có nhãn thời gian cho từng từ (unsegmented data), mô hình sử dụng hàm mất mát **Connectionist Temporal Classification (CTC)** để gióng hàng (align) tự động chuỗi video đầu vào với chuỗi văn bản đầu ra mà không cần gán nhãn từng frame.

## 📊 8. Kết quả & Thảo luận

### Bảng 1: Kết quả nhận dạng từ vựng cô lập (ISLR) trên tập VIEISL
*Đánh giá dựa trên độ chính xác (Accuracy).*

| Model | Backbone / Feature Extractor | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Params (M) | Remarks |
|:---|:---|:---:|:---:|:---:|:---|
| Pipeline | 2D-CNN + BiGRU | *67.8* | *84.2* | ~25.0 | Baseline cơ bản, đóng vai trò mốc so sánh. |
| **MSKA+** | Multi-Scale CNN + Attn | **83.6** | **94.7** | ~32.5 | Thu thập tốt các cử động tay phức tạp đa tốc độ. |
| Lite TCN-BiGRU | TCN + BiGRU | *76.4* | *89.1* | **~8.2** | Tối ưu về số lượng tham số và thời gian suy luận. |
| LSTM-Attention | LSTM + Attention | *79.5* | *91.3* | ~15.4 | Hiệu năng ổn định nhờ tập trung vào Key-frames. |

### Bảng 2: Kết quả nhận dạng câu liên tục (CSLR) trên tập VIECSL
*Đánh giá dựa trên tỷ lệ lỗi từ (Word Error Rate - WER) - Càng thấp càng tốt.*

| Model | Phương pháp huấn luyện (Training Strategy) | WER (%) | Inference Time (ms/vid) | Remarks |
|:---|:---|:---:|:---:|:---|
| Pipeline | Train từ đầu (Scratch) | *51.7* | ~45ms | Khó hội tụ do thiếu dữ liệu lớn. |
| MSKA+ | Train từ đầu (Scratch) | *44.9* | ~60ms | Tốt hơn pipeline nhưng vẫn chưa tối ưu. |
| Pipeline | Finetune (PHOENIX -> VIECSL) | *37.2* | ~45ms | Pretrain trên tiếng Đức giúp trích xuất đặc trưng tốt hơn. |
| **MSKA+** | **Finetune (PHOENIX -> VIECSL)** | **31.8** | ~60ms | Mô hình đạt hiệu quả tốt nhất nhờ Transfer Learning xuyên ngôn ngữ. |
| LSTM-Attention | Finetune (VIEISL -> VIECSL) | *35.4* | ~50ms | Chuyển giao tri thức ISLR sang CSLR thông qua CTC mang lại kết quả rất hứa hẹn. |

**Thảo luận:**
* Kỹ thuật **Transfer Learning** (đặc biệt là việc lấy trọng số từ bài toán ISLR sang giải quyết CSLR và Pretrain trên PHOENIX) đã chứng minh vai trò quyết định, giúp giảm đáng kể tỷ lệ lỗi từ (WER) so với việc huấn luyện từ đầu trên một tập dữ liệu quy mô nhỏ như VSL.

## 🚀 9. Hạn chế & Hướng phát triển

### Hạn chế (Limitations)
* **Kích thước Dataset:** Dữ liệu CSLR tiếng Việt (VIECSL) vẫn còn nhỏ so với các ngôn ngữ khác (như PHOENIX-14T của Đức), dẫn đến mô hình thỉnh thoảng khó tổng quát hóa với những đối tượng mới (unseen signers).
* **Sự biến thiên nội tại (Intra-class variability):** Cùng một câu ký hiệu nhưng những người khiếm thính khác nhau có tốc độ, phương ngữ và cách biểu đạt khác biệt, làm tăng độ phức tạp cho mô hình nhận diện.
* **Tài nguyên tính toán:** Việc tối ưu hóa mô hình Spatio-Temporal cần lượng VRAM lớn và tốn nhiều thời gian huấn luyện.

### Hướng phát triển (Future Work)
* **Tích hợp Multi-Modality:** Kết hợp thêm các đặc trưng chuyên biệt về bộ xương tay (Skeleton/Keypoints qua MediaPipe) và biểu cảm khuôn mặt (Facial Expressions) vào mạng học sâu thay vì chỉ dùng ảnh RGB.
* **Kiến trúc Transformer:** Thử nghiệm thay thế RNN bằng các kiến trúc Transformer (Vision Transformer, Sign Language Transformers) để khai phá khả năng học chuỗi dài (long-term dependencies) tốt hơn.
* **Mở rộng dữ liệu:** Ứng dụng Pseudo-Labeling và Semi-Supervised Learning để tận dụng hàng ngàn video có sẵn trên Internet nhằm tự động gia tăng kích thước tập huấn luyện mà không tốn quá nhiều chi phí gán nhãn thủ công.

---
<div align="center">
  <i>Đồ án tốt nghiệp thực hiện với niềm đam mê ứng dụng Trí tuệ Nhân tạo để giải quyết các vấn đề xã hội.</i><br>
  <b>Tác giả:</b> Ngô Hà Giang, Nguyễn Thái Học, Nguyễn Thị Diễm Phương, Nguyễn Minh Nhựt
  
  <b>Giảng viên hướng dẫn:</b> Huỳnh Văn Thống, Nguyễn Hồng Hải
</div>
