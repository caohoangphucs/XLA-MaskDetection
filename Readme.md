# Mask Detection - Hướng dẫn chạy chương trình

Dự án này sử dụng mô hình Deep Learning (MobileNetV2) để nhận diện người đeo khẩu trang hoặc không đeo khẩu trang thông qua Camera hoặc Video.

## 1. Cài đặt môi trường

### Yêu cầu
- Python 3.8 trở lên
- Pip (trình quản lý gói của Python)

### Các bước cài đặt
1. Tạo môi trường ảo (khuyến nghị):
   ```bash
   python -m venv venv
   ```
2. Kích hoạt môi trường ảo:
   - Trên Windows: `venv\Scripts\activate`
   - Trên Linux/macOS: `source venv/bin/activate`
3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Chạy chương trình

Sau khi đã cài đặt xong các thư viện, bạn có thể khởi chạy server bằng lệnh sau:

```bash
uvicorn main:app --reload
```

Server sẽ mặc định chạy tại địa chỉ: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 3. Cách sử dụng

1. Mở trình duyệt và truy cập vào: [http://127.0.0.1:8000](http://127.0.0.1:8000)
2. Chương trình sẽ yêu cầu quyền truy cập Camera. Hãy nhấn **Allow** (Cho phép).
3. **Chế độ Camera:** Chương trình sẽ tự động nhận diện từ webcam của bạn.
4. **Chế độ Video:** Chọn **Video File** ở góc trên bên trái, sau đó chọn một tệp video từ máy tính của bạn để tiến hành nhận diện.

## 4. Cấu trúc dự án
- `main.py`: File chạy chính (FastAPI Server).
- `index.html`: Giao diện người dùng Web.
- `mask_mobilenet.h5`: Mô hình đã được huấn luyện.
- `models/`: Chứa mô hình phát hiện khuôn mặt (Face Detection).
- `another_models/`: Chứa các file Notebook dùng để huấn luyện và thử nghiệm.

## 5. Tài liệu
- Link file báo cáo: [File báo cáo](https://docs.google.com/document/d/1gXTe8pG9Y8xtTVXVlet-phtJkxBSKJ6G9Z4LJbiiy1s/edit?usp=sharing)
- API doc tại: `/docs` (khi server đang chạy)