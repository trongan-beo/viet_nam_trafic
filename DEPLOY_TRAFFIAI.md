# 🌐 Hướng dẫn Public Web Traffic AI với Docker & Domain `trafficai.online`

Tài liệu này hướng dẫn chi tiết cách triển khai (deploy) ứng dụng **Traffic AI** lên Server/VPS bằng **Docker & Docker Compose**, tự động cấu hình **Nginx Reverse Proxy** và cấp chứng chỉ **SSL HTTPS (Let's Encrypt)** cho tên miền `trafficai.online`.

---

## 📋 1. Chuẩn bị DNS cho Tên Miền `trafficai.online`

Trước khi chạy script lấy SSL, bạn cần trỏ tên miền về địa chỉ IP public của máy chủ VPS của bạn tại nhà cung cấp tên miền (Cloudflare, Namecheap, MatBao, TENTEN, v.v.):

| Loại Record | Tên (Host) | Giá trị (Points to / Target) | TTL |
| :--- | :--- | :--- | :--- |
| **A** | `@` | `<IP_PUBLIC_VPS_CỦA_BẠN>` | Auto / 300 |
| **A** | `www` | `<IP_PUBLIC_VPS_CỦA_BẠN>` | Auto / 300 |

*Ví dụ: Trỏ `@` và `www` về IP `123.45.67.89`.*

---

## 🛠️ 2. Các File Cấu Hình Docker Đã Được Tự Động Thiết Lập

1. [`backend/Dockerfile`](file:///c:/Users/Admin/Downloads/traffic-ai-fullstack/traffic-ai/backend/Dockerfile): Đã cài đặt đầy đủ Python 3.10, PyTorch, OpenCV, PaddleOCR và các thư viện C++ hệ thống (`libgl1`, `libgomp1`, v.v.).
2. [`frontend/Dockerfile`](file:///c:/Users/Admin/Downloads/traffic-ai-fullstack/traffic-ai/frontend/Dockerfile): Multi-stage build ứng dụng React (Vite) và đóng gói vào Nginx Alpine tối ưu.
3. [`nginx/conf.d/traffiai.conf`](file:///c:/Users/Admin/Downloads/traffic-ai-fullstack/traffic-ai/nginx/conf.d/traffiai.conf): Cấu hình Nginx điều hướng `https://trafficai.online` sang Frontend và tất cả API sang Backend.
4. [`docker-compose.yml`](file:///c:/Users/Admin/Downloads/traffic-ai-fullstack/traffic-ai/docker-compose.yml): Quản lý 4 dịch vụ: `backend`, `frontend`, `proxy` (Nginx), và `certbot` (tự động gia hạn SSL 12h/lần).
5. [`init-ssl.sh`](file:///c:/Users/Admin/Downloads/traffic-ai-fullstack/traffic-ai/init-ssl.sh): Script 1-click xin cấp chứng chỉ SSL HTTPS từ Let's Encrypt.
6. [`deploy.sh`](file:///c:/Users/Admin/Downloads/traffic-ai-fullstack/traffic-ai/deploy.sh): Script build & khởi chạy lại toàn bộ hệ thống.

---

## 🚀 3. Hướng Dẫn Chạy Đơn Giản Trên VPS

### Bước 1: Upload code lên VPS
Ủy quyền git clone hoặc copy toàn bộ thư mục dự án lên VPS của bạn.

### Bước 2: Khởi tạo SSL & Chạy Hệ Thống
Cấp quyền thực thi và chạy script `init-ssl.sh`:

```bash
chmod +x init-ssl.sh deploy.sh
./init-ssl.sh
```

> **Script sẽ tự động:**
> - Khởi tạo Nginx ở cổng 80 để phục vụ xác minh tên miền `trafficai.online`.
> - Gửi yêu cầu xin cấp chứng chỉ SSL miễn phí từ Let's Encrypt cho `trafficai.online` & `www.trafficai.online`.
> - Tự động kích hoạt cấu hình HTTPS (port 443) với HTTP -> HTTPS auto redirect.
> - Kích hoạt container `certbot` tự động gia hạn SSL khi gần hết hạn.

---

## 🖥️ 4. Chạy Thử Trên Máy Dịch Vụ Cục Bộ (Local Docker Test)

Nếu muốn test toàn bộ Docker stack trên máy local trước khi đưa lên VPS:

```bash
docker compose up -d --build
```

Trang web sẽ truy cập được tại:
- Frontend: `http://localhost`
- Backend API: `http://localhost/health` hoặc `http://localhost/docs`

---

## ⚙️ 5. Quản Lý Container

- **Xem trạng thái hệ thống:**
  ```bash
  docker compose ps
  ```
- **Xem log kiểm tra:**
  ```bash
  docker compose logs -f
  ```
- **Dừng hệ thống:**
  ```bash
  docker compose down
  ```
- **Rebuild & Deploy lại:**
  ```bash
  ./deploy.sh
  ```
