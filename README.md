# 🚦 Traffic AI — Web App

**Stack:** React (Vite) Frontend + FastAPI Backend + YOLO + PaddleOCR

---

## Cấu trúc thư mục

```
traffic-ai/
├── backend/
│   ├── main.py              ← FastAPI server
│   ├── requirements.txt
│   └── models/
│       └── best_final.pt    ← ← ← Đặt model YOLO của bạn vào đây
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        └── App.jsx
```

---

## BƯỚC 1 — Cài đặt Backend

### Yêu cầu
- Python 3.10+
- CUDA 12.x (nếu dùng GPU)
- File `best_final.pt` từ Google Drive của bạn

### Cài đặt

```bash
cd backend

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Cài numpy trước (PaddleOCR yêu cầu <2.0)
pip install "numpy==1.26.4"

# Cài PaddlePaddle GPU (CUDA 12.x)
pip install paddlepaddle-gpu==3.0.0b1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# Cài các package còn lại
pip install fastapi uvicorn[standard] python-multipart \
            opencv-python-headless pillow ultralytics paddleocr==2.8.1
```

### Đặt model YOLO

```bash
mkdir -p backend/models
# Copy best_final.pt từ Google Drive vào thư mục này:
cp /path/to/best_final.pt backend/models/best_final.pt
```

### Chạy backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

✅ API sẽ chạy tại: http://localhost:8000  
✅ Docs tương tác: http://localhost:8000/docs

---

## BƯỚC 2 — Cài đặt Frontend

### Yêu cầu
- Node.js 18+

### Cài đặt và chạy

```bash
cd frontend
npm install
npm run dev
```

✅ Web sẽ mở tại: http://localhost:3000

---

## BƯỚC 3 — Sử dụng

### Tab "Nhận biển số"
1. Upload ảnh chứa xe có biển số
2. Điều chỉnh slider Confidence YOLO (mặc định 35%)
3. Nhấn **"Phân tích biển số"**
4. Xem kết quả: biển số, confidence YOLO + OCR, hợp lệ TT79?

### Tab "Phát hiện vi phạm"
1. Upload ảnh giao thông
2. Trên canvas hiện ra, **vẽ vùng ROI** (kéo chuột) và **kẻ vạch dừng** (click)
3. Chọn trạng thái đèn: 🔴 hoặc 🟢
4. Nhấn **"Xác nhận & chạy phân tích"**
5. Xem danh sách vi phạm + ảnh bằng chứng

---

## Deploy Production

### Option A — Cùng máy chủ (Nginx reverse proxy)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend (build static)
    location / {
        root /var/www/traffic-ai/frontend/dist;
        try_files $uri /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files (uploads, evidence)
    location /uploads/ { proxy_pass http://127.0.0.1:8000/uploads/; }
    location /evidence/ { proxy_pass http://127.0.0.1:8000/evidence/; }
}
```

Build frontend:
```bash
cd frontend
VITE_API_URL=/api npm run build
```

### Option B — Docker Compose

```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./backend/models:/app/models
      - uploads:/app/uploads
      - evidence:/app/evidence
    environment:
      - PLATE_MODEL_PATH=/app/models/best_final.pt

  frontend:
    build: ./frontend
    ports: ["3000:80"]
    environment:
      - VITE_API_URL=http://backend:8000
    depends_on: [backend]

volumes:
  uploads:
  evidence:
```

---

## API Endpoints

| Method | Path | Mô tả |
|--------|------|--------|
| GET | `/` | Health check |
| GET | `/health` | Status server |
| POST | `/detect-plate` | Nhận biển số từ ảnh |
| POST | `/traffic-violation` | Phát hiện vi phạm đèn đỏ |

### POST /detect-plate
```
Form data:
  file             (ảnh)
  conf_threshold   (float, mặc định 0.35)
  ocr_confidence   (float, mặc định 0.30)

Response:
{
  "image_url": "/uploads/abc123.jpg",
  "detections": [
    {
      "plate_text": "51G-12345",
      "plate_raw":  "51G12345",
      "hop_le":     true,
      "yolo_conf":  0.87,
      "ocr_conf":   0.91,
      "two_line":   false,
      "bbox":       [120, 340, 280, 390]
    }
  ],
  "count": 1
}
```

### POST /traffic-violation
```
Form data:
  file           (ảnh)
  light_status   "RED" | "GREEN"
  roi_x1, roi_y1, roi_x2, roi_y2  (tọa độ ROI)
  line_y         (y-coordinate của vạch dừng)
  conf_vehicle   (float, mặc định 0.45)
  conf_plate     (float, mặc định 0.35)

Response:
{
  "light_status": "RED",
  "vehicles": [...],
  "violations": [
    {
      "time": "14:32:15",
      "vehicle_type": "Ô tô",
      "plate_text": "51G-12345",
      "ocr_conf": 0.88,
      "evidence_url": "/evidence/evidence_51G-12345_143215.jpg",
      "bbox": [100, 200, 300, 450]
    }
  ],
  "total_violations": 1
}
```

---

## Troubleshooting

| Lỗi | Giải pháp |
|-----|----------|
| `Model chưa sẵn sàng` | Kiểm tra path `models/best_final.pt` |
| `CUDA not found` | Dùng `paddlepaddle` (CPU) thay vì `paddlepaddle-gpu` |
| CORS error | Đổi `allow_origins=["http://localhost:3000"]` trong main.py |
| Frontend không kết nối | Kiểm tra đúng API URL trong ô cấu hình |
| `numpy` conflict | Cài lại: `pip install "numpy==1.26.4" --force-reinstall` |
