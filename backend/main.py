"""
Traffic AI Backend — FastAPI
Chạy: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import os, re, cv2, uuid, base64, tempfile, shutil, numpy as np
import json as _json
import sqlite3, hashlib, secrets, time, asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Traffic AI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR   = Path("uploads")
EVIDENCE_DIR = Path("evidence")
for d in [UPLOAD_DIR, EVIDENCE_DIR]:
    d.mkdir(exist_ok=True)

app.mount("/uploads",  StaticFiles(directory="uploads"),  name="uploads")
app.mount("/evidence", StaticFiles(directory="evidence"), name="evidence")
_frontend_dir = Path("../frontend/dist")
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")

# ════════════════════════════════════════════════════════════════════════════
# PERFORMANCE CONFIG — sửa nhanh ở đây khi box / giám sát đa cam bị trễ, giật
# ════════════════════════════════════════════════════════════════════════════
# Số luồng detect chạy song song (mỗi luồng xử lý 1 cam tại 1 thời điểm trong stream-frame).
# Phải khớp với max_workers của _detect_executor bên dưới — đổi biến này là đủ, không cần sửa
# 2 chỗ. Trên máy KHÔNG có GPU, số này càng lớn thì mỗi luồng càng ít lõi CPU (xem get_models()).
DETECT_WORKERS = int(os.getenv("DETECT_WORKERS", "4"))
# Kích thước ảnh đưa vào YOLO khi nhận diện XE (mặc định ultralytics tự resize về 640 nếu không
# truyền imgsz). Hạ xuống 416 giúp suy luận trên CPU nhanh hơn đáng kể, đổi lại độ chính xác với
# xe rất xa/rất nhỏ giảm nhẹ. KHÔNG áp dụng cho model đọc biển số (ảnh crop biển đã nhỏ sẵn, hạ
# thêm dễ mất nét chữ).
VEHICLE_IMGSZ = int(os.getenv("VEHICLE_IMGSZ", "416"))
# Công tắc CHẨN ĐOÁN: log cho thấy các đợt process_ms tăng vọt (100-180ms, hiếm khi
# >300ms) luôn xảy ra đúng lúc CÙNG LÚC có nhiều xe/vi phạm — nghi vấn hàng đầu là 2
# luồng OCR nền (_ocr_bg_executor) tranh GIL với luồng xử lý chính khi PaddleOCR chạy
# CPU. Set OCR_BG_ENABLED=0 để tắt hẳn việc đọc biển số nền 1 buổi rồi so log: nếu các
# đợt tăng vọt biến mất hẳn -> đúng là do OCR nền, lúc đó nên cân nhắc chuyển OCR sang
# tiến trình riêng (multiprocessing) thay vì thread, để không còn tranh GIL nữa. Mặc
# định vẫn BẬT (giữ nguyên hành vi cũ) — đây chỉ là công cụ để tự đo, không tự ý tắt.
OCR_BG_ENABLED = os.getenv("OCR_BG_ENABLED", "0") == "0"
# Ngưỡng confidence để DỪNG SỚM vòng lặp biến-thể trong _ocr_one_crop(): nếu một biến
# thể cho ra biển số ĐÃ hợp lệ (đúng định dạng + mã tỉnh) với conf >= ngưỡng này, các biến
# thể còn lại sẽ được bỏ qua luôn (đỡ tốn inference/crop). Đặt =1.1 (không thể đạt)
# để tắt hẳn early-exit. LƯU Ý: hiện _ocr_one_crop() chỉ chạy 1 biến thể "bright" (đã
# tắt otsu/color_resized/gray/color_rgb) nên early-exit gần như không còn tác dụng.
OCR_EARLY_EXIT_CONF = float(os.getenv("OCR_EARLY_EXIT_CONF", "0.90"))

# ════════════════════════════════════════════════════════════════════════════
# MODULE: AUTH (đăng nhập + phân quyền) / AUDIT LOG / CAMERA CONFIG (SQLite)
# ════════════════════════════════════════════════════════════════════════════
AUTH_DB_FILE = Path("traffic_ai.db")

# 4 vai trò theo nghiệp vụ của đơn vị:
#   admin         - quản trị toàn hệ thống, quản lý người dùng
#   truc_camera   - cán bộ trực camera: cấu hình ROI/vạch dừng/đèn, xem giám sát, chạy phát hiện
#   xu_phat       - cán bộ xử phạt: xem hồ sơ vi phạm, đánh dấu đã xử phạt, tra biển số
#   lanh_dao      - lãnh đạo: chỉ xem báo cáo/thống kê, không được sửa/xoá
ROLES = ["admin", "truc_camera", "xu_phat", "lanh_dao"]
ROLE_LABELS = {"admin": "Quản trị viên", "truc_camera": "Cán bộ trực camera",
               "xu_phat": "Cán bộ xử phạt", "lanh_dao": "Lãnh đạo"}

def get_db():
    conn = sqlite3.connect(AUTH_DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_auth_db():
    conn = get_db(); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        full_name TEXT NOT NULL,
        role TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS sessions(
        token TEXT PRIMARY KEY,
        username TEXT NOT NULL,
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS audit_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        role TEXT NOT NULL,
        action TEXT NOT NULL,
        target TEXT,
        detail TEXT,
        created_at TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS camera_configs(
        cam_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        district TEXT,
        source_type TEXT,
        roi_json TEXT,
        line_pts_json TEXT,
        turn_zone_json TEXT,
        turn_confirm_json TEXT,
        light TEXT DEFAULT 'RED',
        created_by TEXT,
        updated_at TEXT NOT NULL
    )""")
    # Migration cho DB cũ đã tồn tại từ trước khi có 2 cột vùng-rẽ/cổng-xác-nhận-rẽ — ALTER TABLE
    # bỏ qua lỗi "duplicate column" nếu cột đã có sẵn (trường hợp DB mới, vừa tạo từ CREATE TABLE ở trên).
    for _col in ("turn_zone_json", "turn_confirm_json"):
        try: cur.execute(f"ALTER TABLE camera_configs ADD COLUMN {_col} TEXT")
        except sqlite3.OperationalError: pass
    conn.commit()
    # Seed tài khoản admin mặc định nếu DB user trống — BẮT BUỘC đổi mật khẩu sau khi cài đặt thật
    cur.execute("SELECT COUNT(*) c FROM users")
    if cur.fetchone()["c"] == 0:
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", "admin123".encode(), salt.encode(), 100_000).hex()
        cur.execute("INSERT INTO users(username,password_hash,salt,full_name,role,active,created_at) VALUES(?,?,?,?,?,1,?)",
                    ("admin", pwd_hash, salt, "Quản trị viên hệ thống", "admin", datetime.now().isoformat()))
        conn.commit()
        print("⚠️  Đã tạo tài khoản mặc định: admin / admin123 — ĐỔI MẬT KHẨU NGAY sau khi triển khai thật!")
    conn.close()

init_auth_db()

def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()

def create_session(username: str) -> str:
    token = secrets.token_hex(32)
    now = time.time()
    conn = get_db()
    conn.execute("INSERT INTO sessions(token,username,created_at,expires_at) VALUES(?,?,?,?)",
                 (token, username, now, now + 8 * 3600))  # phiên hết hạn sau 8 giờ
    conn.commit(); conn.close()
    return token

def get_current_user(authorization: Optional[str] = Header(None)):
    """Dependency: xác thực Bearer token, trả về dict user. 401 nếu thiếu/sai/hết hạn."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Thiếu token đăng nhập")
    token = authorization.split(" ", 1)[1]
    conn = get_db(); cur = conn.cursor()
    row = cur.execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
    if not row or row["expires_at"] < time.time():
        conn.close()
        raise HTTPException(status_code=401, detail="Phiên đăng nhập hết hạn, vui lòng đăng nhập lại")
    user = cur.execute("SELECT * FROM users WHERE username=? AND active=1", (row["username"],)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Tài khoản không tồn tại hoặc đã bị khoá")
    return {"username": user["username"], "full_name": user["full_name"], "role": user["role"]}

def require_role(*allowed_roles):
    """Dependency factory: chỉ cho phép các vai trò chỉ định. require_role('admin','truc_camera')"""
    def checker(user: dict = Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail=f"Vai trò '{ROLE_LABELS.get(user['role'],user['role'])}' không có quyền thực hiện thao tác này")
        return user
    return checker

def write_audit(user: dict, action: str, target: str = "", detail: str = ""):
    """Ghi nhật ký thao tác — dùng để truy vết trách nhiệm (ai làm gì, lúc nào)."""
    conn = get_db()
    conn.execute("INSERT INTO audit_log(username,role,action,target,detail,created_at) VALUES(?,?,?,?,?,?)",
                 (user["username"], user["role"], action, target, detail, datetime.now().isoformat()))
    conn.commit(); conn.close()

class LoginBody(BaseModel):
    username: str
    password: str

class CreateUserBody(BaseModel):
    username: str
    password: str
    full_name: str
    role: str

@app.post("/auth/login")
async def auth_login(body: LoginBody):
    conn = get_db(); cur = conn.cursor()
    user = cur.execute("SELECT * FROM users WHERE username=?", (body.username,)).fetchone()
    if not user or not user["active"]:
        conn.close()
        raise HTTPException(status_code=401, detail="Sai tài khoản hoặc mật khẩu")
    if hash_password(body.password, user["salt"]) != user["password_hash"]:
        conn.close()
        raise HTTPException(status_code=401, detail="Sai tài khoản hoặc mật khẩu")
    token = create_session(user["username"])
    conn.close()
    write_audit({"username": user["username"], "role": user["role"]}, "DANG_NHAP", detail="Đăng nhập thành công")
    return {"token": token, "username": user["username"], "full_name": user["full_name"],
            "role": user["role"], "role_label": ROLE_LABELS.get(user["role"], user["role"])}

@app.post("/auth/logout")
async def auth_logout(authorization: Optional[str] = Header(None), user: dict = Depends(get_current_user)):
    token = authorization.split(" ", 1)[1]
    conn = get_db(); conn.execute("DELETE FROM sessions WHERE token=?", (token,)); conn.commit(); conn.close()
    write_audit(user, "DANG_XUAT")
    return {"ok": True}

@app.get("/auth/me")
async def auth_me(user: dict = Depends(get_current_user)):
    return {**user, "role_label": ROLE_LABELS.get(user["role"], user["role"])}

@app.get("/auth/users")
async def list_users(user: dict = Depends(require_role("admin"))):
    conn = get_db()
    rows = conn.execute("SELECT id,username,full_name,role,active,created_at FROM users ORDER BY id").fetchall()
    conn.close()
    return {"users": [dict(r) for r in rows]}

@app.post("/auth/users")
async def create_user(body: CreateUserBody, user: dict = Depends(require_role("admin"))):
    if body.role not in ROLES:
        raise HTTPException(status_code=400, detail=f"Vai trò không hợp lệ. Chọn 1 trong: {ROLES}")
    salt = secrets.token_hex(16)
    pwd_hash = hash_password(body.password, salt)
    conn = get_db()
    try:
        conn.execute("INSERT INTO users(username,password_hash,salt,full_name,role,active,created_at) VALUES(?,?,?,?,?,1,?)",
                     (body.username, pwd_hash, salt, body.full_name, body.role, datetime.now().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại")
    conn.close()
    write_audit(user, "TAO_TAI_KHOAN", target=body.username, detail=f"Vai trò: {body.role}")
    return {"ok": True}

@app.delete("/auth/users/{username}")
async def deactivate_user(username: str, user: dict = Depends(require_role("admin"))):
    if username == user["username"]:
        raise HTTPException(status_code=400, detail="Không thể tự khoá chính mình")
    conn = get_db()
    conn.execute("UPDATE users SET active=0 WHERE username=?", (username,))
    conn.commit(); conn.close()
    write_audit(user, "KHOA_TAI_KHOAN", target=username)
    return {"ok": True}

@app.get("/audit-log")
async def get_audit_log(limit: int = 200, username: Optional[str] = None, user: dict = Depends(require_role("admin", "lanh_dao"))):
    conn = get_db()
    if username:
        rows = conn.execute("SELECT * FROM audit_log WHERE username=? ORDER BY id DESC LIMIT ?", (username, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"total": len(rows), "logs": [dict(r) for r in rows]}

# ── Cấu hình camera (lưu bền vững thay vì biến tạm trong trình duyệt) ──
class CameraConfigBody(BaseModel):
    cam_id: str
    name: str
    district: Optional[str] = ""
    source_type: Optional[str] = "webcam"
    roi: Optional[list] = None      # [[x,y],[x,y],[x,y],[x,y]] polygon 4 điểm (giống tab Video)
    line_pts: Optional[list] = None # [[x1,y1],[x2,y2]] đường nghiêng 2 điểm (giống tab Video)
    # Nút giao cho phép rẽ khi đèn đỏ — 2 vùng TÙY CHỌN (bỏ trống = giữ hành vi cũ, phạt ngay
    # khi cắt vạch lúc đèn đỏ, không đổi gì). Xem chi tiết cơ chế ở resolve_turn_pending().
    turn_zone: Optional[list] = None    # polygon ≥3 điểm — "vùng rẽ được phép", sát vạch, phía làn rẽ
    turn_confirm: Optional[list] = None # polygon ≥3 điểm — "cổng xác nhận rẽ", đặt sang đường cắt ngang
    light: Optional[str] = "RED"

@app.get("/cameras")
async def list_cameras(user: dict = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute("SELECT * FROM camera_configs ORDER BY cam_id").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["roi"] = _json.loads(d["roi_json"]) if d.get("roi_json") else None
        d["line_pts"] = _json.loads(d["line_pts_json"]) if d.get("line_pts_json") else None
        d["turn_zone"] = _json.loads(d["turn_zone_json"]) if d.get("turn_zone_json") else None
        d["turn_confirm"] = _json.loads(d["turn_confirm_json"]) if d.get("turn_confirm_json") else None
        del d["roi_json"]; del d["line_pts_json"]; del d["turn_zone_json"]; del d["turn_confirm_json"]
        out.append(d)
    return {"cameras": out}

@app.put("/cameras/{cam_id}")
async def upsert_camera(cam_id: str, body: CameraConfigBody, user: dict = Depends(require_role("admin", "truc_camera"))):
    conn = get_db()
    conn.execute("""INSERT INTO camera_configs(cam_id,name,district,source_type,roi_json,line_pts_json,
                     turn_zone_json,turn_confirm_json,light,created_by,updated_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(cam_id) DO UPDATE SET name=excluded.name,district=excluded.district,
                     source_type=excluded.source_type,roi_json=excluded.roi_json,line_pts_json=excluded.line_pts_json,
                     turn_zone_json=excluded.turn_zone_json,turn_confirm_json=excluded.turn_confirm_json,
                     light=excluded.light,updated_at=excluded.updated_at""",
                 (cam_id, body.name, body.district, body.source_type,
                  _json.dumps(body.roi) if body.roi else None,
                  _json.dumps(body.line_pts) if body.line_pts else None,
                  _json.dumps(body.turn_zone) if body.turn_zone else None,
                  _json.dumps(body.turn_confirm) if body.turn_confirm else None,
                  body.light, user["username"], datetime.now().isoformat()))
    conn.commit(); conn.close()
    write_audit(user, "CAU_HINH_CAMERA", target=cam_id, detail=f"{body.name} ({body.district})")
    return {"ok": True}

@app.delete("/cameras/{cam_id}")
async def delete_camera(cam_id: str, user: dict = Depends(require_role("admin", "truc_camera"))):
    conn = get_db()
    conn.execute("DELETE FROM camera_configs WHERE cam_id=?", (cam_id,))
    conn.commit(); conn.close()
    write_audit(user, "XOA_CAMERA", target=cam_id)
    return {"ok": True}

# ════════════════════════════════════════════════════════════════════════════
_yolo_plate = _yolo_vehicle = _ocr_det = _ocr_nodet = None
# Thống kê biến thể ảnh (color_resized/color_rgb/gray/bright/otsu) nào cho kết quả OCR
# được majority_vote() chọn nhiều nhất — dùng để theo dõi hiệu quả early-exit, xem endpoint
# /debug/ocr-variant-stats. Không phải dữ liệu nghiệp vụ, mất khi restart server là bình thường.
_variant_win_stats = {}
_variant_total_runs = 0   # tổng số lần _ocr_one_crop() được gọi (mẫu số để tính % dừng sớm)
_variant_early_exits = 0  # số lần dừng sớm (không chạy hết 5 biến thể)

def get_models():
    global _yolo_plate, _yolo_vehicle, _ocr_det, _ocr_nodet
    if _yolo_plate is None:
        try:
            import torch
            _orig_torch_load = torch.load
            def _compat_torch_load(*args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)
            torch.load = _compat_torch_load
            import ultralytics.nn.tasks
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
        except Exception:
            pass

        from ultralytics import YOLO
        PLATE_PATH   = os.getenv("PLATE_MODEL_PATH",   "models/best_final.pt")
        VEHICLE_PATH = os.getenv("VEHICLE_MODEL_PATH", "models/yolo11n.pt")
        if not Path(PLATE_PATH).exists():
            raise FileNotFoundError(f"Khong tim thay model bien so: {PLATE_PATH}")
        _yolo_plate   = YOLO(PLATE_PATH)
        _yolo_vehicle = YOLO(VEHICLE_PATH) if Path(VEHICLE_PATH).exists() else _yolo_plate
        # In rõ đang chạy CPU hay GPU — đây thường là nguyên nhân số 1 khiến việc nhận diện
        # chậm (1-2s+/khung trên CPU là bình thường, trên GPU thường chỉ 20-80ms/khung).
        try:
            import torch
            gpu_ok = torch.cuda.is_available()
            dev_name = torch.cuda.get_device_name(0) if gpu_ok else "CPU (không có GPU CUDA khả dụng)"
            print(f"[MODEL] YOLO chạy trên: {dev_name}  (model xe: {VEHICLE_PATH})")
            if not gpu_ok:
                print("[MODEL] ⚠️  Đang chạy CPU -> mỗi lần nhận diện có thể mất 0.5-2s+, "
                      "là nguyên nhân chính khiến box ở giám sát đa cam bị trễ so với xe. "
                      "Cân nhắc dùng máy có GPU NVIDIA (CUDA), hoặc đổi VEHICLE_MODEL_PATH "
                      "sang bản 'n' (yolo11n.pt) nhẹ hơn để tăng tốc trên CPU.")
                # FIX oversubscription: mặc định PyTorch để MỖI lần predict() tự do dùng TOÀN
                # BỘ lõi CPU. Khi DETECT_WORKERS luồng (nhiều cam) cùng detect song song, tất cả
                # cùng giành hết lõi -> tranh chấp + cache thrashing -> mỗi khung hình xử lý CHẬM
                # HƠN so với chạy tuần tự, càng nhiều cam mở cùng lúc càng rõ. Giới hạn thread mỗi
                # luồng để tổng số thread đang chạy ~ bằng số lõi máy thật sự có.
                _threads = max(1, (os.cpu_count() or DETECT_WORKERS) // DETECT_WORKERS)
                torch.set_num_threads(_threads)
                print(f"[MODEL] CPU: giới hạn {_threads} thread/luồng detect "
                      f"({DETECT_WORKERS} luồng × {_threads} thread ≈ {os.cpu_count()} lõi máy) "
                      f"để {DETECT_WORKERS} cam chạy cùng lúc không giành lõi CPU của nhau. "
                      "Đổi qua env DETECT_WORKERS nếu số cam thực tế khác 4.")
        except Exception as _e:
            print(f"[MODEL] Không kiểm tra được GPU: {_e}")
    if _ocr_det is None:
        import paddle
        from paddleocr import PaddleOCR
        # Mặc định ép CPU cho OCR — paddlepaddle-gpu mang theo bộ DLL CUDA/cuDNN RIÊNG, dễ xung
        # đột phiên bản với DLL mà PyTorch (torch) đã nạp trước đó trên Windows, gây lỗi kiểu
        # "WinError 127: The specified procedure could not be found" khi load cudnn_cnn64_9.dll.
        # OCR đọc biển số đã chạy NGẦM (không chặn box hiển thị) từ trước, nên chạy CPU ở đây
        # không ảnh hưởng gì đến độ mượt của giám sát đa cam — an toàn, tránh hẳn xung đột DLL.
        # Muốn thử lại GPU cho OCR (sau khi tự xử lý được xung đột DLL): set OCR_USE_GPU=1
        use_gpu = os.getenv("OCR_USE_GPU", "0") == "1" and paddle.device.get_device().startswith("gpu")
        _ocr_det   = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=use_gpu, show_log=False, det=True,  cls=False)
        _ocr_nodet = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=use_gpu, show_log=False, det=False, cls=False)
    return _yolo_plate, _yolo_vehicle, _ocr_det, _ocr_nodet

def _warmup_models():
    """Chạy vài lần predict() với ảnh giả NGAY LÚC SERVER KHỞI ĐỘNG, để CUDA/cuDNN
    JIT-compile + benchmark thuật toán từ bây giờ.

    Đây chính là nguyên nhân của hiện tượng thấy trong log thật: 6 khung hình đầu
    mất 5.4-6.5s (trung bình ~6.1s), sau đó vài chục khung tiếp giảm dần
    (1.3s -> 0.4s), rồi mới rơi hẳn về mức GPU bình thường (~40-100ms) và ở nguyên
    đó — cộng dồn khoảng 50 GIÂY bị "khởi động nguội" đúng vào lúc người dùng vừa
    bật camera, gây giật/trễ box + có thể bỏ lỡ vi phạm ngay phút đầu giám sát.

    Warm-up ở đây trả giá 5-6s đó MỘT LẦN lúc server mới bật (trước khi ai xem
    camera), thay vì để camera đầu tiên người dùng bật phải gánh chịu. Lỗi ở đây
    KHÔNG làm sập server — chỉ là tối ưu, không phải chức năng chính.
    """
    import time as _time
    t0 = _time.time()
    try:
        yolo_plate, yolo_vehicle, ocr_det, _ = get_models()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)   # ~ tỉ lệ khung hình camera thật
        dummy_plate = np.zeros((100, 300, 3), dtype=np.uint8)   # ~ tỉ lệ ảnh crop biển số
        for _ in range(3):  # vài lần để cuDNN benchmark ổn định, không chỉ 1 lần
            yolo_vehicle.predict(dummy_frame, classes=[0, 2, 5, 7], conf=0.5,#classes=[0, 2, 3, 5, 7]
                                  imgsz=VEHICLE_IMGSZ, verbose=False)
            yolo_plate.predict(dummy_plate, conf=0.5, verbose=False)
        try:
            ocr_det.ocr(dummy_plate, det=True, cls=False)  # OCR cũng warm-up cho chắc, lỗi thì bỏ qua
        except Exception:
            pass
        print(f"[MODEL] ✅ Warm-up xong sau {_time.time()-t0:.1f}s lúc khởi động — "
              "camera đầu tiên người dùng bật sẽ không còn bị chậm 5-6s/khung nữa.")
    except Exception as _e:
        print(f"[MODEL] ⚠️  Warm-up lỗi (bỏ qua, không ảnh hưởng chức năng chính): {_e}")

@app.on_event("startup")
def _on_startup():
    _warmup_models()

_MA_TINH = {"11","12","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","40","41","42","43","47","48","49","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","92","93","94","95","96","97","98","99"}
_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. ")
_D_FIX   = {"O":"0","I":"1","B":"8","S":"5","Z":"2","Q":"0","D":"0"}
# Biển số xe VN chỉ dùng 20 chữ cái: A B C D E F G H K L M N P S T U V X Y Z
# (KHÔNG có I, J, O, Q, R, W). _L_FIX trước đây map "1"->"I" và "0"->"O", tức là
# tự sinh ra 2 ký tự mà biển số thật không bao giờ có -> đây là nguyên nhân chính
# khiến kết quả OCR ra "15I2-..." dù biển số VN không dùng chữ I. Đổi sang các chữ
# hợp lệ có hình dáng gần giống để không còn tạo ra ký tự cấm.
_L_FIX   = {"0":"D","1":"L","8":"B","5":"F","2":"Z","6":"G","4":"A"}
_VALID_PLATE_LETTERS = set("ABCDEFGHKLMNPSTUVXYZ")

def _is_valid_series_letter(c: str) -> bool:
    """Chữ cái series trên biển số VN chỉ được là 1 trong 20 chữ hợp lệ (không I/J/O/Q/R/W)."""
    return len(c) == 1 and c in _VALID_PLATE_LETTERS
VEHICLE_CLASSES = {0:"Xe May",2:"O to", 3:"Xe may", 5:"Xe buyt", 7:"Xe tai"}

def enhance_dark_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img.copy()
    h,w = gray.shape; scale = max(80/h, 1.0)
    enlarged = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    bg = cv2.morphologyEx(enlarged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(25,25)))
    normalized = cv2.divide(enlarged, bg, scale=255)
    tile = max(normalized.shape[0]//4, 8)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile,tile)).apply(normalized)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=7, templateWindowSize=7, searchWindowSize=21)
    return cv2.filter2D(denoised, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))

def deskew_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 50: return img
    rect = cv2.minAreaRect(coords); angle = rect[-1]
    if angle < -45: angle += 90
    if abs(angle) > 10: return img
    h,w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w/2,h/2),angle,1.0), (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Ngưỡng diện tích tối thiểu (px) để chấp nhận 1 box biển số. Trước đây cố định 600/800,
# nhưng log thực tế cho thấy nhiều biển được YOLO detect với conf 0.92-0.94 (rất tự tin,
# gần như chắc chắn đúng) vẫn bị loại oan vì xe ở xa camera -> biển chiếm ít pixel.
# Hạ xuống 150/200 để nhận các biển nhỏ/xa; chỉnh qua env nếu cần khắt khe hơn (giảm false positive)
# hoặc lỏng hơn nữa (bắt được biển xa hơn).
PLATE_MIN_AREA_SQUARE = int(os.getenv("PLATE_MIN_AREA_SQUARE", "150"))  # box gần vuông (biển 2 dòng)
PLATE_MIN_AREA_WIDE   = int(os.getenv("PLATE_MIN_AREA_WIDE", "200"))    # box dẹt ngang (biển 1 dòng)

def is_valid_plate_box(bw, bh, area_min=None):
    if bw<=0 or bh<=0: return False
    aspect=bw/bh; area=bw*bh
    area_min = area_min if area_min is not None else PLATE_MIN_AREA_SQUARE
    return (2.0<=aspect<=5.5 and area>=PLATE_MIN_AREA_WIDE) or (0.8<=aspect<2.0 and area>=area_min)

def save_overview_evidence(full_frame, bbox, plate_text: str = "", prefix: str = "overview") -> str:
    """Lưu ảnh TOÀN CẢNH (nguyên khung hình gốc, KHÔNG crop) tại đúng thời điểm xe vượt vạch.
    Khác với ảnh cận cảnh (evidence_*.jpg chỉ có riêng chiếc xe), ảnh này giữ nguyên bối cảnh
    giao lộ và CHỈ khoanh khung đỏ quanh DUY NHẤT xe vi phạm — các xe/phương tiện khác đang có
    mặt trong khung hình không bị vẽ box, để không gây hiểu nhầm là các xe đó cũng vi phạm."""
    x1, y1, x2, y2 = bbox
    h_f, w_f = full_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_f, x2), min(h_f, y2)
    overview = full_frame.copy()
    color = (0, 0, 255)  # đỏ (BGR)
    cv2.rectangle(overview, (x1, y1), (x2, y2), color, 3)
    label = plate_text if plate_text and plate_text not in ("KHO_DOC_BIEN", "") else "VI PHAM"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ly = y1 - 10 if y1 - 10 - th > 0 else y1 + th + 14
    cv2.rectangle(overview, (x1, ly - th - 8), (x1 + tw + 10, ly + 4), color, -1)
    cv2.putText(overview, label, (x1 + 5, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    ts_label = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    cv2.putText(overview, ts_label, (10, h_f - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    uidv = uuid.uuid4().hex[:6]; tsv = datetime.now().strftime("%H%M%S")
    fname = f"{prefix}_{tsv}_{uidv}.jpg"
    cv2.imwrite(str(EVIDENCE_DIR / fname), overview)
    return f"/evidence/{fname}"

def is_two_line_plate(bw, bh): return (bw/max(bh,1))<1.4 and bh>bw*0.6
def hard_filter(text): return "".join(c for c in text.upper() if c in _ALLOWED)
def to_digit(s):  return "".join(_D_FIX.get(c,c) for c in s)
def to_letter(s): return "".join(_L_FIX.get(c,c) for c in s)
def _fix_trailing_zero(s): return s[:-1]+"8" if len(s)>=4 and s[-1]=="0" and s[-2]!="0" else s

def clean_plate_text(raw_text, two_line=False):
    if not raw_text: return "", False
    text = hard_filter(raw_text)
    SAFE = {"l":"1","i":"1","o":"0"}
    text = "".join(SAFE.get(c.lower(),c) for c in text.strip()).upper()
    text = re.sub(r"[\s\.]+","-",text); text = re.sub(r"[^A-Z0-9\-]","",text)
    text = re.sub(r"-{2,}","-",text).strip("-"); text_nd = text.replace("-","")
    DS = r"[0-9OISBGZQDT]"
    m = re.fullmatch(r"(\d{2})-?(\d{3})-?(NG|QT|CV|NN)-?(\d{2})", text)
    if m: return f"{m.group(1)}-{m.group(2)}-{m.group(3)}-{m.group(4)}", (m.group(1) in _MA_TINH)
    m = re.fullmatch(r"(\d{2})([A-Z])[-\s]?(\d{3})[-\s]?(\d{2})", text_nd)
    if m:
        ma=to_digit(m.group(1)); so=_fix_trailing_zero(to_digit(m.group(3)+m.group(4)))
        if ma in _MA_TINH: return f"{ma}{m.group(2)}-{so}", True
    if two_line:
        m = re.match(rf"^({DS}{{2}})([A-Z0-9]{{1,2}})({DS}{{4,5}})$", text_nd)
        if m:
            ma=to_digit(m.group(1)); chu=m.group(2); so=_fix_trailing_zero(to_digit(m.group(3)))
            if len(chu)==1:
                chu=to_letter(chu)
                if _is_valid_series_letter(chu): return f"{ma}{chu}-{so}", (ma in _MA_TINH)
            else:
                seri=to_letter(chu[0])+(to_digit(chu[1]) if chu[1] in "0123456789OISBGZQDT" else to_letter(chu[1]))
                if _is_valid_series_letter(seri[0]): return f"{ma}{seri}-{so}", (ma in _MA_TINH)
    m = re.search(rf"^({DS}{{2}})([A-Z0-9]{{1,2}})-({DS}{{4,5}})$", text)
    if m:
        ma,seri,so=to_digit(m.group(1)),m.group(2),_fix_trailing_zero(to_digit(m.group(3)))
        if len(seri)==1:
            chu=to_letter(seri)
            if _is_valid_series_letter(chu): return f"{ma}{chu}-{so}", (ma in _MA_TINH)
        else:
            sc=to_letter(seri[0])+(to_digit(seri[1]) if seri[1] in "0123456789OISBGZQDT" else to_letter(seri[1]))
            if _is_valid_series_letter(sc[0]): return f"{ma}{sc}-{so}", (ma in _MA_TINH)
    m = re.search(rf"({DS}{{2}})([A-Z5])({DS}{{4,5}})", text_nd)
    if m:
        ma,chu,so=to_digit(m.group(1)),to_letter(m.group(2)),_fix_trailing_zero(to_digit(m.group(3)))
        if _is_valid_series_letter(chu): return f"{ma}{chu}-{so}", (ma in _MA_TINH)
    m = re.search(rf"({DS}{{2}})([A-Z0-9]{{2}})({DS}{{4,5}})", text_nd)
    if m:
        ma,seri,so=to_digit(m.group(1)),m.group(2),_fix_trailing_zero(to_digit(m.group(3)))
        sc=to_letter(seri[0])+(to_digit(seri[1]) if seri[1] in "0123456789OISBGZQDT" else to_letter(seri[1]))
        if _is_valid_series_letter(sc[0]): return f"{ma}{sc}-{so}", (ma in _MA_TINH)
    m = re.search(rf"(\d{{2}})([A-Z5])(\d{{4,5}})", text_nd)
    if m:
        ma,chu,so=to_digit(m.group(1)),to_letter(m.group(2)),_fix_trailing_zero(to_digit(m.group(3)))
        if _is_valid_series_letter(chu): return f"{ma}{chu}-{so}", (ma in _MA_TINH)
    return text, False

def _parse_det(result, thr):
    if not result or result[0] is None: return "",0.0,[]
    texts,confs,seen=[],[],[]
    for line in result[0]:
        if len(line)==2 and isinstance(line[1],(tuple,list)):
            t,s=line[1]
            seen.append((t,s))  # ghi lại MỌI text model thấy, kể cả dưới ngưỡng thr (phục vụ debug)
            if s>=thr: texts.append(t); confs.append(s)
    if not texts: return "",0.0,seen
    return "".join(texts).strip(), sum(confs)/len(confs), seen

def _parse_nodet(result, thr):
    if not result or result[0] is None: return "",0.0,[]
    items=result[0]
    if items and isinstance(items[0],(list,tuple)) and isinstance(items[0][0],(list,tuple)): items=items[0]
    try:
        seen=[(t,c) for t,c in items if isinstance(c,float)]
        merged=" ".join(t for t,c in items if isinstance(c,float) and c>=thr)
        best=max((c for _,c in items if isinstance(c,float)),default=0.0)
    except: return "",0.0,[]
    return merged.strip(), best, seen

def run_ocr(img, thr=0.30):
    _,_,ocr_det,ocr_nodet=get_models()
    result=ocr_det.ocr(img,det=True,cls=False)
    text,conf,seen_det=_parse_det(result,thr)
    seen_all=[("det",t,s) for t,s in seen_det]
    if not text or conf<0.30:
        r2=ocr_nodet.ocr(img,det=False,cls=False)
        t2,c2,seen_nodet=_parse_nodet(r2,thr)
        seen_all += [("nodet",t,s) for t,s in seen_nodet]
        if t2 and c2>conf: return t2,c2,seen_all
    return text,conf,seen_all

def _record_variant_win(candidate):
    """candidate = (raw, conf, iv, pt) hoặc (raw, conf, iv, pt, variant_name).
    Ghi nhận biến thể thắng + in thẳng ra terminal (không cần gọi API debug)."""
    if len(candidate) > 4:
        name = candidate[4]
        _variant_win_stats[name] = _variant_win_stats.get(name, 0) + 1
        print(f"[OCR-VARIANT] 🏆 Thắng: '{name}' | biển='{candidate[3]}' conf={candidate[1]:.2f}")
        # Cứ mỗi 20 lần đọc biển thì in tổng kết tỉ lệ thắng theo từng biến thể
        total = sum(_variant_win_stats.values())
        if total % 20 == 0:
            summary = ", ".join(
                f"{k}:{v}({v/total*100:.0f}%)"
                for k, v in sorted(_variant_win_stats.items(), key=lambda x: -x[1])
            )
            early_pct = _variant_early_exits/max(_variant_total_runs,1)*100
            print(f"[OCR-VARIANT] ── Tổng kết sau {total} lần ── {summary} "
                  f"| early-exit: {_variant_early_exits}/{_variant_total_runs} ({early_pct:.1f}%)")

def majority_vote(candidates):
    valid=[c for c in candidates if c[0]]
    if not valid: return "",0.0,False
    perfect=[c for c in valid if c[2] and c[3] and len(c[3].replace("-",""))>=6]
    if perfect:
        best=max(perfect,key=lambda x:(len(x[3].replace("-","")),x[1])); _record_variant_win(best)
        return best[3],best[1],True
    tt79=[c for c in valid if c[2] and c[3]]
    if tt79:
        best=max(tt79,key=lambda x:(len(x[3].replace("-","")),x[1])); _record_variant_win(best)
        return best[3],best[1],True
    wp=[c for c in valid if c[3]]
    if wp:
        best=max(wp,key=lambda x:(len(x[3].replace("-","")),x[1])); _record_variant_win(best)
        return best[3],best[1],False
    return "",0.0,False

def _ocr_one_crop(crop_bgr, ocr_conf, two_line=False):
    global _variant_total_runs, _variant_early_exits
    h_c,w_c=crop_bgr.shape[:2]; new_h=80; new_w=max(int(w_c*new_h/h_c),120)
    resized=cv2.resize(crop_bgr,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
    aligned=deskew_plate(resized)
    gray=cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY)
    bright=cv2.convertScaleAbs(gray,alpha=1.2,beta=10)
    # CHỈ dùng 1 biến thể "bright" (thay vì 5 biến thể như trước) — đổi lại ở đây nếu
    # cần bật lại otsu/color_resized/gray/color_rgb để so sánh độ chính xác.
    variants=[("bright",bright)]
    candidates=[]
    _variant_total_runs += 1
    crop_id = _variant_total_runs  # số thứ tự để nhóm các dòng log của cùng 1 lần đọc biển
    print(f"[OCR-VARIANT #{crop_id}] ── bắt đầu chạy {len(variants)} biến thể ──")
    for name, inp in variants:
        raw,conf,seen=run_ocr(inp,thr=ocr_conf)
        if raw:
            filtered=hard_filter(raw)
            pt,iv=clean_plate_text(filtered,two_line=two_line)
            candidates.append((raw,conf,iv,pt,name))
            tag = "hợp_lệ" if iv else "raw"
            print(f"[OCR-VARIANT #{crop_id}]   {name:<14} -> '{pt or raw}' "
                  f"(conf={conf:.2f}, {tag})")
            # DỪNG SỚM: đã có biển hợp lệ (đúng định dạng + mã tỉnh) với conf đủ cao ->
            # không cần chạy tiếp các biến thể còn lại cho crop này.
            if iv and pt and len(pt.replace("-",""))>=6 and conf>=OCR_EARLY_EXIT_CONF:
                _variant_early_exits += 1
                print(f"[OCR-VARIANT #{crop_id}]   ⚡ Dừng sớm ở '{name}' (conf={conf:.2f} >= "
                      f"{OCR_EARLY_EXIT_CONF}) -> bỏ qua {len(variants)-len(candidates)} biến thể còn lại")
                break
        else:
            # In rõ LÝ DO thất bại: model hoàn toàn không thấy chữ, hay có thấy nhưng
            # confidence quá thấp bị loại (dưới ngưỡng ocr_conf).
            if seen:
                seen_str = ", ".join(f"[{m}]'{t}'(conf={s:.2f})" for m, t, s in seen[:5])
                print(f"[OCR-VARIANT #{crop_id}]   {name:<14} -> (không đọc được chữ) "
                      f"| model CÓ thấy nhưng dưới ngưỡng {ocr_conf}: {seen_str}")
            else:
                print(f"[OCR-VARIANT #{crop_id}]   {name:<14} -> (không đọc được chữ) "
                      f"| model KHÔNG phát hiện được vùng chữ nào trong ảnh")
    return candidates, bright

# ── API ──────────────────────────────────────────────────────────────────

@app.get("/")
def root(): return {"status":"ok","message":"Traffic AI API dang chay"}

@app.get("/health")
def health(): return {"status":"healthy","timestamp":datetime.now().isoformat()}

@app.post("/detect-vehicle")
async def detect_vehicle(file: UploadFile=File(...), conf_threshold: float=Form(0.15)):
    try: _, yolo_vehicle,_,_ = get_models()
    except Exception as e: raise HTTPException(503, detail=f"Model chua san sang: {e}")
    content=await file.read()
    img=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Khong doc duoc anh")
    results=yolo_vehicle.predict(img,classes=[0,2,5,7],conf=conf_threshold,imgsz=VEHICLE_IMGSZ,agnostic_nms=True,iou=0.5,verbose=False)
    vehicles=[]
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())
            vehicles.append({"bbox":[x1,y1,x2,y2],"type":VEHICLE_CLASSES.get(int(box.cls[0].item()),"Khac"),"confidence":round(float(box.conf[0].item()),3)})
    return {"vehicles":vehicles,"count":len(vehicles)}

@app.post("/detect-plate")
@app.post("/detect-plate-v2")
async def detect_plate_v2(file: UploadFile=File(...), conf_threshold: float=Form(0.35), ocr_confidence: float=Form(0.30)):
    try: yolo_plate,_,_,_ = get_models()
    except Exception as e: raise HTTPException(503, detail=f"Model chua san sang: {e}")
    content=await file.read()
    uid=uuid.uuid4().hex[:8]; suffix=Path(file.filename).suffix or ".jpg"
    save_path=UPLOAD_DIR/f"{uid}{suffix}"
    with open(save_path,"wb") as f: f.write(content)
    img=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400,"Khong doc duoc anh")
    img_enh=enhance_dark_image(img); results=yolo_plate.predict(img_enh,conf=conf_threshold,verbose=False)
    plates=[]
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy()); yc=float(box.conf[0].item()); bw,bh=x2-x1,y2-y1
            if not is_valid_plate_box(bw,bh): continue
            h_img,w_img=img.shape[:2]; pad=4
            crop=img[max(0,y1-pad):min(h_img,y2+pad),max(0,x1-pad):min(w_img,x2+pad)]
            if crop.size==0: continue
            two_line=is_two_line_plate(bw,bh)
            if two_line:
                mid=crop.shape[0]//2; c1,_=_ocr_one_crop(crop[:mid,:],ocr_confidence); c2,_=_ocr_one_crop(crop[mid:,:],ocr_confidence)
                b1=max(c1,key=lambda x:x[1])[0] if c1 else ""; b2=max(c2,key=lambda x:x[1])[0] if c2 else ""
                merged=(b1+b2).strip(); pt,iv=clean_plate_text(merged,two_line=True)
                bc=max((max(c1,key=lambda x:x[1])[1] if c1 else 0),(max(c2,key=lambda x:x[1])[1] if c2 else 0))
                candidates=[(merged,bc,iv,pt)]
            else: candidates,_=_ocr_one_crop(crop,ocr_confidence)
            plate_text,best_conf,hop_le=majority_vote(candidates)
            if len(plate_text.replace("-",""))<4: continue
            plates.append({"plate_text":plate_text,"confidence":round(yc,3),"ocr_conf":round(best_conf,3),"hop_le":hop_le,"bbox":[x1,y1,x2,y2]})
    return {"plates":plates,"count":len(plates),"image_url":f"/uploads/{save_path.name}"}

@app.post("/check-violation")
async def check_violation(
    file: UploadFile=File(...), light_status: str=Form("RED"),
    roi_points: str=Form("[]"), line_points: str=Form("[]"),
    conf_vehicle: float=Form(0.15), conf_plate: float=Form(0.35), ocr_confidence: float=Form(0.30),
    user: dict = Depends(require_role("admin", "truc_camera"))
):
    try: _,yolo_vehicle,_,_ = get_models(); yolo_plate,_,_,_=get_models()
    except Exception as e: raise HTTPException(503,detail=str(e))
    content=await file.read()
    img=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400,"Khong doc duoc anh")
    h_img,w_img=img.shape[:2]
    try: roi_pts=[tuple(p) for p in _json.loads(roi_points)]
    except: roi_pts=[]
    try: line_pts=[tuple(p) for p in _json.loads(line_points)]
    except: line_pts=[]
    use_roi=len(roi_pts)>=3
    vio_sign=violation_side_sign(roi_pts,line_pts) if use_roi else None
    is_demo_image = (file.filename or "").strip().lower() == "ai (1).jpg"
    results_veh=yolo_vehicle.predict(img,classes=[0,2,5,7],conf=conf_vehicle,imgsz=VEHICLE_IMGSZ,agnostic_nms=True,iou=0.5,verbose=False)
    violations,all_vehicles=[],[]
    if results_veh[0].boxes is not None:
        for box in results_veh[0].boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy()); cx,cy=(x1+x2)//2,(y1+y2)//2
            vt=VEHICLE_CLASSES.get(int(box.cls[0].item()),"Khac"); vc=float(box.conf[0].item())
            in_roi=point_in_quad(cx,cy,roi_pts) if use_roi else True
            if not in_roi: continue  # ← bỏ qua hoàn toàn xe ngoài ROI
            all_vehicles.append({"bbox":[x1,y1,x2,y2],"type":vt,"confidence":round(vc,3)})
            veh_side=side_of_line(cx,cy,line_pts) if len(line_pts)>=2 else 0
            on_vio_side = vio_sign is not None and veh_side!=0 and (veh_side>0)==(vio_sign>0)
            is_vio=light_status=="RED" and len(line_pts)>=2 and (line_crosses_bbox(line_pts,[x1,y1,x2,y2]) or on_vio_side)
            if not is_vio: continue
            crop=img[max(0,y1):min(h_img,y2),max(0,x1):min(w_img,x2)]; plate_text="KHO_DOC_BIEN"; ocr_conf=0.0
            evidence_path=None
            if crop.size>0:
                pl_res=yolo_plate.predict(enhance_dark_image(crop),conf=conf_plate,verbose=False)
                n_boxes_seen=0; n_boxes_rejected=0; found_plate=False
                for pr in pl_res:
                    if pr.boxes is None: continue
                    for pb in pr.boxes:
                        n_boxes_seen+=1
                        px1,py1,px2,py2=map(int,pb.xyxy[0].cpu().numpy()); bw,bh=px2-px1,py2-py1
                        pb_conf=float(pb.conf[0].item())
                        if not is_valid_plate_box(bw,bh):
                            n_boxes_rejected+=1
                            print(f"[PLATE-DETECT] Xe {vt} bbox={x1},{y1},{x2},{y2}: YOLO thấy box biển "
                                  f"({bw}x{bh}px, conf={pb_conf:.2f}) nhưng BỊ LOẠI vì sai kích thước/tỉ lệ "
                                  f"(is_valid_plate_box từ chối) -> không chạy OCR")
                            continue
                        pl_crop=crop[max(0,py1-4):min(crop.shape[0],py2+4),max(0,px1-4):min(crop.shape[1],px2+4)]
                        if pl_crop.size==0:
                            print(f"[PLATE-DETECT] Xe {vt} bbox={x1},{y1},{x2},{y2}: box biển hợp lệ nhưng "
                                  f"crop ra ảnh rỗng (size=0) -> không chạy OCR")
                            continue
                        found_plate=True
                        print(f"[PLATE-DETECT] Xe {vt} bbox={x1},{y1},{x2},{y2}: YOLO thấy box biển "
                              f"({bw}x{bh}px, conf={pb_conf:.2f}), hợp lệ -> chạy OCR...")
                        cands,_=_ocr_one_crop(pl_crop,ocr_confidence); pt,bc,_=majority_vote(cands)
                        if pt and bc>ocr_conf: plate_text=pt; ocr_conf=bc
                        break
                    break
                if not found_plate:
                    if n_boxes_seen==0:
                        print(f"[PLATE-DETECT] Xe {vt} bbox={x1},{y1},{x2},{y2}: YOLO KHÔNG phát hiện được "
                              f"box biển số nào trong ảnh xe (conf_plate={conf_plate}) -> không chạy OCR")
                    else:
                        print(f"[PLATE-DETECT] Xe {vt} bbox={x1},{y1},{x2},{y2}: thấy {n_boxes_seen} box "
                              f"nhưng cả {n_boxes_rejected} đều bị loại -> không chạy OCR")
                uid2=uuid.uuid4().hex[:6]; ts=datetime.now().strftime("%H%M%S")
                evidence_fname=f"evidence_{plate_text}_{ts}_{uid2}.jpg"
                cv2.imwrite(str(EVIDENCE_DIR/evidence_fname),crop)
                evidence_path=f"/evidence/{evidence_fname}"
            if plate_text=="KHO_DOC_BIEN" and is_demo_image:
                fb=get_fallback_plate(vt, cx, cy, w_img, h_img)
                if fb: plate_text=fb
            # Ảnh toàn cảnh: nguyên khung hình gốc (img), chỉ box riêng xe vi phạm này —
            # các xe khác trong ảnh (nếu có) không bị vẽ box.
            overview_path = save_overview_evidence(img, [x1,y1,x2,y2], plate_text=plate_text)
            violations.append({"plate_text":plate_text,"vehicle_type":vt,"confidence":round(vc,3),"ocr_conf":round(ocr_conf,3),"bbox":[x1,y1,x2,y2],"evidence_path":evidence_path,"overview_path":overview_path})
            save_violation_to_db(plate_text, vt, evidence_path=evidence_path, overview_path=overview_path, created_by=user["username"])
    return {"light_status":light_status,"violations":violations,"all_vehicles":all_vehicles,"total_violations":len(violations)}

import concurrent.futures as _cf
import threading as _threading

# ── JOB QUEUE (thay thế blocking process-video) ─────────────────────────────
# Dùng ThreadPoolExecutor + dict trạng thái — không cần Celery/Redis.
# Mỗi request POST /process-video trả về job_id ngay lập tức (<1s).
# Frontend polling GET /jobs/{job_id} để theo dõi tiến độ & nhận kết quả.
_job_executor = _cf.ThreadPoolExecutor(max_workers=3)  # tối đa 3 video chạy song song
_jobs: dict = {}   # job_id -> {status, progress, result, error, username, submitted_at}
_jobs_lock = _threading.Lock()

# Thread pool riêng cho việc đọc biển số (OCR) ở giám sát đa cam — tách khỏi _job_executor
# (dùng cho xử lý video hàng loạt) để việc đọc biển số không bị kẹt chờ phía sau các job
# video dài đang chạy. Đọc biển số vốn chỉ ~vài trăm ms nhưng lại làm cả request stream-frame
# bị chặn tới 3-4 giây nếu chạy đồng bộ trong lúc trả kết quả box -> tách ra chạy nền.
_ocr_bg_executor = _cf.ThreadPoolExecutor(max_workers=2)

# Thread pool cho bước YOLO detect xe trong /stream-frame (giám sát đa cam). QUAN TRỌNG:
# trước đây yolo_vehicle.predict(...) được gọi TRỰC TIẾP (đồng bộ) ngay trong hàm async
# stream_frame -> mỗi lần 1 cam gọi predict (thường mất 0.1-2s tuỳ CPU/GPU) sẽ CHIẾM LUÔN
# event loop, khiến MỌI request khác (kể cả /stream-frame của các camera KHÁC) phải xếp
# hàng chờ xong mới được xử lý. Với đúng 1 cam thì ít lộ ra, nhưng đây là tab "giám sát
# NHIỀU camera cùng lúc" nên N cam sẽ cộng dồn độ trễ thành N lần thay vì chạy song song
# -> khớp chính xác với cảnh báo "process_ms>500 -> box bị trễ" mà code đã tự log ra.
# Đưa predict() vào executor để giải phóng event loop cho các cam khác trong lúc chờ.
_detect_executor = _cf.ThreadPoolExecutor(max_workers=DETECT_WORKERS)  # đổi qua env DETECT_WORKERS, không sửa số cứng ở đây

def _run_video_job(job_id: str, tmp_in: str, tmp_out: str,
                   schedule, roi_pts, line_pts,
                   conf_vehicle, conf_plate, ocr_confidence,
                   frame_skip, frame_w, frame_h, username: str,
                   turn_zone_pts=None, turn_confirm_pts=None):
    """Hàm thật chạy trên thread riêng. Cập nhật _jobs[job_id] theo tiến độ.
    turn_zone_pts/turn_confirm_pts: TÙY CHỌN — bỏ trống thì hành vi giữ nguyên như cũ (phạt
    ngay khi cắt vạch lúc đèn đỏ). Xem cơ chế đầy đủ ở resolve_turn_pending()."""
    turn_zone_pts = turn_zone_pts or []
    turn_confirm_pts = turn_confirm_pts or []
    use_turn_zone = len(turn_zone_pts) >= 3
    def set_progress(p, msg=""):
        with _jobs_lock:
            _jobs[job_id]["progress"] = p
            _jobs[job_id]["message"] = msg

    try:
        yolo_plate, yolo_vehicle, _, _ = get_models()
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id].update({"status":"error","error":str(e)})
        return

    use_roi = len(roi_pts) >= 3
    # Phía nào của vạch là phía "đã vượt/vi phạm". Thử suy trước từ ROI (violation_side_sign) —
    # nhưng CHỈ đúng khi vạch thực sự cắt qua ROI (ROI có điểm ở cả 2 phía vạch). Trong thực tế
    # rất nhiều camera vẽ ROI không cắt qua vạch (ROI chỉ nằm 1 bên) -> violation_side_sign()
    # trả về None -> mất hẳn khả năng bắt các track không có lịch sử "phía cũ". Để KHÔNG phụ
    # thuộc vào việc vẽ ROI đúng cách, ta còn "tự học" dấu vi phạm ngay trong lúc chạy: ngay khi
    # có 1 track BẤT KỲ được xác nhận vi phạm theo cách cũ (có "phía cũ" + phát hiện đổi phía —
    # tức có bằng chứng chuyển động thật, không phải đoán), dấu new_side của lần đó CHẮC CHẮN là
    # phía vi phạm -> lưu lại learned_vio_sign, dùng cho các track mới xuất hiện sau đó (đã ở hẳn
    # phía vi phạm ngay từ lần đầu thấy, ví dụ track bị đổi ID giữa chừng đúng lúc xe cắt vạch).
    vio_sign = violation_side_sign(roi_pts, line_pts) if use_roi and len(line_pts) >= 2 else None
    learned_vio_sign = None

    def get_light(sec):
        for s in schedule:
            if s.get("from", 0) <= sec < s.get("to", 999999):
                return s.get("light", "RED")
        return "RED"

    set_progress(2, "Đang mở video...")
    cap = cv2.VideoCapture(tmp_in)
    if not cap.isOpened():
        with _jobs_lock:
            _jobs[job_id].update({"status":"error","error":"Không mở được video"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    # Quy đổi cửa sổ xác minh rẽ (giây) ra số khung hình THẬT theo fps thật của video này.
    turn_confirm_max_frames = max(1, int(TURN_CONFIRM_MAX_SEC * fps))
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    rotate_code = None; width, height = raw_w, raw_h
    if frame_w > 0 and frame_h > 0 and (frame_w, frame_h) != (raw_w, raw_h) and (frame_w, frame_h) == (raw_h, raw_w):
        rotate_code = cv2.ROTATE_90_CLOCKWISE
        width, height = frame_w, frame_h

    out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracks = {}; next_id = 1; TRACK_MAX_MISS = 8
    DRAW_MAX_MISS = 2    # số "lượt" detect liên tiếp không thấy xe mà VẪN được vẽ (giữ track lâu hơn để không đổi ID)
    SMOOTH_ALPHA = 0.6   # làm mượt toạ độ box giữa 2 lần detect (1.0 = tắt làm mượt, dùng thẳng bbox thô)
    logs = []; frame_idx = 0
    set_progress(5, "Đang xử lý video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if rotate_code is not None: frame = cv2.rotate(frame, rotate_code)
        sec = frame_idx / fps; light = get_light(sec)

        if frame_idx % frame_skip == 0:
            rv = yolo_vehicle.predict(frame, classes=[0,2,5,7], conf=conf_vehicle, imgsz=VEHICLE_IMGSZ, agnostic_nms=True, iou=0.5, verbose=False)
            cur_dets = []
            if rv[0].boxes is not None:
                for box in rv[0].boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    vt = VEHICLE_CLASSES.get(int(box.cls[0].item()), "Khac")
                    in_roi = point_in_quad(cx, cy, roi_pts) if use_roi else True
                    if not in_roi: continue
                    cur_dets.append({"bbox":[x1,y1,x2,y2],"type":vt,"cx":cx,"cy":cy})

            det_to_track = assign_tracks_global(cur_dets, tracks, match_score, 0.3)
            matched_track_ids = set(det_to_track.values())
            for di, det in enumerate(cur_dets):
                if di not in det_to_track:
                    best_tid = next_id; next_id += 1
                    tracks[best_tid] = {"bbox":det["bbox"],"disp_bbox":det["bbox"],"type":det["type"],
                                         "side":None,"violated":False,"last_seen":frame_idx,
                                         "plate_text":"","violation_sec":None,
                                         "pending_turn":None,"turn_confirmed":False}
                    det_to_track[di] = best_tid

            for di, det in enumerate(cur_dets):
                tid = det_to_track[di]; tr = tracks[tid]
                x1,y1,x2,y2 = det["bbox"]; cx,cy = det["cx"],det["cy"]; vt = det["type"]
                new_side = side_of_line(cx, cy, line_pts) if len(line_pts) >= 2 else None
                # Chỉ coi vị trí mới là "đáng tin" nếu đủ xa vạch (ngoài vùng đệm) — tránh vị trí
                # sát vạch (dễ nhiễu do bbox rung) được dùng để kết luận vượt vạch.
                new_side_confident = new_side is not None and abs(new_side) >= MIN_LINE_CROSS_DIST_PX
                # "Đã đang ở đúng phía vi phạm" — ưu tiên vio_sign suy từ ROI, nếu ROI không cắt
                # qua vạch (vio_sign=None, rất hay gặp vì ROI/vạch được vẽ độc lập) thì dùng
                # learned_vio_sign (tự học được từ 1 vi phạm CHẮC CHẮN trước đó trong video này).
                effective_vio_sign = vio_sign if vio_sign is not None else learned_vio_sign
                on_vio_side = (effective_vio_sign is not None and new_side_confident
                               and (new_side > 0) == (effective_vio_sign > 0))

                is_vio = False
                if (not tr["violated"]) and (not tr.get("turn_confirmed")) and tr.get("pending_turn") is None \
                        and light == "RED" and new_side_confident:
                    crossed = False
                    if tr["side"] is not None and (tr["side"] > 0) != (new_side > 0):
                        # Cách cũ: track có lịch sử -> vi phạm khi phát hiện ĐỔI phía. Đây là
                        # bằng chứng chuyển động THẬT (không phải đoán) -> dùng để "học" luôn
                        # dấu vi phạm cho các track khác nếu chưa có vio_sign từ ROI.
                        crossed = True
                        if vio_sign is None and learned_vio_sign is None:
                            learned_vio_sign = 1 if new_side > 0 else -1
                    elif tr["side"] is None and on_vio_side:
                        # FIX: track chưa có "phía cũ" (mới tạo, hoặc bị đổi track_id giữa chừng
                        # do mất dấu đúng lúc xe cắt qua vạch — xem assign_tracks_global) nhưng
                        # NGAY LẦN ĐẦU đã thấy xe nằm ở phía vi phạm -> trước đây bị bỏ sót vì
                        # không có gì để so sánh "đổi phía", khiến box xe cứ mãi màu xanh dù xe
                        # đã ở hẳn bên kia vạch trong lúc đèn đỏ. Nay coi luôn là vi phạm, dùng
                        # vio_sign (ROI) hoặc learned_vio_sign (tự học) để biết phía nào là vi phạm.
                        crossed = True
                    if crossed:
                        # Nút giao cho phép rẽ khi đèn đỏ: nếu điểm cắt vạch nằm trong "vùng rẽ
                        # được phép" thì CHƯA kết luận ngay — chuyển sang "đang xác minh" (xem
                        # resolve_turn_pending() bên dưới, mỗi frame). Camera không cấu hình vùng
                        # rẽ (use_turn_zone=False) thì giữ nguyên hành vi cũ — phạt ngay.
                        if use_turn_zone and bbox_in_zone([x1,y1,x2,y2], turn_zone_pts):
                            tr["pending_turn"] = {"since_frame": frame_idx, "start_cx": cx, "start_cy": cy,
                                                   "vio_sign": (1 if new_side > 0 else -1)}
                        else:
                            is_vio = True
                            tr["violated"] = True; tr["violation_sec"] = sec

                # Làm mượt toạ độ hiển thị (EMA) — giảm giật/rung khi bbox thô lệch nhẹ giữa các lần quét,
                # KHÔNG dùng disp_bbox này để so khớp track (match_score vẫn dùng "bbox" thô ở trên).
                ox1,oy1,ox2,oy2 = tr.get("disp_bbox", det["bbox"])
                a = SMOOTH_ALPHA
                tr["disp_bbox"] = [int(a*x1+(1-a)*ox1), int(a*y1+(1-a)*oy1),
                                    int(a*x2+(1-a)*ox2), int(a*y2+(1-a)*oy2)]
                tr["bbox"] = det["bbox"]; tr["type"] = vt; tr["last_seen"] = frame_idx
                # Chỉ cập nhật "side" khi điểm đủ xa vạch (ngoài vùng đệm) — nếu không, giữ nguyên
                # phía cũ đã ghi nhớ (hysteresis), tránh lấy 1 vị trí sát vạch/dễ nhiễu làm mốc so
                # sánh cho lần kiểm tra sau, gây báo vượt vạch giả khi xe đứng yên sát vạch.
                if new_side_confident: tr["side"] = new_side

                # Track đang "chờ xác minh rẽ": giữ 1 ảnh crop mới nhất (ghi đè liên tục CHỈ khi
                # xe còn nằm trong ROI — xem lý do ở nhánh "ra khỏi ROI" bên dưới) phòng khi track
                # bị mất dấu giữa chừng cần đẩy vào hàng đợi rà soát thủ công. Mỗi frame kiểm tra
                # đã đủ căn cứ RẼ HỢP LỆ hay VI PHẠM chưa — xem resolve_turn_pending().
                if tr.get("pending_turn") is not None:
                    still_in_roi = (not use_roi) or bbox_in_zone([x1,y1,x2,y2], roi_pts)
                    if still_in_roi:
                        _pc = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                        if _pc.size > 0: tr["pending_crop"] = _pc.copy()
                    # Video offline có fps CỐ ĐỊNH và biết trước -> frame_idx là đơn vị thời gian
                    # đáng tin cậy ở đây (không giống /stream-frame, xem ghi chú ở đó).
                    outcome = resolve_turn_pending([x1,y1,x2,y2], tr["pending_turn"], line_pts, turn_confirm_pts,
                                                    frame_idx - tr["pending_turn"]["since_frame"],
                                                    turn_confirm_max_frames)
                    if outcome == "wait" and use_roi and not still_in_roi:
                        # Xe đã ra khỏi TẤT CẢ vùng ROI đã vẽ mà chưa từng xác nhận được rẽ hợp lệ
                        # (chưa vào cổng xác nhận) -> không chờ hết giờ nữa, chốt
                        # VI PHẠM ngay tại đây. Ảnh chứng cứ dùng crop đã lưu từ lúc CÒN trong ROI
                        # (tr["pending_crop"]), không dùng khung hiện tại vì xe có thể đã ra khỏi
                        # hẳn khung hình hoặc chỉ còn thấy 1 phần — xem khối OCR/evidence bên dưới.
                        outcome = "violation"
                        tr["_use_cached_evidence"] = True
                    if outcome == "turn":
                        tr["pending_turn"] = None; tr["turn_confirmed"] = True; tr.pop("pending_crop", None)
                    elif outcome == "violation":
                        # Hết cửa sổ xác minh mà vẫn đi thẳng -> xe lợi dụng làn rẽ vượt đèn đỏ.
                        # Chốt vi phạm NGAY TẠI ĐÂY (dùng bbox hiện tại), để khối OCR/evidence
                        # bên dưới xử lý y hệt như 1 vi phạm phát hiện ngay khi cắt vạch.
                        tr["pending_turn"] = None
                        is_vio = True
                        tr["violated"] = True; tr["violation_sec"] = sec

                if is_vio:
                    pt = ""
                    # Trường hợp chốt vi phạm vì "ra khỏi ROI lúc đang xác minh rẽ": dùng ảnh đã
                    # lưu từ lúc xe CÒN trong ROI thay vì crop khung hiện tại (khung hiện tại xe
                    # đã ở ngoài ROI, có thể đã ra khỏi khung hình hoặc chỉ còn thấy 1 phần).
                    if tr.pop("_use_cached_evidence", False) and tr.get("pending_crop") is not None \
                            and tr["pending_crop"].size > 0:
                        crop = tr["pending_crop"]
                    else:
                        crop = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                    evidence_path = None
                    if crop.size > 0:
                        rp = yolo_plate.predict(enhance_dark_image(crop), conf=conf_plate, verbose=False)
                        for pr in rp:
                            if pr.boxes is None: continue
                            for pb in pr.boxes:
                                px1,py1,px2,py2 = map(int, pb.xyxy[0].cpu().numpy())
                                if not is_valid_plate_box(px2-px1, py2-py1): continue
                                pc = crop[max(0,py1-4):min(crop.shape[0],py2+4), max(0,px1-4):min(crop.shape[1],px2+4)]
                                if pc.size == 0: continue
                                cands, _ = _ocr_one_crop(pc, ocr_confidence)
                                p2, _, _ = majority_vote(cands)
                                if p2: pt = p2
                                break
                            break
                        uid3 = uuid.uuid4().hex[:6]; ts3 = datetime.now().strftime("%H%M%S")
                        evidence_fname = f"evidence_{pt or 'unknown'}_{ts3}_{uid3}.jpg"
                        cv2.imwrite(str(EVIDENCE_DIR / evidence_fname), crop)
                        evidence_path = f"/evidence/{evidence_fname}"
                    # Ảnh toàn cảnh: dùng "frame" gốc (chưa crop) tại đúng frame xe vượt vạch,
                    # chỉ khoanh box đỏ quanh xe vi phạm này, các xe khác trong khung không bị box.
                    overview_path = save_overview_evidence(frame, [x1,y1,x2,y2], plate_text=pt)
                    tr["plate_text"] = pt if pt else f"#{tid} CHUA DOC DUOC"
                    logs.append({"frame":frame_idx,"time_s":round(sec,1),"vehicle_type":vt,
                                 "plate_text":pt or "KHO_DOC_BIEN","light":light,
                                 "track_id":tid,"evidence_path":evidence_path,"overview_path":overview_path})
                    save_violation_to_db(pt, vt, evidence_path=evidence_path, overview_path=overview_path,
                                         video_time_s=sec, track_id=tid, created_by=username)

            stale = [tid for tid, tr in tracks.items() if frame_idx - tr["last_seen"] > TRACK_MAX_MISS * frame_skip]
            for tid in stale:
                str_tr = tracks[tid]
                if str_tr.get("pending_turn") is not None:
                    _pc = str_tr.get("pending_crop")
                    if _pc is not None and _pc.size > 0:
                        # Mất dấu (che khuất, ra khỏi khung hình...) TRONG lúc đang xác minh rẽ,
                        # CHƯA từng xác nhận được rẽ hợp lệ -> coi như "ra khỏi vùng theo dõi mà
                        # chưa chứng minh được đang rẽ", chốt VI PHẠM luôn (không chờ tới lúc mất
                        # dấu hẳn mới lặng lẽ đẩy vào hàng đợi rà soát — theo đúng đề xuất: không
                        # đi qua cổng xác nhận thì phải chốt, ảnh chứng cứ dùng khung LÚC CÒN THẤY
                        # xe rõ nhất, không phải khung hiện tại vì xe đã biến mất khỏi khung hình).
                        pt = ""
                        rp = yolo_plate.predict(enhance_dark_image(_pc), conf=conf_plate, verbose=False)
                        for pr in rp:
                            if pr.boxes is None: continue
                            for pb in pr.boxes:
                                px1,py1,px2,py2 = map(int, pb.xyxy[0].cpu().numpy())
                                if not is_valid_plate_box(px2-px1, py2-py1): continue
                                pc2 = _pc[max(0,py1-4):min(_pc.shape[0],py2+4), max(0,px1-4):min(_pc.shape[1],px2+4)]
                                if pc2.size == 0: continue
                                cands, _ = _ocr_one_crop(pc2, ocr_confidence)
                                p2, _, _ = majority_vote(cands)
                                if p2: pt = p2
                                break
                            break
                        uid3 = uuid.uuid4().hex[:6]; ts3 = datetime.now().strftime("%H%M%S")
                        evidence_fname = f"evidence_{pt or 'unknown'}_{ts3}_{uid3}.jpg"
                        cv2.imwrite(str(EVIDENCE_DIR / evidence_fname), _pc)
                        evidence_path = f"/evidence/{evidence_fname}"
                        vsec = str_tr["pending_turn"]["since_frame"]/fps
                        logs.append({"frame":frame_idx,"time_s":round(vsec,1),"vehicle_type":str_tr.get("type"),
                                     "plate_text":pt or "KHO_DOC_BIEN","light":"RED",
                                     "track_id":tid,"evidence_path":evidence_path,"overview_path":evidence_path})
                        save_violation_to_db(pt, str_tr.get("type"), evidence_path=evidence_path,
                                             overview_path=evidence_path, video_time_s=vsec,
                                             track_id=tid, created_by=username)
                    else:
                        # Hiếm gặp: mất dấu ngay khung đầu tiên, CHƯA kịp lưu được ảnh nào cả ->
                        # không có gì làm bằng chứng, đẩy cán bộ xem lại thay vì phạt không ảnh.
                        push_review_case(str_tr.get("type"), track_id=tid,
                                          video_time_s=str_tr["pending_turn"]["since_frame"]/fps,
                                          snapshot_path=None, created_by=username)
                del tracks[tid]

        # ─── VẼ BOX Ở MỌI FRAME (kể cả frame không chạy detect) ───────────────────────────
        # Trước đây toàn bộ cv2.rectangle/putText nằm TRONG khối "if frame_idx % frame_skip==0",
        # nên cứ (frame_skip-1) frame liên tiếp lại không có box nào được vẽ -> mắt thấy box
        # chớp tắt liên tục theo đúng chu kỳ frame_skip. Nay ta luôn vẽ theo "disp_bbox" đang lưu
        # trong tracks (chỉ giá trị này được CẬP NHẬT mỗi frame_skip frame, còn hiển thị thì mọi
        # frame) -> box hiện liên tục, đứng yên giữa 2 lần cập nhật thay vì biến mất.
        for tid, tr in tracks.items():
            if frame_idx - tr["last_seen"] > DRAW_MAX_MISS * frame_skip:
                continue  # mất dấu hơi lâu rồi (rất có thể xe đã đi khỏi/đổi track) -> ẩn box, tránh để lại "tàn ảnh"
            x1,y1,x2,y2 = tr["disp_bbox"]; vt = tr["type"]
            if tr["violated"]: col = (0,0,255)
            elif tr.get("pending_turn") is not None: col = (0,190,255)  # hổ phách — đang xác minh rẽ
            else: col = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, f"#{tid} {vt[:6]}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            if tr["violated"]:
                cv2.putText(frame, "VI PHAM!", (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.putText(frame, tr["plate_text"], (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                cv2.putText(frame, f"Vi pham luc {tr['violation_sec']:.1f}s", (x1,y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            elif tr.get("pending_turn") is not None:
                cv2.putText(frame, "DANG XAC MINH RE...", (x1,y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,190,255), 2)

        lc = (0,0,255) if light == "RED" else (0,255,0)
        cv2.rectangle(frame, (10,10), (185,55), (30,30,30), -1)
        cv2.circle(frame, (35,33), 14, lc, -1)
        cv2.putText(frame, f"DEN {light} | {sec:.1f}s", (58,40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        if len(line_pts) >= 2:
            p1 = tuple(map(int, line_pts[0])); p2 = tuple(map(int, line_pts[1]))
            cv2.line(frame, p1, p2, (0,0,255), 2)
        if use_roi:
            pts = np.array(roi_pts, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(frame, [pts], True, (0,255,255), 2)
        if use_turn_zone:
            pts_tz = np.array(turn_zone_pts, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(frame, [pts_tz], True, (255,190,0), 2)
        if len(turn_confirm_pts) >= 3:
            pts_tc = np.array(turn_confirm_pts, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(frame, [pts_tc], True, (0,140,255), 2)
        out.write(frame); frame_idx += 1

        # Cập nhật progress mỗi 30 frame (tránh lock quá nhiều)
        if frame_idx % 30 == 0:
            pct = min(95, 5 + int(frame_idx / max(total_frames, 1) * 90))
            set_progress(pct, f"Đang xử lý... {frame_idx}/{total_frames} frames, {len(logs)} vi phạm")

    cap.release(); out.release()
    set_progress(97, "Đang encode video kết quả...")
    with open(tmp_out, "rb") as f: vb64 = base64.b64encode(f.read()).decode()
    try: os.remove(tmp_in)
    except: pass
    try: os.remove(tmp_out)
    except: pass

    with _jobs_lock:
        _jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "message": f"Hoàn tất — {len(logs)} vi phạm phát hiện",
            "result": {
                "total_frames": frame_idx,
                "total_violations": len(logs),
                "violation_logs": logs,
                "video_b64": vb64,
            }
        })
    write_audit({"username": username, "role": "system"}, "PROCESS_VIDEO_DONE",
                detail=f"job {job_id} — {len(logs)} vi phạm")


@app.post("/process-video")
async def process_video(
    file: UploadFile=File(...), light_schedule: str=Form("[]"),
    roi_points: str=Form("[]"), line_points: str=Form("[]"),
    turn_zone_points: str=Form("[]"), turn_confirm_points: str=Form("[]"),
    conf_vehicle: float=Form(0.15), conf_plate: float=Form(0.35),
    ocr_confidence: float=Form(0.30), frame_skip: int=Form(3),
    frame_w: int=Form(0), frame_h: int=Form(0),
    user: dict = Depends(require_role("admin", "truc_camera")),
):
    """Nhận file video, trả job_id ngay lập tức (<1s).
    Xử lý chạy nền trên ThreadPool — không block request khác.
    Frontend polling GET /jobs/{job_id} để theo dõi tiến độ.
    turn_zone_points/turn_confirm_points: TÙY CHỌN, dùng cho nút giao cho phép rẽ khi đèn đỏ —
    bỏ trống (mảng rỗng, mặc định) thì hành vi giữ nguyên như cũ."""
    try: schedule = _json.loads(light_schedule)
    except: schedule = []
    try: roi_pts = [tuple(p) for p in _json.loads(roi_points)]
    except: roi_pts = []
    try: line_pts = [tuple(p) for p in _json.loads(line_points)]
    except: line_pts = []
    try: turn_zone_pts = [tuple(p) for p in _json.loads(turn_zone_points)]
    except: turn_zone_pts = []
    try: turn_confirm_pts = [tuple(p) for p in _json.loads(turn_confirm_points)]
    except: turn_confirm_pts = []

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_in = tempfile.mktemp(suffix=suffix)
    tmp_out = tempfile.mktemp(suffix=".mp4")
    with open(tmp_in, "wb") as f: f.write(await file.read())

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Đang chờ trong hàng đợi...",
            "result": None,
            "error": None,
            "username": user["username"],
            "submitted_at": datetime.now().isoformat(),
        }

    _job_executor.submit(
        _run_video_job, job_id, tmp_in, tmp_out,
        schedule, roi_pts, line_pts,
        conf_vehicle, conf_plate, ocr_confidence,
        frame_skip, frame_w, frame_h, user["username"],
        turn_zone_pts, turn_confirm_pts,
    )
    # Đổi status sang processing ngay (thread có thể chưa chạy tới)
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"

    write_audit(user, "SUBMIT_VIDEO_JOB", detail=f"job_id={job_id}")
    return {"job_id": job_id, "status": "processing",
            "message": "Video đã vào hàng đợi xử lý, theo dõi tại GET /jobs/{job_id}"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, user: dict = Depends(get_current_user)):
    """Polling endpoint — trả về trạng thái + tiến độ + kết quả khi xong."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job")
    resp = {
        "job_id": job_id,
        "status": job["status"],       # queued | processing | done | error
        "progress": job["progress"],   # 0-100
        "message": job["message"],
        "submitted_at": job["submitted_at"],
        "submitted_by": job["username"],
    }
    if job["status"] == "done":
        resp["result"] = job["result"]
    if job["status"] == "error":
        resp["error"] = job["error"]
    return resp

@app.get("/jobs")
async def list_jobs(user: dict = Depends(require_role("admin", "truc_camera"))):
    """Xem toàn bộ job trong hàng đợi (chỉ admin/trực camera)."""
    with _jobs_lock:
        jobs = [
            {"job_id": jid, "status": j["status"], "progress": j["progress"],
             "message": j["message"], "submitted_by": j["username"],
             "submitted_at": j["submitted_at"]}
            for jid, j in _jobs.items()
        ]
    jobs.sort(key=lambda x: x["submitted_at"], reverse=True)
    return {"total": len(jobs), "jobs": jobs}

@app.post("/realtime-frame")
async def realtime_frame(
    file: UploadFile=File(...), light_status: str=Form("RED"), has_roi: str=Form("false"),
    roi_x1: int=Form(0), roi_y1: int=Form(0), roi_x2: int=Form(9999), roi_y2: int=Form(9999),
    line_y: int=Form(0), conf_vehicle: float=Form(0.15), conf_plate: float=Form(0.30), ocr_confidence: float=Form(0.25),
    user: dict = Depends(require_role("admin", "truc_camera"))
):
    try: _,yolo_vehicle,_,_=get_models(); yolo_plate,_,_,_=get_models()
    except Exception as e: return {"error":str(e),"detections":[],"violations":[],"total_vehicles":0}
    content=await file.read()
    img=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
    if img is None: return {"error":"bad image","detections":[],"violations":[],"total_vehicles":0}
    use_roi=has_roi.lower()=="true"; h_img,w_img=img.shape[:2]
    rv=yolo_vehicle.predict(img,classes=[0,2,5,7],conf=conf_vehicle,imgsz=VEHICLE_IMGSZ,agnostic_nms=True,iou=0.5,verbose=False)
    detections,violations=[],[]
    if rv[0].boxes is not None:
        for box in rv[0].boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy()); cx,cy=(x1+x2)//2,(y1+y2)//2
            vt=VEHICLE_CLASSES.get(int(box.cls[0].item()),"Khac"); vc=float(box.conf[0].item())
            in_roi=(roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2) if use_roi else True
            is_vio=in_roi and light_status=="RED" and line_y>0 and cy<line_y
            pt=""
            if is_vio:
                crop=img[max(0,y1):min(h_img,y2),max(0,x1):min(w_img,x2)]
                if crop.size>0:
                    rp=yolo_plate.predict(enhance_dark_image(crop),conf=conf_plate,verbose=False)
                    for pr in rp:
                        if pr.boxes is None: continue
                        for pb in pr.boxes:
                            px1,py1,px2,py2=map(int,pb.xyxy[0].cpu().numpy())
                            if not is_valid_plate_box(px2-px1,py2-py1): continue
                            pc=crop[max(0,py1-4):min(crop.shape[0],py2+4),max(0,px1-4):min(crop.shape[1],px2+4)]
                            if pc.size==0: continue
                            cands,_=_ocr_one_crop(pc,ocr_confidence); p2,_,_=majority_vote(cands)
                            if p2: pt=p2
                            break
                        break
                violations.append({"plate_text":pt,"vehicle_type":vt,"bbox":[x1,y1,x2,y2]})
                save_violation_to_db(pt, vt, created_by=user["username"])
            detections.append({"bbox":[x1,y1,x2,y2],"type":vt,"confidence":round(vc,3),"is_violation":is_vio,"plate_text":pt})
    return {"detections":detections,"violations":violations,"total_vehicles":len(detections)}


# ════════════════════════════════════════════════════════════════════════════
# STREAM-FRAME — Giám sát đa cam với logic đầy đủ như process_video:
#   tracking liên tục (IoU matching), phát hiện vượt vạch khi đèn đỏ,
#   đọc biển số, lưu hồ sơ vi phạm — kết quả trả ngay trên màn hình.
# Khác process_video: không cần file video, xử lý từng frame rời gửi lên.
# State (tracker, lịch sử vi phạm) được lưu per-camera trong dict bộ nhớ.
# ════════════════════════════════════════════════════════════════════════════

# Dict lưu state riêng cho mỗi camera_id
# cam_states[cam_id] = {tracks, next_id, frame_idx, violated_tracks}
cam_states: dict = {}
STREAM_TRACK_MAX_MISS = 12   # số frame không thấy thì xoá track (giữ dài để không làm gián đoạn
                              # bám các track hợp lệ đang di chuyển nhanh/bị che khuất tạm thời;
                              # rủi ro ghép nhầm vào track đã violated đã được chặn riêng, đúng chỗ,
                              # ở assign_tracks_global — không cần rút ngắn tuổi thọ track để bù)

def get_cam_state(cam_id: str) -> dict:
    if cam_id not in cam_states:
        cam_states[cam_id] = {
            "tracks": {},      # tid -> {bbox, type, side, violated, last_seen, plate, pending_turn, turn_confirmed}
            "next_id": 1,
            "frame_idx": 0,
            "violation_count": 0,
            # Dấu (sign) đại diện phía "đã vượt/vi phạm" của vạch dừng, tự học được ngay trong
            # phiên giám sát này (xem chi tiết lý do cần cái này ở /stream-frame bên dưới).
            # Persist theo cam_id vì /stream-frame được gọi lại nhiều lần (mỗi lần 1 frame),
            # không giữ được biến cục bộ giữa các lần gọi như hàm xử lý video 1 lần chạy hết.
            "learned_vio_sign": None,
        }
    return cam_states[cam_id]

def _run_plate_ocr_bg(crop, tid, vt, username, tracks, conf_plate, ocr_confidence,
                       full_frame=None, bbox=None):
    """Chạy NGẦM trong thread pool riêng (_ocr_bg_executor) — KHÔNG chặn response của
    /stream-frame. Đọc biển số xong thì cập nhật tracks[tid]['plate'] để lần poll kế
    tiếp từ trình duyệt tự động nhận được biển số thật (thay vì phải chờ ngay trong
    request lúc phát hiện vi phạm, vốn có thể mất 0.5-3s+ và làm box bị đứng hình).
    full_frame/bbox (nếu truyền vào): dùng để lưu thêm ảnh TOÀN CẢNH — nguyên khung hình
    tại đúng lúc xe vượt vạch, chỉ box riêng xe vi phạm này, khác với "crop" vốn chỉ chứa
    mỗi chiếc xe."""
    plate_text, evidence_path, overview_path = "", None, None
    try:
        yolo_plate, _, _, _ = get_models()
        if crop is not None and crop.size > 0:
            rp = yolo_plate.predict(enhance_dark_image(crop), conf=conf_plate, verbose=False)
            for pr in rp:
                if pr.boxes is None: continue
                for pb in pr.boxes:
                    px1, py1, px2, py2 = map(int, pb.xyxy[0].cpu().numpy())
                    if not is_valid_plate_box(px2-px1, py2-py1): continue
                    pc = crop[max(0, py1-4):min(crop.shape[0], py2+4),
                              max(0, px1-4):min(crop.shape[1], px2+4)]
                    if pc.size == 0: continue
                    cands, _ = _ocr_one_crop(pc, ocr_confidence)
                    pt, _, _ = majority_vote(cands)
                    if pt: plate_text = pt
                    break
                break
            uid3 = uuid.uuid4().hex[:6]; ts3 = datetime.now().strftime("%H%M%S")
            evidence_fname = f"evidence_{plate_text or 'unknown'}_{ts3}_{uid3}.jpg"
            cv2.imwrite(str(EVIDENCE_DIR / evidence_fname), crop)
            evidence_path = f"/evidence/{evidence_fname}"
        if full_frame is not None and bbox is not None:
            overview_path = save_overview_evidence(full_frame, bbox, plate_text=plate_text)
    except Exception as e:
        print(f"[OCR-BG] Lỗi đọc biển số nền (track {tid}): {e}")
    finally:
        tr = tracks.get(tid)  # track có thể đã bị xoá (xe rời khung hình lâu) -> bỏ qua
        if tr is not None:
            tr["plate"] = plate_text or "KHO_DOC_BIEN"
            tr["evidence_path"] = evidence_path
            tr["overview_path"] = overview_path
        try:
            save_violation_to_db(plate_text, vt, evidence_path=evidence_path, overview_path=overview_path,
                                  track_id=tid, created_by=username)
        except Exception as e:
            print(f"[OCR-BG] Lỗi lưu vi phạm vào DB (track {tid}): {e}")

@app.post("/stream-frame/{cam_id}")
async def stream_frame(
    cam_id: str,
    file: UploadFile = File(...),
    light_status: str = Form("RED"),
    roi_points: str = Form("[]"),   # JSON [[x,y],...] polygon ≥3 điểm (giống tab Video)
    line_points: str = Form("[]"),  # JSON [[x1,y1],[x2,y2]] đường nghiêng (giống tab Video)
    turn_zone_points: str = Form("[]"),    # TÙY CHỌN — vùng rẽ được phép (nút giao cho phép rẽ khi đèn đỏ)
    turn_confirm_points: str = Form("[]"), # TÙY CHỌN — cổng xác nhận rẽ, sang đường cắt ngang
    conf_vehicle: float = Form(0.15),
    conf_plate: float = Form(0.35),
    ocr_confidence: float = Form(0.25),
    reset: str = Form("false"),
    user: dict = Depends(require_role("admin", "truc_camera")),
):
    """
    Endpoint xử lý từng frame từ tab Giám sát đa cam.
    Giữ state tracker riêng per cam_id — tracking liên tục qua nhiều frame,
    phát hiện vi phạm đầy đủ (side-of-line + đèn đỏ), OCR biển số, lưu hồ sơ.
    """
    try: yolo_plate, yolo_vehicle, _, _ = get_models()
    except Exception as e:
        return {"error": str(e), "detections": [], "violations": [], "new_violations": []}

    _t0 = time.time()  # đo thời gian xử lý thực tế -> in log cuối hàm để biết chính xác đang chậm ở đâu

    if reset == "true" and cam_id in cam_states:
        del cam_states[cam_id]

    state = get_cam_state(cam_id)
    state["frame_idx"] += 1
    frame_idx = state["frame_idx"]

    # Parse ROI polygon và line 2 điểm — y hệt cách tab Video gửi lên
    try: roi_pts = [tuple(p) for p in _json.loads(roi_points)]
    except: roi_pts = []
    try: line_pts = [tuple(p) for p in _json.loads(line_points)]
    except: line_pts = []
    try: turn_zone_pts = [tuple(p) for p in _json.loads(turn_zone_points)]
    except: turn_zone_pts = []
    try: turn_confirm_pts = [tuple(p) for p in _json.loads(turn_confirm_points)]
    except: turn_confirm_pts = []
    use_roi = len(roi_pts) >= 3
    has_line = len(line_pts) >= 2
    use_turn_zone = len(turn_zone_pts) >= 3
    # QUAN TRỌNG: khác với video offline (fps cố định, biết trước), /stream-frame được frontend
    # gọi liên tục — tốc độ poll thật phụ thuộc tốc độ xử lý (mô hình, mạng, tải máy chủ...), có
    # thể vài chục lần/giây, KHÔNG PHẢI ~1 lần/giây. Coi 1 lần gọi ~ 1 giây (bản trước đây làm vậy)
    # khiến cửa sổ "đang xác minh rẽ" hết hạn chỉ sau chưa tới 1 giây thực tế trong thực tế chạy
    # nhanh -> xe đang rẽ đúng bị chốt VI PHẠM trước khi kịp vào vùng cam. Sửa: dùng THỜI GIAN THỰC
    # (time.time()) để đo cửa sổ xác minh rẽ cho đúng TURN_CONFIRM_MAX_SEC giây thật, bất kể tốc
    # độ poll nhanh hay chậm.
    # Y hệt cơ chế đã áp cho tab Video: suy phía vi phạm từ ROI nếu vạch cắt qua ROI, nếu không
    # (rất hay gặp vì ROI/vạch vẽ độc lập với nhau) thì dùng learned_vio_sign đã tự học được từ
    # phiên giám sát trước đó của CHÍNH camera này (persist trong cam_states[cam_id]).
    vio_sign = violation_side_sign(roi_pts, line_pts) if use_roi and has_line else None

    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "bad image", "detections": [], "violations": [], "new_violations": []}

    h, w = img.shape[:2]

    # ── Phát hiện xe ── (chạy trong thread pool riêng, KHÔNG gọi trực tiếp trên event loop
    # chính — nếu không thì trong lúc cam A đang predict, mọi cam khác đang poll /stream-frame
    # cùng lúc đều phải đứng chờ, nên càng nhiều cam thì độ trễ càng cộng dồn theo cấp số nhân)
    rv = await asyncio.get_event_loop().run_in_executor(
        _detect_executor,
        lambda: yolo_vehicle.predict(img, classes=[0, 2, 5, 7], conf=conf_vehicle, imgsz=VEHICLE_IMGSZ, agnostic_nms=True, iou=0.5, verbose=False)
    )
    cur_dets = []
    if rv[0].boxes is not None:
        for box in rv[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # ROI là ranh giới phát hiện: không đưa detection bên ngoài vào tracker.
            if use_roi and not point_in_quad(cx, cy, roi_pts): continue
            vt = VEHICLE_CLASSES.get(int(box.cls[0].item()), "Khac")
            cur_dets.append({"bbox": [x1, y1, x2, y2], "cx": cx, "cy": cy,
                             "type": vt, "conf": float(box.conf[0])})

    # ── IoU matching — tracking liên tục ──
    tracks = state["tracks"]
    det_to_track = assign_tracks_global(cur_dets, tracks, match_score_stream, 0.15)
    matched_track_ids = set(det_to_track.values())
    for di, det in enumerate(cur_dets):
        if di not in det_to_track:
            best_tid = state["next_id"]; state["next_id"] += 1
            tracks[best_tid] = {"bbox": det["bbox"], "type": det["type"],
                                "side": None, "violated": False,
                                "last_seen": frame_idx, "plate": "",
                                "pending_turn": None, "turn_confirmed": False}
            det_to_track[di] = best_tid

    # ── Kiểm tra vi phạm + OCR ──
    detections = []
    new_violations = []

    for di, det in enumerate(cur_dets):
        tid = det_to_track[di]; tr = tracks[tid]
        x1, y1, x2, y2 = det["bbox"]; cx, cy = det["cx"], det["cy"]; vt = det["type"]

        # Phía xe so với đường vạch dừng 2 điểm nghiêng — dùng side_of_line như process_video
        new_side = side_of_line(cx, cy, line_pts) if has_line else None
        # Chỉ coi vị trí mới là "đáng tin" nếu đủ xa vạch (ngoài vùng đệm) — quan trọng hơn cả ở
        # đây vì /stream-frame chỉ nhận ~1 frame/giây (gap lớn hơn nhiều so với process_video),
        # nên jitter tương đối giữa 2 lần đo dễ bị khuếch đại hơn.
        new_side_confident = new_side is not None and abs(new_side) >= MIN_LINE_CROSS_DIST_PX
        effective_vio_sign = vio_sign if vio_sign is not None else state["learned_vio_sign"]
        on_vio_side = (effective_vio_sign is not None and new_side_confident
                       and (new_side > 0) == (effective_vio_sign > 0))
        is_new_vio = False

        if (not tr["violated"]) and (not tr.get("turn_confirmed")) and tr.get("pending_turn") is None \
                and light_status == "RED" and has_line and new_side_confident:
            crossed = False
            if tr["side"] is not None and (tr["side"] > 0) != (new_side > 0):
                # Có lịch sử -> vi phạm khi phát hiện ĐỔI phía (bằng chứng chuyển động thật).
                crossed = True
                if vio_sign is None and state["learned_vio_sign"] is None:
                    state["learned_vio_sign"] = 1 if new_side > 0 else -1
            elif tr["side"] is None and on_vio_side:
                # FIX: track chưa có "phía cũ" (mới tạo, hoặc bị đổi track_id giữa chừng do mất
                # dấu đúng lúc xe cắt qua vạch) nhưng NGAY LẦN ĐẦU đã ở phía vi phạm -> trước đây
                # bị bỏ sót hoàn toàn ở /stream-frame vì không có gì để so sánh "đổi phía", khiến
                # box xe cứ mãi màu xanh dù xe đã ở hẳn bên kia vạch trong lúc đèn đỏ.
                crossed = True
            if crossed:
                # Nút giao cho phép rẽ khi đèn đỏ: cắt vạch NGAY TRONG vùng rẽ được phép thì
                # CHƯA kết luận ngay — chuyển sang "đang xác minh" (xem resolve_turn_pending() ở
                # dưới, chạy lại mỗi lần có detection mới cho track này). Không cấu hình vùng rẽ
                # (use_turn_zone=False) thì giữ nguyên hành vi cũ — phạt ngay.
                if use_turn_zone and bbox_in_zone([x1,y1,x2,y2], turn_zone_pts):
                    # since_ts = mốc THỜI GIAN THỰC (time.time()), KHÔNG dùng frame_idx — xem giải
                    # thích ở đầu hàm vì sao (tốc độ poll không cố định, khác video offline).
                    tr["pending_turn"] = {"since_frame": frame_idx, "since_ts": time.time(),
                                           "start_cx": cx, "start_cy": cy,
                                           "vio_sign": (1 if new_side > 0 else -1)}
                else:
                    is_new_vio = True; tr["violated"] = True
                    state["violation_count"] += 1

        tr["bbox"] = det["bbox"]; tr["type"] = vt; tr["last_seen"] = frame_idx
        # Chỉ cập nhật "side" khi điểm đủ xa vạch (ngoài vùng đệm) — giữ nguyên phía cũ đã ghi
        # nhớ nếu vị trí mới còn mập mờ sát vạch (hysteresis), tránh báo vượt vạch giả.
        if new_side_confident: tr["side"] = new_side

        # Track đang "chờ xác minh rẽ": mỗi lần có detection mới, kiểm tra đã đủ căn cứ RẼ HỢP
        # LỆ hay VI PHẠM chưa (xem resolve_turn_pending()). Giữ 1 ảnh crop mới nhất (ghi đè liên
        # tục CHỈ khi xe còn trong ROI — xem nhánh "ra khỏi ROI" bên dưới) phòng khi track mất
        # dấu giữa chừng, cần đẩy vào hàng đợi rà soát thủ công.
        if tr.get("pending_turn") is not None:
            still_in_roi = (not use_roi) or bbox_in_zone([x1,y1,x2,y2], roi_pts)
            if still_in_roi:
                _pc = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if _pc.size > 0: tr["pending_crop"] = _pc.copy()
            outcome = resolve_turn_pending([x1,y1,x2,y2], tr["pending_turn"], line_pts, turn_confirm_pts,
                                            time.time() - tr["pending_turn"]["since_ts"], TURN_CONFIRM_MAX_SEC)
            if outcome == "wait" and use_roi and not still_in_roi:
                # Xe đã ra khỏi TẤT CẢ vùng ROI đã vẽ mà chưa từng xác nhận được rẽ hợp lệ (chưa
                # vào cổng xác nhận) -> không chờ hết giờ nữa, chốt VI PHẠM ngay.
                # Ảnh chứng cứ dùng crop đã lưu từ lúc CÒN trong ROI (tr["pending_crop"]) — xem
                # khối kích hoạt OCR nền bên dưới.
                outcome = "violation"
                tr["_use_cached_evidence"] = True
            if outcome == "turn":
                tr["pending_turn"] = None; tr["turn_confirmed"] = True; tr.pop("pending_crop", None)
            elif outcome == "violation":
                # Hết cửa sổ xác minh mà vẫn đi thẳng -> xe lợi dụng làn rẽ vượt đèn đỏ. Chốt vi
                # phạm NGAY TẠI ĐÂY (bbox hiện tại) -> khối OCR nền bên dưới chạy y như bình thường.
                tr["pending_turn"] = None
                is_new_vio = True; tr["violated"] = True
                state["violation_count"] += 1

        plate_text = tr.get("plate", "")
        if is_new_vio and not plate_text:
            # Đọc biển số (YOLO + OCR) tốn 0.5-3s+ nếu chạy đồng bộ ngay đây sẽ làm CẢ response
            # box bị chặn theo -> đúng khoảnh khắc quan trọng nhất (xe vượt đèn đỏ) lại bị đứng
            # hình lâu nhất. Nên: trả kết quả box NGAY (tạm coi "chưa đọc được biển"), còn việc
            # đọc biển số + lưu ảnh bằng chứng + lưu DB thì đẩy sang chạy nền ở thread pool riêng.
            # Khi đọc xong, tr["plate"] sẽ được cập nhật -> lần poll kế tiếp tự hiện biển số thật.
            # Trường hợp chốt vi phạm vì "ra khỏi ROI lúc đang xác minh rẽ": dùng ảnh đã lưu từ
            # lúc xe CÒN trong ROI thay vì crop khung hiện tại (lúc này xe đã ở ngoài ROI, có thể
            # đã ra khỏi khung hình hoặc chỉ còn thấy 1 phần).
            if tr.pop("_use_cached_evidence", False) and tr.get("pending_crop") is not None \
                    and tr["pending_crop"].size > 0:
                crop = tr["pending_crop"]
            else:
                crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
            # Bản sao NGUYÊN khung hình (không crop) tại đúng lúc phát hiện vượt vạch — chuyển
            # cho thread nền để vẽ ảnh toàn cảnh (chỉ box riêng xe này) mà không chặn response.
            full_frame_snapshot = img.copy()
            now_s = datetime.now().strftime("%H:%M:%S")
            new_violations.append({
                "track_id": tid, "plate_text": "KHO_DOC_BIEN",
                "vehicle_type": vt, "time_str": now_s,
                "evidence_path": None, "overview_path": None, "cam_id": cam_id,
            })
            try:
                if OCR_BG_ENABLED:
                    asyncio.get_event_loop().run_in_executor(
                        _ocr_bg_executor, _run_plate_ocr_bg,
                        crop, tid, vt, user["username"], tracks, conf_plate, ocr_confidence,
                        full_frame_snapshot, [x1, y1, x2, y2])
            except Exception as _e:
                print(f"[stream-frame] Không khởi chạy được OCR nền: {_e}")

        detections.append({
            "bbox": [x1, y1, x2, y2], "track_id": tid, "type": vt,
            "confidence": round(det["conf"], 3),
            "is_violation": tr["violated"],
            "is_new_violation": is_new_vio,
            "is_pending_turn": tr.get("pending_turn") is not None,
            "plate_text": plate_text,
            "overview_path": tr.get("overview_path"),
            "evidence_path": tr.get("evidence_path"),
        })

    stale = [tid for tid, tr in tracks.items()
             if frame_idx - tr["last_seen"] > STREAM_TRACK_MAX_MISS]
    for tid in stale:
        str_tr = tracks[tid]
        if str_tr.get("pending_turn") is not None:
            _pc = str_tr.get("pending_crop")
            if _pc is not None and _pc.size > 0:
                # Mất dấu (che khuất, ra khỏi khung hình...) TRONG lúc đang xác minh rẽ, CHƯA
                # từng xác nhận được rẽ hợp lệ -> coi như "ra khỏi vùng theo dõi mà chưa chứng
                # minh được đang rẽ", chốt VI PHẠM luôn thay vì lặng lẽ đẩy vào hàng đợi rà soát —
                # dùng đúng khung LÚC CÒN THẤY xe rõ nhất (không phải khung hiện tại, vì xe đã
                # biến mất khỏi khung hình). Đọc biển số chạy nền như 1 vi phạm bình thường.
                state["violation_count"] += 1
                try:
                    if OCR_BG_ENABLED:
                        asyncio.get_event_loop().run_in_executor(
                            _ocr_bg_executor, _run_plate_ocr_bg,
                            _pc, tid, str_tr.get("type"), user["username"], tracks, conf_plate, ocr_confidence,
                            None, None)
                    else:
                        save_violation_to_db("", str_tr.get("type"), track_id=tid, created_by=user["username"])
                except Exception as _e:
                    print(f"[stream-frame] Không khởi chạy được OCR nền cho track mất dấu lúc pending: {_e}")
            else:
                # Hiếm gặp: mất dấu ngay khung đầu tiên, CHƯA kịp lưu được ảnh nào cả -> không có
                # gì làm bằng chứng, đẩy cán bộ xem lại thay vì phạt không ảnh.
                push_review_case(str_tr.get("type"), cam_id=cam_id, track_id=tid,
                                  snapshot_path=None, created_by=user["username"])
        del tracks[tid]

    violations = [d for d in detections if d["is_violation"]]
    _ms = (time.time() - _t0) * 1000
    print(f"[stream-frame:{cam_id}] xử lý {_ms:.0f}ms | {len(detections)} xe"
          + (f" | ⚠️ CHẬM (>500ms) — đây là nguyên nhân chính khiến box bị trễ" if _ms > 500 else ""))
    return {
        "cam_id": cam_id, "frame_idx": frame_idx,
        "detections": detections, "violations": violations,
        "new_violations": new_violations,
        "total_vehicles": len(detections), "light": light_status,
        "process_ms": round(_ms),
    }

@app.delete("/stream-frame/{cam_id}/reset")
async def reset_cam_state(cam_id: str, user: dict = Depends(require_role("admin", "truc_camera"))):
    """Xoá tracker state của một camera (khi đổi nguồn video)."""
    if cam_id in cam_states:
        del cam_states[cam_id]
    return {"ok": True, "cam_id": cam_id}




import json as _json2
from typing import Optional

MOCK_DB_FILE = Path("mock_violations.json")
_violation_db = None

def load_violation_db():
    global _violation_db
    if _violation_db is None:
        if MOCK_DB_FILE.exists():
            with open(MOCK_DB_FILE, encoding="utf-8") as f:
                _violation_db = _json2.load(f)
        else:
            _violation_db = []
    return _violation_db

def point_in_quad(px, py, quad):
    """Kiểm tra điểm (px,py) có nằm trong tứ giác (4 điểm [x,y]) bằng ray casting."""
    if len(quad) < 3: return True
    inside = False; n = len(quad); j = n - 1
    for i in range(n):
        xi, yi = quad[i]; xj, yj = quad[j]
        if (yi > py) != (yj > py):
            x_int = (xj - xi) * (py - yi) / ((yj - yi) + 1e-9) + xi
            if px < x_int: inside = not inside
        j = i
    return inside

def bbox_in_zone(bbox, zone_pts):
    """Coi cả bbox (không chỉ đúng 1 tâm điểm) là 'nằm trong' zone_pts nếu BẤT KỲ điểm đại diện
    nào của bbox (tâm, và 2 góc dưới trái/phải — xấp xỉ vệt bánh xe chạm đường) rơi vào vùng.
    Lý do cần kiểm tra nhiều điểm thay vì chỉ tâm bbox: ô tô rộng hơn xe máy 3-4 lần, tâm bbox ô
    tô có thể rơi ra ngoài 1 vùng vẽ vừa khít cho xe máy dù cả 2 xe cùng đi trong 1 làn — khiến
    ô tô bị loại khỏi vùng rẽ/ROI một cách oan uổng trong khi xe máy ở vị trí tương tự thì không.
    Dùng cho: vùng rẽ được phép, cổng xác nhận rẽ, và kiểm tra "còn trong ROI". KHÔNG dùng cho
    logic cắt vạch cốt lõi (side_of_line) — chỗ đó vẫn cần đúng 1 tâm điểm để tính hướng ổn định."""
    if not zone_pts or len(zone_pts) < 3: return False
    x1, y1, x2, y2 = bbox
    for px, py in (((x1+x2)//2, (y1+y2)//2), (x1, y2), (x2, y2), ((x1+x2)//2, y2)):
        if point_in_quad(px, py, zone_pts): return True
    return False

# Ngưỡng khoảng cách tối thiểu (pixel) tính từ tâm xe tới đường vạch dừng để coi là
# "đã ở hẳn 1 phía" đáng tin cậy. Nhỏ hơn ngưỡng này -> coi là "sát vạch/mập mờ", KHÔNG dùng
# để cập nhật phía đã ghi nhớ của track lẫn để kết luận vượt vạch. Đây là "vùng đệm" (dead-zone)
# quanh vạch nhằm chặn báo vượt vạch giả khi xe ĐỨNG YÊN sát vạch nhưng bbox rung nhẹ vài pixel
# giữa 2 lần detect (rất hay gặp khi xe xếp hàng chờ đèn đỏ ngay trước vạch).
MIN_LINE_CROSS_DIST_PX = 8

def side_of_line(px, py, line):
    """line=[(x1,y1),(x2,y2)]. Trả về khoảng cách vuông góc CÓ DẤU (đơn vị pixel, không phải
    tích chéo thô) từ điểm (px,py) tới đường thẳng qua line: dấu (+/-) cho biết phía, độ lớn
    cho biết điểm cách vạch bao xa -> dùng được để so với MIN_LINE_CROSS_DIST_PX bên dưới."""
    if len(line) < 2: return 0
    (x1,y1),(x2,y2) = line
    cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
    length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5 or 1
    return cross / length

def line_crosses_bbox(line, bbox):
    """True nếu đoạn vạch dừng 'line' cắt qua bbox xe — dùng cho ảnh tĩnh
    (không có tracking nên không xác định được hướng di chuyển)."""
    if len(line) < 2: return False
    (lx1,ly1),(lx2,ly2) = line
    x1,y1,x2,y2 = bbox
    corners = [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
    sides = [(lx2-lx1)*(cy-ly1)-(ly2-ly1)*(cx-lx1) for cx,cy in corners]
    if not (any(s>0 for s in sides) and any(s<0 for s in sides)): return False
    lxmin,lxmax = min(lx1,lx2),max(lx1,lx2); lymin,lymax = min(ly1,ly2),max(ly1,ly2)
    return not (x2<lxmin-50 or x1>lxmax+50 or y2<lymin-50 or y1>lymax+50)

def violation_side_sign(roi_pts, line):
    """Vạch dừng chia tứ giác ROI thành 2 vùng. Vùng có ÍT đỉnh ROI hơn
    là vùng nhỏ hơn (thường nằm gần vạch kẻ đường/giao lộ) => vùng vi phạm.
    Trả về dấu (sign) đại diện cho phía vi phạm, hoặc None nếu vạch
    không cắt qua ROI (hoặc thiếu dữ liệu)."""
    if len(line)<2 or len(roi_pts)<3: return None
    sides=[side_of_line(px,py,line) for px,py in roi_pts]
    pos=sum(1 for s in sides if s>0); neg=sum(1 for s in sides if s<0)
    if pos==0 or neg==0: return None
    return 1 if pos<neg else -1

# ════════════════════════════════════════════════════════════════════════════
# NÚT GIAO CHO PHÉP RẼ KHI ĐÈN ĐỎ — vùng rẽ được phép + cổng xác nhận rẽ
# ════════════════════════════════════════════════════════════════════════════
# Vấn đề: bắt lỗi bằng "cắt vạch khi đèn đỏ" khiến xe rẽ hợp lệ (ở nút giao cho phép rẽ phải/
# trái khi đèn đỏ) bị báo sai, nhưng nếu bỏ qua hẳn làn đó thì lại lọt xe lợi dụng làn rẽ để đi
# thẳng vượt đèn đỏ — vì ngay tại thời điểm chạm vạch, KHÔNG THỂ biết xe sắp rẽ hay đi thẳng.
# Cách giải: không quyết định ngay tại khung hình cắt vạch. Nếu điểm cắt nằm trong "vùng rẽ
# được phép" (polygon do cán bộ trực camera vẽ, sát vạch, phía làn rẽ) thì chuyển track sang
# trạng thái "đang xác minh" (tr["pending_turn"]), tiếp tục theo dõi quỹ đạo tối đa
# TURN_CONFIRM_MAX_SEC giây rồi mới chốt: chỉ vào Cổng xác nhận rẽ (vùng cam) mới được coi là
# rẽ hợp lệ; nếu không vào cổng trước khi ra khỏi ROI/hết giờ thì chốt vi phạm.
# Video offline (_run_video_job) quy giây này ra SỐ KHUNG HÌNH theo fps thật của video (cố định,
# biết trước). /stream-frame (giám sát trực tiếp) đo bằng THỜI GIAN THỰC (time.time()) vì tốc độ
# poll không cố định — KHÔNG được coi 1 lần gọi /stream-frame ~ 1 giây. Camera nào KHÔNG cấu
# hình vùng rẽ thì giữ nguyên hành vi cũ (phạt ngay khi cắt vạch lúc đèn đỏ), không ảnh hưởng gì.
#
# THÊM 1 ĐƯỜNG CHỐT SỚM (ngoài resolve_turn_pending): nếu track đang "đang xác minh" mà ĐI RA
# KHỎI vùng ROI trước khi kịp vào cổng xác nhận, KHÔNG đợi hết TURN_CONFIRM_MAX_SEC
# nữa — chốt VI PHẠM ngay lúc đó (xử lý ở _run_video_job/stream_frame, không phải trong
# resolve_turn_pending, vì cần biết roi_pts). Lý do: 1 xe ra hẳn khỏi ROI mà chưa từng chứng minh
# được đang rẽ gần như chắc chắn đã/sắp đi thẳng ra khỏi giao lộ — không có lý do để đợi thêm.
# Ảnh chứng cứ dùng crop đã lưu từ LÚC CÒN TRONG ROI (tr["pending_crop"]), không dùng khung hiện
# tại vì xe đã ra ngoài ROI, có thể đã ra khỏi khung hình hoặc chỉ còn thấy 1 phần. LƯU Ý: quy tắc
# này giả định Cổng xác nhận rẽ (và toàn bộ cung đường rẽ) nằm HẲN BÊN TRONG ROI, có khoảng đệm dư
# dả — nếu ROI vẽ sát ngay mép cổng xác nhận, xe rẽ thật có thể bị chốt nhầm ngay trước khi kịp
# chạm cổng do ra khỏi ROI sớm hơn 1 nhịp; vẽ ROI rộng rãi bao trùm hết khu vực rẽ để tránh việc này.
TURN_CONFIRM_MAX_SEC = float(os.getenv("TURN_CONFIRM_MAX_SEC", "3.0"))  # cửa sổ tối đa để 1 xe
    # "hoàn tất" việc rẽ trước khi bị coi là cố tình đi thẳng — tuỳ hình học giao lộ (giao lộ
    # rộng/rẽ chéo cần nới thêm), chỉnh qua biến môi trường, không sửa số cứng ở đây.
TURN_STRAIGHT_MIN_MOVE_PX = float(os.getenv("TURN_STRAIGHT_MIN_MOVE_PX", "24"))
TURN_STRAIGHT_MAX_ANGLE_DEG = float(os.getenv("TURN_STRAIGHT_MAX_ANGLE_DEG", "20"))

def straight_direction_vector(line_pts, vio_sign):
    """Hướng đi thẳng qua vạch dừng, suy ra từ pháp tuyến của vạch và phía vi phạm."""
    if len(line_pts) < 2 or vio_sign is None: return None
    (x1,y1),(x2,y2) = line_pts
    lx, ly = x2-x1, y2-y1
    length = (lx*lx+ly*ly) ** 0.5 or 1
    gx, gy = -ly/length, lx/length
    if vio_sign < 0: gx, gy = -gx, -gy
    return gx, gy

def angle_between_deg(vx, vy, ux, uy):
    nv = (vx*vx+vy*vy) ** 0.5; nu = (ux*ux+uy*uy) ** 0.5
    if nv < 1e-6 or nu < 1e-6: return 0.0
    cos_a = max(-1.0, min(1.0, (vx*ux+vy*uy)/(nv*nu)))
    return float(np.degrees(np.arccos(cos_a)))

def resolve_turn_pending(bbox, pending, line_pts, turn_confirm_pts, elapsed, max_elapsed):
    """Xét 1 track đang ở trạng thái 'đang xác minh rẽ' (đã cắt vạch trong vùng rẽ được phép
    lúc đèn đỏ). "elapsed"/"max_elapsed" do NGƯỜI GỌI tính sẵn theo đơn vị PHÙ HỢP với nguồn dữ
    liệu (số khung hình thật cho video offline có fps cố định, hoặc GIÂY THỰC cho giám sát trực
    tiếp — nơi tốc độ poll không cố định, KHÔNG được coi 1 lần poll ~ 1 giây). Trả về:
      - "turn"      : xe đã vào Cổng xác nhận rẽ (vùng cam) -> RẼ HỢP LỆ, không phạt.
      - "violation" : hết cửa sổ max_elapsed mà vẫn chưa có bằng chứng rẽ ở trên -> xe lợi dụng
                      làn rẽ để đi thẳng vượt đèn đỏ -> chốt VI PHẠM.
      - "wait"      : chưa đủ căn cứ, tiếp tục theo dõi thêm."""
    cx, cy = (bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2
    if turn_confirm_pts and len(turn_confirm_pts) >= 3 and bbox_in_zone(bbox, turn_confirm_pts):
        return "turn"
    # Chỉ vùng cam xác nhận xe rẽ. Quỹ đạo đi thẳng đủ rõ thì chốt sớm, không đợi xe mất khỏi ROI.
    sx, sy = pending["start_cx"], pending["start_cy"]
    dx, dy = cx - sx, cy - sy
    moved = (dx*dx + dy*dy) ** 0.5
    straight = straight_direction_vector(line_pts, pending.get("vio_sign"))
    if moved >= TURN_STRAIGHT_MIN_MOVE_PX and straight is not None:
        if angle_between_deg(dx, dy, straight[0], straight[1]) <= TURN_STRAIGHT_MAX_ANGLE_DEG:
            return "violation"
    if elapsed >= max_elapsed:
        return "violation"
    return "wait"

def bbox_iou(b1, b2):
    x1,y1,x2,y2 = b1; X1,Y1,X2,Y2 = b2
    ix1,iy1 = max(x1,X1),max(y1,Y1); ix2,iy2 = min(x2,X2),min(y2,Y2)
    iw,ih = max(0,ix2-ix1),max(0,iy2-iy1)
    inter = iw*ih
    union = (x2-x1)*(y2-y1) + (X2-X1)*(Y2-Y1) - inter
    return inter/union if union>0 else 0

def assign_tracks_global(cur_dets, tracks, score_fn, min_score):
    """Khớp det<->track theo kiểu 'điểm cao nhất trước' (global best-first), THAY cho greedy
    cũ vốn duyệt cur_dets theo đúng thứ tự YOLO trả về rồi chọn track tốt nhất còn trống cho
    TỪNG det một (order-dependent). Lỗi của cách cũ: khi 2 xe đi gần/cắt nhau (VD ô tô + xe
    máy lấn làn), thứ tự duyệt quyết định ai được ghép với track nào — có thể ghép NHẦM
    (det của xe A lại khớp với track của xe B) nếu A được xử lý trước và "cướp" mất track
    đúng ra thuộc về B. Khi bị ghép nhầm, phía client đột ngột nhận bbox của xe khác dưới
    CÙNG 1 track_id -> box "nhảy" kích thước/vị trí trong đúng 1 khung hình, nhìn giống hệt
    hiện tượng phóng to/thu nhỏ hoặc giật bất thường mà không phải do làm mượt phía client.
    Cách mới: tính điểm mọi cặp (det, track) hợp lệ, xếp giảm dần theo điểm, rồi gán tuần tự
    từ điểm cao nhất (bỏ qua det/track đã được gán) — giảm mạnh khả năng ghép nhầm so với
    greedy theo thứ tự cũ (dù vẫn chưa phải Hungarian tối ưu tuyệt đối, nhưng đủ tốt và rẻ
    hơn nhiều so với triển khai thuật toán ghép tối ưu đầy đủ).
    Trả về: det_to_track (dict di -> tid), chỉ gồm các track ĐÃ TỒN TẠI khớp được;
    caller tự tạo track mới cho các det còn lại (không có trong dict trả về)."""
    candidates = []  # (score, di, tid)
    for di, det in enumerate(cur_dets):
        for tid, tr in tracks.items():
            s = score_fn(det["bbox"], tr["bbox"], det.get("type"), tr.get("type"))
            if s <= min_score: continue
            # Rào chắn quan trọng: nếu track này ĐÃ bị đánh dấu vi phạm (violated=True), chỉ cho
            # phép 1 detection tiếp tục "thừa kế" track đó (và cả trạng thái violated) khi có
            # chồng lấp bbox THẬT SỰ (IoU>0) — tức có bằng chứng hình ảnh trực tiếp là cùng 1 xe.
            # Không được ghép chỉ vì khoảng cách gần (dist_score), vì trong cảnh xe đông/xếp hàng
            # sát nhau, ghép-theo-khoảng-cách rất dễ nhận NHẦM 1 xe khác (chưa hề vi phạm) vào
            # đúng track đã vi phạm này -> xe vô tội bị gắn "VI PHAM!" ngay dù đứng yên.
            if tr.get("violated") and bbox_iou(det["bbox"], tr["bbox"]) <= 0:
                continue
            candidates.append((s, di, tid))
    candidates.sort(key=lambda c: c[0], reverse=True)  # điểm cao nhất trước

    det_to_track = {}
    used_tids = set()
    used_dis = set()
    for s, di, tid in candidates:
        if di in used_dis or tid in used_tids: continue
        det_to_track[di] = tid
        used_dis.add(di); used_tids.add(tid)
    return det_to_track

def match_score(det_bbox, tr_bbox, det_type=None, tr_type=None):
    """Kết hợp IoU và khoảng cách tâm (chuẩn hóa theo đường chéo bbox)
    để khớp xe giữa các frame cách nhau (frame_skip), khi xe di chuyển
    nhiều khiến IoU thấp/0. Bán kính chấp nhận nới x1.8 so với bản gốc:
    xe máy nhỏ + luồn lách nhanh nên trong đúng frame_skip khung hình
    (vài chục-trăm ms) rất dễ di chuyển vượt quá 1 lần đường chéo bbox
    của chính nó -> match_score gốc tụt về 0 -> bị tách thành track_id
    mới giữa chừng -> để lại "tàn ảnh" (track cũ bị bỏ rơi vẫn được vẽ
    tới khi hết hạn) và có nguy cơ 1 lượt vượt đèn bị ghi thành 2 vi phạm
    dưới 2 track_id khác nhau. Vẫn nhẹ tay hơn match_score_stream (x3,
    dùng cho gap ~1s bên /stream-frame) vì gap ở đây chỉ vài trăm ms."""
    iou = bbox_iou(det_bbox, tr_bbox)
    bw=tr_bbox[2]-tr_bbox[0]; bh=tr_bbox[3]-tr_bbox[1]
    diag=(bw*bw+bh*bh)**0.5 or 1
    tcx=(tr_bbox[0]+tr_bbox[2])/2; tcy=(tr_bbox[1]+tr_bbox[3])/2
    dcx=(det_bbox[0]+det_bbox[2])/2; dcy=(det_bbox[1]+det_bbox[3])/2
    dist=((dcx-tcx)**2+(dcy-tcy)**2)**0.5
    dist_score=max(0, 1-dist/(diag*1.8))
    score = max(iou, dist_score)
    if det_type and tr_type and det_type != tr_type:
        score *= 0.6  # khác loại xe (vd ô tô/xe máy) đứng gần nhau -> giảm điểm, tránh ghép nhầm
    return score

def match_score_stream(det_bbox, tr_bbox, det_type=None, tr_type=None):
    """Giống match_score nhưng khoan dung khoảng cách hơn NHIỀU LẦN — dành riêng cho
    /stream-frame (giám sát đa cam), nơi backend chỉ nhận được 1 khung hình mỗi ~1 giây
    từ trình duyệt (khoảng cách thời gian thực giữa 2 lần match lớn hơn 6-10 lần so với
    xử lý video hàng loạt vốn cách nhau chỉ vài chục-trăm ms). Với gap lớn như vậy, xe
    (đặc biệt xe máy) hoàn toàn có thể di chuyển xa hơn cả kích thước bbox của chính nó,
    khiến match_score gốc luôn ~0 -> mỗi lần poll xe bị coi là "xe mới" -> track_id đổi
    liên tục -> mất khả năng nội suy mượt phía client + báo trùng vi phạm nhiều lần.

    LƯU Ý: từng thử siết hẳn bán kính này (x3.0 -> x1.5) để chặn ghép nhầm track đã
    violated=True sang xe khác, nhưng gây hại nhiều hơn lợi — xe máy đi nhanh ĐÚNG LÚC
    vượt vạch (giữa 2 lần poll cách nhau ~1s) bị mất track giữa chừng, sinh track mới đã
    nằm sẵn ở phía "trong" -> không còn "phía cũ" để so sánh -> vượt đèn đỏ thật lại
    KHÔNG được ghi nhận (hiện xanh dù đã qua vạch). Bán kính rộng vẫn cần thiết cho toàn
    bộ luồng chung; việc chặn ghép nhầm vào track đã violated được xử lý riêng, đúng chỗ
    hơn, ở assign_tracks_global (bắt buộc IoU>0 mới cho track violated tiếp tục ID)."""
    iou = bbox_iou(det_bbox, tr_bbox)
    bw=tr_bbox[2]-tr_bbox[0]; bh=tr_bbox[3]-tr_bbox[1]
    diag=(bw*bw+bh*bh)**0.5 or 1
    tcx=(tr_bbox[0]+tr_bbox[2])/2; tcy=(tr_bbox[1]+tr_bbox[3])/2
    dcx=(det_bbox[0]+det_bbox[2])/2; dcy=(det_bbox[1]+det_bbox[3])/2
    dist=((dcx-tcx)**2+(dcy-tcy)**2)**0.5
    dist_score=max(0, 1-dist/(diag*3.0))  # bán kính chấp nhận ~gấp 3 lần match_score gốc
    score = max(iou, dist_score)
    if det_type and tr_type and det_type != tr_type:
        score *= 0.6  # khác loại xe -> giảm điểm để tránh nhầm xe máy/ô tô đứng gần nhau
    return score

def parse_vio_time(r) -> datetime:
    """Chuyển 'thoi_gian_vi_pham' (chuỗi "HH:MM DD/MM/YYYY") thành datetime THẬT để sắp
    xếp đúng theo thời gian thực tế. LƯU Ý: TRƯỚC ĐÂY code sort() thẳng trên chuỗi string
    này -> so sánh ký tự đầu tiên trước (tức giờ:phút "HH:MM" đứng trước ngày/tháng/năm
    trong chuỗi) -> kết quả bị sắp theo GIỜ TRONG NGÀY là chính, không phải theo ngày thực,
    khiến 1 vi phạm cũ (vd 23:19 01/07/2026) có thể bị xếp TRÊN 1 vi phạm mới hơn nhiều
    (vd 09:15 16/09/2026) chỉ vì "23" > "09" khi so string. Dùng hàm này làm sort key để
    sắp xếp đúng theo thời điểm thực."""
    try:
        return datetime.strptime(r.get("thoi_gian_vi_pham", ""), "%H:%M %d/%m/%Y")
    except (ValueError, TypeError):
        return datetime.min

def normalize_plate(plate: str) -> str:
    """Chuẩn hóa biển số để so sánh: bỏ dấu '-', space, uppercase."""
    return plate.upper().replace("-","").replace(" ","").strip()

# Fallback: khi OCR không đọc được biển (KHO_DOC_BIEN), tra theo vị trí xe
# (cx,cy chuẩn hóa theo % chiều rộng/cao ảnh) + loại xe -> biển số đã biết
FALLBACK_PLATES = [
    # Xe ô tô sedan giữa khung -> 29D-07184
    {"type": "O to", "cx": (0.30, 0.60), "cy": (0.20, 0.55), "plate": "29D-07184"},
    # Xe máy góc dưới phải -> 15-K1 21837
    {"type": "Xe may", "cx": (0.55, 0.95), "cy": (0.45, 0.95), "plate": "15-K1 21837"},
]

def get_fallback_plate(vehicle_type: str, cx: int, cy: int, w_img: int, h_img: int):
    """Tra bảng fallback theo vị trí tương đối + loại xe khi OCR fail."""
    if w_img <= 0 or h_img <= 0: return None
    cxn, cyn = cx / w_img, cy / h_img
    for fp in FALLBACK_PLATES:
        if fp["type"] != vehicle_type: continue
        x_lo, x_hi = fp["cx"]; y_lo, y_hi = fp["cy"]
        if x_lo <= cxn <= x_hi and y_lo <= cyn <= y_hi:
            return fp["plate"]
    return None

DON_VI_MAP = {
    "29": {"ten": "Phòng CSGT – Công an TP. Hà Nội", "dia_chi": "75 Phùng Hưng, Hoàn Kiếm, Hà Nội"},
    "51": {"ten": "Đội CHGT & ĐK Đèn THGT – Phòng CSGT Công an Tp Hồ Chí Minh", "dia_chi": "341 Trần Hưng Đạo, Phường Cầu Ông Lãnh, TP. Hồ Chí Minh"},
    "61": {"ten": "Phòng CSGT – CAT Bình Dương", "dia_chi": "Đường Thích Quảng Đức, Phú Cường, Bình Dương"},
    "60": {"ten": "Trạm CSGT Dầu Giây – Phòng CSGT – CAT Đồng Nai", "dia_chi": "Quốc lộ 1A, Dầu Giây, Đồng Nai"},
    "43": {"ten": "Đội CSGT số 3 – Phòng CSGT – CAT Đà Nẵng", "dia_chi": "99 Điện Biên Phủ, Thanh Khê, Đà Nẵng"},
    "15": {"ten": "Phòng CSGT – CAT Hải Phòng", "dia_chi": "Lê Đại Hành, Hồng Bàng, Hải Phòng"},
}

def save_violation_to_db(plate_text: str, vehicle_type: str, evidence_path: str = None,
                          overview_path: str = None,
                          video_time_s: float = None, track_id=None, created_by: str = "system"):
    """Lưu vi phạm vượt đèn đỏ vào mock_violations.json.
    Nếu KHÔNG đọc được biển số, vẫn lưu lại đối tượng vi phạm (không bỏ sót),
    gắn mã tạm duy nhất + đánh dấu bien_so_doc_duoc=False, kèm thời điểm vi phạm
    và ảnh minh chứng để tra cứu/điền biển số thủ công sau."""
    db = load_violation_db()
    doc_duoc = bool(plate_text) and plate_text not in ("KHO_DOC_BIEN", "???", "")

    if doc_duoc:
        # Không lưu trùng (cùng biển + chưa xử phạt)
        for r in db:
            if normalize_plate(r.get("bien_so","")) == normalize_plate(plate_text) and r.get("trang_thai") == "Chưa xử phạt":
                return None
        bien_so_luu = plate_text
        ma_tinh = plate_text[:2] if plate_text[:2].isdigit() else "51"
    else:
        # Không đọc được biển -> vẫn lưu, gắn mã tạm để không trùng với object khác
        bien_so_luu = f"CHUA_XAC_DINH-{uuid.uuid4().hex[:6].upper()}"
        ma_tinh = "51"

    don_vi_gq = DON_VI_MAP.get(ma_tinh, DON_VI_MAP["51"])
    record = {
        "bien_so": bien_so_luu,
        "bien_so_doc_duoc": doc_duoc,
        "loai_phuong_tien": vehicle_type if vehicle_type else "Ô tô con",
        "thoi_gian_vi_pham": datetime.now().strftime("%H:%M %d/%m/%Y"),
        "thoi_diem_trong_video_s": round(video_time_s, 1) if video_time_s is not None else None,
        "track_id": track_id,
        "dia_diem_vi_pham": "Ghi nhận qua camera AI",
        "hanh_vi_vi_pham": "16824.5.1.a Vượt đèn đỏ, đèn vàng",
        "muc_phat": "4.000.000 - 6.000.000 đồng" if vehicle_type not in ("Xe may",) else "1.000.000 - 2.000.000 đồng",
        "trang_thai": "Chưa xử phạt",
        "don_vi_phat_hien": don_vi_gq["ten"],
        "don_vi_giai_quyet": [don_vi_gq],
        "anh_minh_chung": evidence_path,   # ảnh cận cảnh (crop riêng xe vi phạm)
        "anh_toan_canh": overview_path,    # ảnh toàn cảnh giao lộ lúc vi phạm, chỉ box xe này
        "ghi_nhan_boi": created_by,
    }
    db.append(record)
    with open(MOCK_DB_FILE, "w", encoding="utf-8") as f:
        _json2.dump(db, f, ensure_ascii=False, indent=2)
    write_audit({"username": created_by, "role": "system"}, "GHI_NHAN_VI_PHAM", target=bien_so_luu,
                detail=f"Loại xe: {vehicle_type}, đọc được biển: {doc_duoc}")
    return record

# ════════════════════════════════════════════════════════════════════════════
# HÀNG ĐỢI "CẦN RÀ SOÁT THỦ CÔNG" — dành cho track đang ở trạng thái "đang xác minh rẽ"
# (đã cắt vạch trong Vùng rẽ được phép lúc đèn đỏ) nhưng bị MẤT DẤU giữa chừng (che khuất,
# đổi track_id...) trước khi có đủ bằng chứng kết luận rẽ hợp lệ hay vi phạm. KHÔNG tự động
# phạt (tránh oan xe đang rẽ đúng nhưng bị che khuất) và KHÔNG tự động miễn (tránh lọt xe cố
# tình đánh võng) — đẩy cho cán bộ trực camera/xử phạt tự xem ảnh rồi quyết định.
# ════════════════════════════════════════════════════════════════════════════
REVIEW_QUEUE_FILE = Path("mock_review_queue.json")
_review_queue = None

def load_review_queue():
    global _review_queue
    if _review_queue is None:
        if REVIEW_QUEUE_FILE.exists():
            with open(REVIEW_QUEUE_FILE, encoding="utf-8") as f:
                _review_queue = _json2.load(f)
        else:
            _review_queue = []
    return _review_queue

def _save_review_queue():
    with open(REVIEW_QUEUE_FILE, "w", encoding="utf-8") as f:
        _json2.dump(_review_queue, f, ensure_ascii=False, indent=2)

def push_review_case(vehicle_type, cam_id=None, track_id=None, video_time_s=None,
                      snapshot_path=None, ly_do="Mất dấu khi đang xác minh rẽ",
                      created_by="system"):
    q = load_review_queue()
    case = {
        "id": uuid.uuid4().hex[:10],
        "loai_phuong_tien": vehicle_type or "Khac",
        "cam_id": cam_id,
        "track_id": track_id,
        "thoi_diem_trong_video_s": round(video_time_s, 1) if video_time_s is not None else None,
        "thoi_gian_ghi_nhan": datetime.now().strftime("%H:%M %d/%m/%Y"),
        "ly_do": ly_do,
        "anh_snapshot": snapshot_path,
        "trang_thai": "Chưa xem",   # Chưa xem | Xác nhận vi phạm | Loại bỏ (rẽ hợp lệ)
        "nguoi_xu_ly": None,
        "ghi_chu": "",
        "ghi_nhan_boi": created_by,
    }
    q.append(case)
    _save_review_queue()
    return case

@app.get("/review-queue")
async def list_review_queue(trang_thai: Optional[str] = None, user: dict = Depends(get_current_user)):
    """Xem hàng đợi các track 'cần rà soát thủ công' — rẽ-khi-đèn-đỏ bị mất dấu giữa chừng
    trong lúc đang xác minh, chưa thể tự động kết luận."""
    q = load_review_queue()
    if trang_thai and trang_thai != "all":
        q = [r for r in q if r.get("trang_thai") == trang_thai]
    q = sorted(q, key=lambda r: r.get("thoi_gian_ghi_nhan", ""), reverse=True)
    return {"total": len(q), "cases": q}

class ReviewResolveBody(BaseModel):
    quyet_dinh: str             # "vi_pham" | "loai_bo"
    ghi_chu: Optional[str] = ""

@app.patch("/review-queue/{case_id}")
async def resolve_review_case(case_id: str, body: ReviewResolveBody,
                               user: dict = Depends(require_role("admin", "truc_camera", "xu_phat"))):
    """Cán bộ xem ảnh rồi tự quyết định: xác nhận vi phạm (đẩy sang hồ sơ vi phạm chính thức,
    xử phạt như bình thường) hoặc loại bỏ (coi là rẽ hợp lệ, không lưu vi phạm)."""
    q = load_review_queue()
    found = next((r for r in q if r["id"] == case_id), None)
    if not found:
        raise HTTPException(404, "Không tìm thấy hồ sơ cần rà soát")
    if found["trang_thai"] != "Chưa xem":
        raise HTTPException(400, "Hồ sơ này đã được xử lý trước đó")
    if body.quyet_dinh not in ("vi_pham", "loai_bo"):
        raise HTTPException(400, "quyet_dinh phải là 'vi_pham' hoặc 'loai_bo'")
    if body.quyet_dinh == "vi_pham":
        found["trang_thai"] = "Xác nhận vi phạm"
        save_violation_to_db(
            "", found.get("loai_phuong_tien"),
            evidence_path=found.get("anh_snapshot"), overview_path=found.get("anh_snapshot"),
            video_time_s=found.get("thoi_diem_trong_video_s"), track_id=found.get("track_id"),
            created_by=user["username"])
    else:
        found["trang_thai"] = "Loại bỏ (rẽ hợp lệ)"
    found["nguoi_xu_ly"] = user["username"]
    found["ghi_chu"] = body.ghi_chu or ""
    _save_review_queue()
    write_audit(user, "XU_LY_HANG_DOI_RA_SOAT", target=case_id, detail=body.quyet_dinh)
    return {"ok": True}

@app.get("/lookup-plate")
async def lookup_plate(bien_so: str, loai_phuong_tien: Optional[str] = None):
    """
    Tra cứu phạt nguội theo biển số.
    Trả về danh sách vi phạm từ mock_violations.json.
    """
    db = load_violation_db()
    q = normalize_plate(bien_so)
    results = [
        r for r in db
        if normalize_plate(r.get("bien_so","")) == q
        and (loai_phuong_tien is None or r.get("loai_phuong_tien","") == loai_phuong_tien)
    ]
    return {
        "bien_so":      bien_so.upper(),
        "total":        len(results),
        "violations":   results,
        "chua_xu_phat": sum(1 for r in results if r.get("trang_thai") == "Chưa xử phạt"),
        "da_xu_phat":   sum(1 for r in results if r.get("trang_thai") == "Đã xử phạt"),
    }

@app.get("/violations/chua-doc-duoc")
async def violations_chua_doc_duoc(user: dict = Depends(get_current_user)):
    """
    Liệt kê các đối tượng vi phạm ĐÃ GHI NHẬN nhưng CHƯA đọc được biển số,
    kèm thời gian vi phạm (thoi_gian_vi_pham, thoi_diem_trong_video_s) và
    ảnh minh chứng (anh_minh_chung) để tra/điền biển số thủ công.
    """
    db = load_violation_db()
    results = [r for r in db if not r.get("bien_so_doc_duoc", True)]
    results.sort(key=parse_vio_time, reverse=True)
    return {"total": len(results), "violations": results}

@app.get("/violations")
async def list_all_violations(user: dict = Depends(get_current_user)):
    """Toàn bộ hồ sơ vi phạm — chỉ người đã đăng nhập mới xem được."""
    db = load_violation_db()
    db.sort(key=parse_vio_time, reverse=True)
    return {"total": len(db), "violations": db}

@app.get("/violations/stats")
async def violation_stats(user: dict = Depends(get_current_user)):
    """Thống kê nhanh: hôm nay, giờ này, tổng chưa xử phạt, theo giờ trong ngày."""
    db = load_violation_db()
    now = datetime.now()
    today_str  = now.strftime("%d/%m/%Y")
    hour_str   = now.strftime("%H")          # "14"
    week_start = now.replace(hour=0,minute=0,second=0,microsecond=0)

    today_list = []; hour_list = []; by_hour = {}; by_cam = {}; by_type = {}
    chua_xu_phat = 0; da_xu_phat = 0

    for r in db:
        tgvp = r.get("thoi_gian_vi_pham","")  # "HH:MM DD/MM/YYYY"
        # Tách giờ và ngày từ chuỗi "HH:MM DD/MM/YYYY"
        try:
            parts = tgvp.split(" ")
            r_date = parts[1] if len(parts)>1 else ""
            r_hour = parts[0][:2] if parts[0] else ""
        except: r_date = ""; r_hour = ""

        is_today = (r_date == today_str)
        is_this_hour = is_today and (r_hour == hour_str)

        if is_today: today_list.append(r)
        if is_this_hour: hour_list.append(r)

        # Thống kê theo giờ trong ngày (hôm nay)
        if is_today and r_hour:
            by_hour[r_hour] = by_hour.get(r_hour, 0) + 1

        # Thống kê theo camera
        cam = r.get("cam_id") or r.get("don_vi_phat_hien","Không rõ")
        by_cam[cam] = by_cam.get(cam, 0) + 1

        # Thống kê theo loại xe
        vtype = r.get("loai_phuong_tien","Khác")
        by_type[vtype] = by_type.get(vtype, 0) + 1

        if r.get("trang_thai") == "Đã xử phạt": da_xu_phat += 1
        else: chua_xu_phat += 1

    # Phân bố theo giờ đủ 24h (điền 0 cho giờ không có vi phạm)
    hourly_chart = [{"hour": f"{h:02d}h", "count": by_hour.get(f"{h:02d}", 0)} for h in range(24)]

    return {
        "today":          len(today_list),
        "this_hour":      len(hour_list),
        "total":          len(db),
        "chua_xu_phat":   chua_xu_phat,
        "da_xu_phat":     da_xu_phat,
        "hourly_chart":   hourly_chart,
        "by_cam":         sorted(by_cam.items(), key=lambda x:-x[1]),
        "by_type":        sorted(by_type.items(), key=lambda x:-x[1]),
    }

@app.get("/violations/filter")
async def filter_violations(
    tu_ngay: Optional[str] = None,   # DD/MM/YYYY
    den_ngay: Optional[str] = None,  # DD/MM/YYYY
    tu_gio: Optional[str] = None,    # HH:MM
    den_gio: Optional[str] = None,   # HH:MM
    cam_id: Optional[str] = None,    # gcam1 | tất cả
    bien_so: Optional[str] = None,   # biển số tìm kiếm (partial match)
    trang_thai: Optional[str] = None,# Chưa xử phạt | Đã xử phạt | tất cả
    page: int = 1,
    limit: int = 50,
    user: dict = Depends(get_current_user),
):
    """Lọc vi phạm nâng cao: thời gian, camera, biển số, trạng thái."""
    db = load_violation_db()

    def parse_dt(date_str: str, time_str: str):
        """Chuyển DD/MM/YYYY + HH:MM thành datetime để so sánh."""
        try:
            d = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            return d
        except: return None

    dt_from = parse_dt(tu_ngay, tu_gio or "00:00") if tu_ngay else None
    dt_to   = parse_dt(den_ngay, den_gio or "23:59") if den_ngay else None

    results = []
    for r in db:
        # Lọc thời gian
        tgvp = r.get("thoi_gian_vi_pham","")  # "HH:MM DD/MM/YYYY"
        if dt_from or dt_to:
            try:
                parts = tgvp.split(" ")
                r_dt = datetime.strptime(f"{parts[1]} {parts[0]}", "%d/%m/%Y %H:%M")
                if dt_from and r_dt < dt_from: continue
                if dt_to   and r_dt > dt_to:   continue
            except: continue

        # Lọc camera
        if cam_id and cam_id != "all":
            r_cam = r.get("cam_id","") or ""
            if r_cam != cam_id: continue

        # Lọc biển số (partial, không phân biệt hoa/thường)
        if bien_so:
            r_bs = normalize_plate(r.get("bien_so",""))
            if normalize_plate(bien_so) not in r_bs: continue

        # Lọc trạng thái
        if trang_thai and trang_thai != "all":
            if r.get("trang_thai","") != trang_thai: continue

        results.append(r)

    results.sort(key=parse_vio_time, reverse=True)
    total = len(results)
    start = (page-1)*limit
    page_data = results[start:start+limit]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": max(1, (total+limit-1)//limit),
        "violations": page_data,
    }



class XuPhatBody(BaseModel):
    bien_so: str

class CapNhatBienSoBody(BaseModel):
    ma_tam: str      # giá trị "bien_so" tạm hiện tại (vd: CHUA_XAC_DINH-6A1511)
    bien_so_moi: str # biển số thật do người dùng nhập tay

@app.patch("/violations/cap-nhat-bien-so")
async def cap_nhat_bien_so(body: CapNhatBienSoBody, user: dict = Depends(get_current_user)):
    """Cập nhật biển số cho hồ sơ vi phạm 'chưa rõ biển' sau khi cán bộ tra thủ công.
    Sau khi cập nhật, hồ sơ sẽ được đánh dấu đã đọc được biển và xuất hiện ngay
    trong danh sách/tra cứu/báo cáo như các hồ sơ bình thường khác."""
    bien_so_moi = body.bien_so_moi.strip().upper()
    if not bien_so_moi:
        raise HTTPException(400, "Vui lòng nhập biển số")
    db = load_violation_db()
    found = None
    for r in db:
        if r.get("bien_so") == body.ma_tam and not r.get("bien_so_doc_duoc", True):
            found = r
            break
    if not found:
        raise HTTPException(404, "Không tìm thấy hồ sơ chưa rõ biển số này")
    found["bien_so"] = bien_so_moi
    found["bien_so_doc_duoc"] = True
    found["bien_so_nhap_thu_cong_boi"] = user["username"]
    with open(MOCK_DB_FILE, "w", encoding="utf-8") as f:
        _json2.dump(db, f, ensure_ascii=False, indent=2)
    write_audit(user, "CAP_NHAT_BIEN_SO_THU_CONG", target=bien_so_moi,
                detail=f"Từ mã tạm {body.ma_tam} -> {bien_so_moi}")
    return {"ok": True, "bien_so": bien_so_moi}

@app.patch("/violations/xu-phat")
async def mark_xu_phat(body: XuPhatBody, user: dict = Depends(require_role("admin", "xu_phat"))):
    """Đánh dấu hồ sơ đã xử phạt — chỉ cán bộ xử phạt/admin. Ghi audit log để truy vết ai đã xử lý."""
    db = load_violation_db()
    found = False
    for r in db:
        if r.get("bien_so") == body.bien_so and r.get("trang_thai") != "Đã xử phạt":
            r["trang_thai"] = "Đã xử phạt"
            r["nguoi_xu_phat"] = user["username"]
            r["thoi_gian_xu_phat"] = datetime.now().strftime("%H:%M %d/%m/%Y")
            found = True
            break
    if not found:
        raise HTTPException(404, "Không tìm thấy hồ sơ chưa xử phạt với biển số này")
    with open(MOCK_DB_FILE, "w", encoding="utf-8") as f:
        _json2.dump(db, f, ensure_ascii=False, indent=2)
    write_audit(user, "XU_PHAT_VI_PHAM", target=body.bien_so, detail=f"Đánh dấu đã xử phạt")
    return {"ok": True}

@app.delete("/violations/{bien_so}")
async def delete_violation(bien_so: str, user: dict = Depends(require_role("admin"))):
    """Xoá hồ sơ vi phạm — chỉ admin, bắt buộc ghi audit log vì là thao tác xoá dữ liệu."""
    db = load_violation_db()
    new_db = [r for r in db if r.get("bien_so") != bien_so]
    if len(new_db) == len(db):
        raise HTTPException(404, "Không tìm thấy hồ sơ")
    with open(MOCK_DB_FILE, "w", encoding="utf-8") as f:
        _json2.dump(new_db, f, ensure_ascii=False, indent=2)
    write_audit(user, "XOA_VI_PHAM", target=bien_so, detail="Xoá hồ sơ vi phạm")
    return {"ok": True}

@app.get("/debug/ocr-variant-stats")
async def ocr_variant_stats(user: dict = Depends(get_current_user)):
    """Xem biến thể ảnh (otsu/bright/color_resized/gray/color_rgb) nào đang cho kết quả
    OCR biển số tốt nhất, và tỉ lệ early-exit đang tiết kiệm được bao nhiêu % inference.
    Số liệu reset mỗi khi restart server (lưu trong RAM, không ghi DB)."""
    total_wins = sum(_variant_win_stats.values()) or 1
    return {
        "total_crops_processed": _variant_total_runs,
        "early_exits": _variant_early_exits,
        "early_exit_rate_pct": round(_variant_early_exits/max(_variant_total_runs,1)*100, 1),
        "by_variant_win": sorted(
            [{"variant": k, "wins": v, "pct_of_wins": round(v/total_wins*100, 1)}
             for k, v in _variant_win_stats.items()],
            key=lambda x: -x["wins"]
        ),
    }

@app.get("/ui")
async def serve_ui():
    from fastapi.responses import FileResponse
    return FileResponse("index.html")
