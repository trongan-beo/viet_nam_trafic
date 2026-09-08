"""
Tạo mock database phạt nguội theo chuẩn VNeID / csgt.vn
Nghị định 168/2024/NĐ-CP
"""
import json, random
from datetime import datetime, timedelta

# ── Dữ liệu tham chiếu ─────────────────────────────────────────────────────

MA_TINH = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh", "15": "Hải Phòng",
    "16": "Hải Dương", "17": "Hưng Yên", "18": "Thái Bình", "19": "Nam Định",
    "20": "Ninh Bình", "21": "Hà Nội", "29": "Hà Nội", "30": "Hà Nội",
    "31": "Hà Nội", "32": "Hà Nội", "33": "Hà Nội", "34": "Hà Nội",
    "40": "Hà Nội", "47": "Đắk Lắk", "48": "Đắk Nông", "49": "Lâm Đồng",
    "51": "TP. Hồ Chí Minh", "52": "TP. Hồ Chí Minh", "53": "TP. Hồ Chí Minh",
    "54": "TP. Hồ Chí Minh", "55": "TP. Hồ Chí Minh", "56": "TP. Hồ Chí Minh",
    "57": "TP. Hồ Chí Minh", "58": "TP. Hồ Chí Minh", "59": "TP. Hồ Chí Minh",
    "60": "Đồng Nai", "61": "Bình Dương", "62": "Long An", "63": "Tiền Giang",
    "64": "Vĩnh Long", "65": "Cần Thơ", "67": "An Giang", "68": "Kiên Giang",
    "69": "Cà Mau", "70": "Tây Ninh", "71": "Bến Tre", "72": "Bà Rịa–Vũng Tàu",
    "73": "Quảng Bình", "74": "Quảng Trị", "75": "Thừa Thiên Huế",
    "76": "Quảng Ngãi", "77": "Bình Định", "78": "Phú Yên", "79": "Khánh Hòa",
    "80": "Kon Tum", "81": "Gia Lai", "82": "Bình Phước", "83": "Trà Vinh",
    "84": "Sóc Trăng", "85": "Đồng Tháp", "86": "Hậu Giang", "88": "Vĩnh Phúc",
    "89": "Hưng Yên", "90": "Hà Nam", "92": "Thanh Hóa", "93": "Nghệ An",
    "94": "Hà Tĩnh", "95": "Bắc Ninh", "96": "Bắc Giang", "97": "Phú Thọ",
    "98": "Thái Nguyên", "99": "Bắc Kạn",
}

LOAI_PHUONG_TIEN = ["Ô tô con", "Xe máy", "Xe buýt", "Xe tải", "Xe khách"]

MAU_BIEN = {
    "Ô tô con":  "Nền màu trắng, chữ và số màu đen",
    "Xe máy":    "Nền màu trắng, chữ và số màu đen",
    "Xe buýt":   "Nền màu vàng, chữ và số màu đen",
    "Xe tải":    "Nền màu xanh lá, chữ và số màu trắng",
    "Xe khách":  "Nền màu vàng, chữ và số màu đen",
}

# Hành vi vi phạm theo Nghị định 168/2024/NĐ-CP
HANH_VI = [
    {
        "ma": "16824.6.2.đ.21",
        "mo_ta": 'Dừng xe nơi có biển "Cấm dừng xe và đỗ xe"',
        "muc_phat_oto": "800.000 - 1.000.000 đồng",
        "muc_phat_xe_may": "400.000 - 600.000 đồng",
    },
    {
        "ma": "16824.5.1.a",
        "mo_ta": "Vượt đèn đỏ, đèn vàng",
        "muc_phat_oto": "4.000.000 - 6.000.000 đồng",
        "muc_phat_xe_may": "1.000.000 - 2.000.000 đồng",
    },
    {
        "ma": "16824.6.1.c",
        "mo_ta": "Đi ngược chiều đường một chiều",
        "muc_phat_oto": "3.000.000 - 5.000.000 đồng",
        "muc_phat_xe_may": "1.000.000 - 2.000.000 đồng",
    },
    {
        "ma": "16824.6.2.a",
        "mo_ta": "Dừng xe, đỗ xe trên phần đường xe chạy ảnh hưởng đến giao thông",
        "muc_phat_oto": "800.000 - 1.000.000 đồng",
        "muc_phat_xe_may": "200.000 - 400.000 đồng",
    },
    {
        "ma": "16824.7.1.b",
        "mo_ta": "Vượt xe trong hầm đường bộ",
        "muc_phat_oto": "3.000.000 - 5.000.000 đồng",
        "muc_phat_xe_may": "1.000.000 - 2.000.000 đồng",
    },
    {
        "ma": "16824.5.3.đ",
        "mo_ta": "Chạy quá tốc độ quy định từ 10 km/h đến dưới 20 km/h",
        "muc_phat_oto": "3.000.000 - 5.000.000 đồng",
        "muc_phat_xe_may": "1.000.000 - 2.000.000 đồng",
    },
    {
        "ma": "16824.5.3.e",
        "mo_ta": "Chạy quá tốc độ quy định từ 20 km/h đến dưới 35 km/h",
        "muc_phat_oto": "6.000.000 - 8.000.000 đồng",
        "muc_phat_xe_may": "3.000.000 - 5.000.000 đồng",
    },
    {
        "ma": "16824.6.4.c",
        "mo_ta": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        "muc_phat_oto": "4.000.000 - 6.000.000 đồng",
        "muc_phat_xe_may": "1.000.000 - 2.000.000 đồng",
    },
    {
        "ma": "16824.5.2.b",
        "mo_ta": "Đi không đúng phần đường, làn đường quy định",
        "muc_phat_oto": "800.000 - 1.000.000 đồng",
        "muc_phat_xe_may": "300.000 - 400.000 đồng",
    },
    {
        "ma": "16824.6.3.a",
        "mo_ta": "Đỗ xe trên cầu gây cản trở giao thông",
        "muc_phat_oto": "1.000.000 - 2.000.000 đồng",
        "muc_phat_xe_may": "400.000 - 600.000 đồng",
    },
]

DON_VI_PHAT_HIEN = [
    "Đội CHGT & ĐK Đèn THGT – Phòng CSGT Công an Tp Hồ Chí Minh",
    "Phòng CSGT – Công an TP. Hà Nội",
    "Trạm CSGT Cầu Nghìn – Phòng CSGT – CAT Hưng Yên",
    "Đội CSGT số 3 – Phòng CSGT – CAT Đà Nẵng",
    "Phòng CSGT – CAT Bình Dương",
    "Đội CSGT – Trật tự – CAT Quận 7 – TP. HCM",
    "Trạm CSGT Dầu Giây – Phòng CSGT – CAT Đồng Nai",
    "Phòng CSGT – CAT Hải Phòng",
    "Đội CSGT số 1 – Phòng CSGT – CAT Nghệ An",
    "Phòng CSGT – CAT Khánh Hòa",
]

DON_VI_GIAI_QUYET = {
    "Đội CHGT & ĐK Đèn THGT – Phòng CSGT Công an Tp Hồ Chí Minh": [
        {"ten": "Đội CHGT & ĐK Đèn THGT – Phòng CSGT Công an Tp Hồ Chí Minh",
         "dia_chi": "341 Trần Hưng Đạo, Phường Cầu Ông Lãnh, TP. Hồ Chí Minh"},
    ],
    "Phòng CSGT – Công an TP. Hà Nội": [
        {"ten": "Phòng CSGT – Công an TP. Hà Nội",
         "dia_chi": "75 Phùng Hưng, Hoàn Kiếm, Hà Nội"},
    ],
    "Trạm CSGT Cầu Nghìn – Phòng CSGT – CAT Hưng Yên": [
        {"ten": "Trạm CSGT Cầu Nghìn – Phòng CSGT – CAT Hưng Yên",
         "dia_chi": "Tổ 5, Hưng Yên"},
    ],
    "Đội CSGT số 3 – Phòng CSGT – CAT Đà Nẵng": [
        {"ten": "Đội CSGT số 3 – Phòng CSGT – CAT Đà Nẵng",
         "dia_chi": "99 Điện Biên Phủ, Thanh Khê, Đà Nẵng"},
    ],
    "Phòng CSGT – CAT Bình Dương": [
        {"ten": "Phòng CSGT – CAT Bình Dương",
         "dia_chi": "Đường Thích Quảng Đức, Phú Cường, Bình Dương"},
    ],
    "Đội CSGT – Trật tự – CAT Quận 7 – TP. HCM": [
        {"ten": "Đội CSGT – Trật tự – CAT Quận 7 – TP. HCM",
         "dia_chi": "Nguyễn Thị Thập, Tân Phú, Quận 7, TP. HCM"},
    ],
    "Trạm CSGT Dầu Giây – Phòng CSGT – CAT Đồng Nai": [
        {"ten": "Trạm CSGT Dầu Giây – Phòng CSGT – CAT Đồng Nai",
         "dia_chi": "Quốc lộ 1A, Dầu Giây, Đồng Nai"},
    ],
    "Phòng CSGT – CAT Hải Phòng": [
        {"ten": "Phòng CSGT – CAT Hải Phòng",
         "dia_chi": "Lê Đại Hành, Hồng Bàng, Hải Phòng"},
    ],
    "Đội CSGT số 1 – Phòng CSGT – CAT Nghệ An": [
        {"ten": "Đội CSGT số 1 – Phòng CSGT – CAT Nghệ An",
         "dia_chi": "Nguyễn Thị Minh Khai, Vinh, Nghệ An"},
    ],
    "Phòng CSGT – CAT Khánh Hòa": [
        {"ten": "Phòng CSGT – CAT Khánh Hòa",
         "dia_chi": "4 Yersin, Vĩnh Hải, Nha Trang, Khánh Hòa"},
    ],
}

DIA_DIEM_THEO_TINH = {
    "TP. Hồ Chí Minh": [
        "Đường 400, Phường Tăng Nhơn Phú, TP. Hồ Chí Minh",
        "Xa lộ Hà Nội, Phường Hiệp Phú, Quận 9, TP. Hồ Chí Minh",
        "Đinh Tiên Hoàng, Phường 3, Bình Thạnh, TP. Hồ Chí Minh",
        "Nguyễn Văn Linh, Phường Tân Phong, Quận 7, TP. Hồ Chí Minh",
        "Cộng Hòa, Phường 13, Tân Bình, TP. Hồ Chí Minh",
    ],
    "Hà Nội": [
        "Nguyễn Trãi, Thanh Xuân, Hà Nội",
        "Giải Phóng, Hoàng Mai, Hà Nội",
        "Phạm Văn Đồng, Bắc Từ Liêm, Hà Nội",
        "Trần Duy Hưng, Cầu Giấy, Hà Nội",
        "Láng Hạ, Đống Đa, Hà Nội",
    ],
    "Bình Dương": [
        "QL 13, Phường Lái Thiêu, Thuận An, Bình Dương",
        "Đại lộ Bình Dương, Thủ Dầu Một, Bình Dương",
    ],
    "Đồng Nai": [
        "Quốc lộ 1A, Trảng Bom, Đồng Nai",
        "Quốc lộ 51, Long Thành, Đồng Nai",
    ],
    "Hải Phòng": [
        "Lạch Tray, Ngô Quyền, Hải Phòng",
        "Trần Phú, Hồng Bàng, Hải Phòng",
    ],
    "Đà Nẵng": [
        "Điện Biên Phủ, Thanh Khê, Đà Nẵng",
        "Nguyễn Văn Linh, Hải Châu, Đà Nẵng",
    ],
    "Khánh Hòa": [
        "Trần Phú, Vĩnh Nguyên, Nha Trang, Khánh Hòa",
        "Quốc lộ 1A, Diên Khánh, Khánh Hòa",
    ],
    "Hưng Yên": [
        "Quốc lộ 5, Mỹ Hào, Hưng Yên",
    ],
    "Nghệ An": [
        "Quốc lộ 1A, Nghi Lộc, Nghệ An",
        "Lê Nin, Vinh, Nghệ An",
    ],
}

TRANG_THAI = ["Chưa xử phạt", "Đã xử phạt"]
TRANG_THAI_WEIGHT = [0.65, 0.35]

# ── Generator ──────────────────────────────────────────────────────────────

def gen_plate(ma_tinh: str) -> str:
    series_letter = random.choice("ABCDEFGHKLMNPSTUVXY")
    so = random.randint(10000, 99999)
    return f"{ma_tinh}{series_letter}-{so}"

def random_datetime(days_back=180) -> datetime:
    base = datetime(2025, 1, 1)
    delta = timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(6, 22),
        minutes=random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]),
    )
    return base + delta

def get_tinh_from_don_vi(don_vi: str) -> str:
    if "Hồ Chí Minh" in don_vi: return "TP. Hồ Chí Minh"
    if "Hà Nội" in don_vi: return "Hà Nội"
    if "Hưng Yên" in don_vi: return "Hưng Yên"
    if "Đà Nẵng" in don_vi: return "Đà Nẵng"
    if "Bình Dương" in don_vi: return "Bình Dương"
    if "Đồng Nai" in don_vi: return "Đồng Nai"
    if "Hải Phòng" in don_vi: return "Hải Phòng"
    if "Nghệ An" in don_vi: return "Nghệ An"
    if "Khánh Hòa" in don_vi: return "Khánh Hòa"
    return "TP. Hồ Chí Minh"

def get_ma_tinh_for_tinh(tinh: str) -> str:
    mapping = {
        "TP. Hồ Chí Minh": random.choice(["51","52","53","54","55","56","57","58","59"]),
        "Hà Nội": random.choice(["29","30","31","32","33","34","40"]),
        "Hưng Yên": "89",
        "Đà Nẵng": "43",
        "Bình Dương": "61",
        "Đồng Nai": "60",
        "Hải Phòng": "15",
        "Nghệ An": "93",
        "Khánh Hòa": "79",
    }
    return mapping.get(tinh, "51")

def build_record(plate_text: str = None) -> dict:
    don_vi = random.choice(DON_VI_PHAT_HIEN)
    tinh = get_tinh_from_don_vi(don_vi)
    ma_tinh = get_ma_tinh_for_tinh(tinh)

    if plate_text is None:
        plate_text = gen_plate(ma_tinh)

    loai = random.choice(LOAI_PHUONG_TIEN)
    hanh_vi = random.choice(HANH_VI)
    trang_thai = random.choices(TRANG_THAI, weights=TRANG_THAI_WEIGHT)[0]
    dt = random_datetime()
    dia_diem_list = DIA_DIEM_THEO_TINH.get(tinh, ["Quốc lộ 1A"])
    dia_diem = random.choice(dia_diem_list)
    don_vi_gq = DON_VI_GIAI_QUYET.get(don_vi, [{"ten": don_vi, "dia_chi": ""}])

    is_oto = loai in ["Ô tô con", "Xe buýt", "Xe tải", "Xe khách"]
    muc_phat = hanh_vi["muc_phat_oto"] if is_oto else hanh_vi["muc_phat_xe_may"]

    return {
        "bien_so": plate_text,
        "loai_phuong_tien": loai,
        "thoi_gian_vi_pham": dt.strftime("%H:%M %d/%m/%Y"),
        "dia_diem_vi_pham": dia_diem,
        "hanh_vi_vi_pham": f"{hanh_vi['ma']} {hanh_vi['mo_ta']}",
        "muc_phat": muc_phat,
        "trang_thai": trang_thai,
        "don_vi_phat_hien": don_vi,
        "don_vi_giai_quyet": don_vi_gq,
    }

# ── Main ───────────────────────────────────────────────────────────────────

random.seed(42)
records = [build_record() for _ in range(500)]

# Thêm một số bản ghi với biển số cụ thể để dễ test
test_plates = ["89H09275", "51F12345", "29A99999", "43B56789", "61C11111"]
for p in test_plates:
    records.append(build_record(plate_text=p))

with open("mock_violations.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✅ Đã tạo {len(records)} bản ghi → mock_violations.json")
print(f"   Test plates: {test_plates}")
