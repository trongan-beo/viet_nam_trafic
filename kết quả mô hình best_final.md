là mô hình để bắt biển số  
khảo sát 3 mô hình và có kết quả sau
yolov8n
 mAP50=0.9856 | mAP50-95=0.9212 | Size=6.25MB
yolov8s
mAP50=0.9882 | mAP50-95=0.9316 | Size=22.51MB
yolov8n_seg
mAP50=0.9862 | mAP50-95=0.871 | Size=6.78MB
📋 BẢNG TỔNG HỢP KHẢO SÁT 3 MÔ HÌNH
======================================================================
               mAP50  mAP50-95  Precision  Recall  Size (MB)
Model                                                                  
yolov8n             0.9856    0.9212     0.9816  0.9616       6.25
yolov8s             0.9882    0.9316     0.9739  0.9764      22.51
yolov8n_seg         0.9862    0.8710     0.9790  0.9655       6.78
YOLOv8n (detect)         : weighted score = 0.9462
YOLOv8s (detect)         : weighted score = 0.9564
YOLOv8n-seg              : weighted score = 0.9224

🏆 MÔ HÌNH TỐT NHẤT: YOLOv8s (detect)
   mAP50=0.9882 | mAP50-95=0.9316 | Recall=0.9764
   # ── 4.1 Hyperparameter tối ưu ────────────────────────────────────────
FINAL_EPOCHS    = 100
FINAL_BATCH     = 16      # Tăng lên 32 nếu RAM GPU còn trống
FINAL_IMG_SIZE  = 640     # Tăng từ 512 → 640 để bắt biển số nhỏ hơn
FINAL_LR0       = 0.01    # Learning rate khởi đầu
FINAL_LRF       = 0.001   # Learning rate cuối (cosine decay)
FINAL_MOMENTUM  = 0.937
FINAL_WD        = 0.0005  # Weight decay
FINAL_PATIENCE  = 20      # Early stopping

print(f'⚙️  Cấu hình train chuyên sâu:')
print(f'   Model    : {BEST_PRETRAINED}')
print(f'   Epochs   : {FINAL_EPOCHS}')
print(f'   Img size : {FINAL_IMG_SIZE}')
print(f'   Batch    : {FINAL_BATCH}')
print(f'   LR       : {FINAL_LR0} → {FINAL_LRF}')
kết quả cuối của yolov8s 
========================================
  mAP50     : 0.9818
  mAP50-95  : 0.9182
  Precision : 0.9808
  Recall    : 0.9709
  F1 Score  : 0.9758