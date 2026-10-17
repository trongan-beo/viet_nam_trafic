#!/bin/bash
# ==============================================================================
# Script Cập nhật & Rebuild ứng dụng Traffic AI trên Server
# ==============================================================================

echo "🔄 Đang build và khởi chạy lại các container Docker..."
docker compose down
docker compose up -d --build

echo "⚡ Trạng thái các container:"
docker compose ps

echo "✅ Hoàn tất deploy!"
