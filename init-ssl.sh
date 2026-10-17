#!/bin/bash
# ==============================================================================
# Script khởi tạo SSL Let's Encrypt tự động cho traffiai.online bằng Docker
# ==============================================================================

DOMAIN="trafficai.online"
EMAIL="admin@trafficai.online" # Thay đổi email nếu muốn nhận thông báo từ Let's Encrypt

echo "🚀 Bắt đầu cấu hình & phát hành SSL cho miền: $DOMAIN ..."

# 1. Tạo thư mục tạm thời cho Nginx cấu hình HTTP ban đầu (để xác minh Certbot)
mkdir -p nginx/conf.d

# Sao lưu cấu hình HTTPS đầy đủ
cp nginx/conf.d/traffiai.conf nginx/conf.d/traffiai.conf.bak

# Tạo cấu hình HTTP tạm thời
cat <<EOF > nginx/conf.d/traffiai.conf
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN www.$DOMAIN;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        proxy_pass http://frontend:80;
        proxy_set_header Host \$host;
    }
}
EOF

echo "📦 Đang khởi động Nginx (HTTP mode)..."
docker compose up -d proxy frontend backend

echo "🔑 Đang yêu cầu chứng chỉ SSL miễn phí từ Let's Encrypt cho $DOMAIN..."
docker compose run --rm certbot certonly --webroot \
    --webroot-path=/var/www/certbot \
    -d $DOMAIN -d www.$DOMAIN \
    --email $EMAIL --agree-tos --no-eff-email --force-renewal

# Restore cấu hình HTTPS đầy đủ
cp nginx/conf.d/traffiai.conf.bak nginx/conf.d/traffiai.conf
rm -f nginx/conf.d/traffiai.conf.bak

echo "🔄 Khởi động lại Nginx với SSL (HTTPS)..."
docker compose restart proxy

echo "✅ Hoàn tất! Trang web của bạn đã hoạt động tại https://$DOMAIN"
