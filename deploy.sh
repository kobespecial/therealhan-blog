#!/bin/bash

set -e

echo "▶ 生成 Hugo 静态文件..."
hugo --cleanDestinationDir

echo "▶ 同步到 VPS..."
rsync -avz -e "ssh -p 2288" --delete public/ root@89.208.245.123:/usr/share/nginx/html/

echo "▶ 远程 reload nginx..."
ssh -p 2288 root@89.208.245.123 "systemctl reload nginx"

echo "✅ 部署完成"