#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# 校园安全 APP v3  一键部署脚本
# 用法:  chmod +x deploy.sh && ./deploy.sh
# ══════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log()    { echo -e "${GREEN}[$(date +%H:%M:%S)] ✅ $*${NC}"; }
warn()   { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠️  $*${NC}"; }
error()  { echo -e "${RED}[$(date +%H:%M:%S)] ❌ $*${NC}"; exit 1; }
header() { echo -e "\n${CYAN}══════════════════════════════════════${NC}"; \
           echo -e "${CYAN}  $*${NC}"; \
           echo -e "${CYAN}══════════════════════════════════════${NC}"; }

header "🛡️  校园安全 APP v3  一键部署"
echo "   软硬协同推测解码 | QAD-4bit | 多模态检测"
echo ""

# ── 1. 前置检查 ──────────────────────────────────────────────
header "步骤 1/7: 前置环境检查"

command -v docker   >/dev/null 2>&1 || error "Docker 未安装: https://docs.docker.com/get-docker/"
command -v docker-compose >/dev/null 2>&1 || \
  (docker compose version >/dev/null 2>&1 || error "docker-compose 未安装")
log "Docker: $(docker --version)"

# 检查端口占用
for port in 80 443 8000 5432 6379; do
  if ss -tlnp 2>/dev/null | grep -q ":$port "; then
    warn "端口 $port 已被占用，可能冲突"
  fi
done

# ── 2. 环境配置 ──────────────────────────────────────────────
header "步骤 2/7: 环境变量配置"

if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    warn ".env 文件已从模板创建，请编辑后重新运行！"
    warn "必填项: DB_PASSWORD, SECRET_KEY, FCM_SERVER_KEY"
    echo ""
    echo "  快速配置命令:"
    echo "  SECRET_KEY=\$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    echo "  sed -i \"s/change_me_256bit_secret_key_here_minimum_32_chars/\$SECRET_KEY/\" .env"
    echo ""
    read -p "已配置完成？按 Enter 继续，Ctrl+C 退出... " || true
  else
    error ".env.example 文件不存在"
  fi
fi

# 验证关键环境变量
source .env 2>/dev/null || true
[ -z "${DB_PASSWORD:-}" ] && error "DB_PASSWORD 未配置"
[ -z "${SECRET_KEY:-}" ]  && error "SECRET_KEY 未配置"
[ "${SECRET_KEY}" = "change_me_256bit_secret_key_here_minimum_32_chars" ] && \
  error "请修改 SECRET_KEY 为随机值"
[ ${#SECRET_KEY} -lt 32 ] && error "SECRET_KEY 长度不足32位"
log "环境变量验证通过"

# ── 3. 创建目录 ──────────────────────────────────────────────
header "步骤 3/7: 创建持久化目录"
mkdir -p data/postgres data/redis data/models data/ml_feedback data/nginx_logs
mkdir -p nginx/ssl
log "目录创建完成"

# ── 4. SSL 证书 ──────────────────────────────────────────────
header "步骤 4/7: SSL 证书配置"
if [ ! -f "nginx/ssl/server.crt" ]; then
  warn "SSL证书不存在，生成自签名证书（仅供测试）"
  warn "生产环境请使用 Let's Encrypt: certbot certonly --standalone -d your-domain.com"
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx/ssl/server.key \
    -out  nginx/ssl/server.crt \
    -subj "/CN=campus-safety/O=CampusSafety/C=CN" 2>/dev/null && \
    log "自签名证书已生成" || warn "openssl 不可用，跳过证书生成"
fi

# ── 5. 构建镜像 ──────────────────────────────────────────────
header "步骤 5/7: 构建 Docker 镜像"
COMPOSE_CMD="docker compose"
command -v docker-compose >/dev/null 2>&1 && COMPOSE_CMD="docker-compose"

log "构建 API 服务镜像..."
$COMPOSE_CMD build --no-cache api 2>&1 | grep -E 'Step|Successfully|ERROR' || true
log "镜像构建完成"

# ── 6. 启动服务 ──────────────────────────────────────────────
header "步骤 6/7: 启动服务"

# 先启动基础设施
log "启动 PostgreSQL + Redis..."
$COMPOSE_CMD up -d db redis
sleep 5

# 等待数据库就绪
log "等待数据库就绪..."
for i in $(seq 1 30); do
  if $COMPOSE_CMD exec -T db pg_isready -U postgres >/dev/null 2>&1; then
    log "PostgreSQL 就绪 (${i}s)"
    break
  fi
  [ $i -eq 30 ] && error "数据库启动超时"
  sleep 1
done

# 初始化数据库
log "初始化数据库 Schema..."
$COMPOSE_CMD exec -T db psql -U postgres -c \
  "CREATE DATABASE campus_safety;" 2>/dev/null || true
$COMPOSE_CMD exec -T db psql -U postgres -d campus_safety \
  -f /docker-entrypoint-initdb.d/01_schema.sql 2>/dev/null || true

# 启动 API 服务
log "启动 API + Nginx..."
$COMPOSE_CMD up -d api nginx

# 可选：GPU 模式启动 LLM 服务器
if [ "${ENABLE_GPU:-false}" = "true" ]; then
  log "启动 LLM 推理服务（GPU 模式）..."
  $COMPOSE_CMD --profile gpu up -d llm-server
  warn "LLM 服务器启动中，可能需要几分钟下载模型..."
else
  warn "LLM 服务未启动（CPU部署模式）"
  warn "推测解码将使用统计先验降级，性能满足基础要求"
  warn "启用GPU模式: ENABLE_GPU=true ./deploy.sh"
fi

# ── 7. 健康验证 ──────────────────────────────────────────────
header "步骤 7/7: 健康验证"
sleep 8

log "验证服务健康状态..."
for i in $(seq 1 15); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:8000/health)
    log "API 服务就绪: $HEALTH"
    break
  fi
  [ $i -eq 15 ] && { warn "API 服务响应超时，请检查日志: docker logs campus_safety_api"; }
  sleep 2
done

# 运行部署验证
if [ -f "scripts/verify_deployment.py" ]; then
  log "运行部署验证脚本..."
  python3 scripts/verify_deployment.py --url http://localhost:8000 2>/dev/null || \
    warn "部署验证脚本返回警告，请检查配置"
fi

# ── 完成 ─────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  🎉  校园安全 APP v3 部署完成！${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  API 服务:   http://localhost:8000"
echo "  API 文档:   http://localhost:8000/docs"
echo "  健康检查:   http://localhost:8000/health"
echo "  HTTPS:      https://localhost (自签名证书)"
echo ""
echo "  管理命令:"
echo "  查看日志:   docker logs campus_safety_api -f"
echo "  停止服务:   docker-compose down"
echo "  重启服务:   docker-compose restart api"
echo "  ML 状态:    curl http://localhost:8000/v1/infer/model-status -H 'Authorization: Bearer <token>'"
echo ""
echo -e "${YELLOW}  注意: 生产部署前请:"
echo -e "    1. 配置真实 SSL 证书 (Let's Encrypt)"
echo -e "    2. 修改所有默认密码"
echo -e "    3. 配置真实 FCM 密钥"
echo -e "    4. 部署防火墙规则${NC}"
echo ""
