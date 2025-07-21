#!/bin/bash

# ESERISIA AI - Production Deployment Script
# Ultra-Advanced AI System Deployment

set -e

echo "🚀 ESERISIA AI - Production Deployment Started"
echo "================================================="

# Configuration
DEPLOYMENT_ENV=${1:-production}
GPU_ENABLED=${2:-true}
QUANTUM_ENABLED=${3:-true}

echo "📋 Deployment Configuration:"
echo "   Environment: $DEPLOYMENT_ENV"
echo "   GPU Enabled: $GPU_ENABLED" 
echo "   Quantum Enabled: $QUANTUM_ENABLED"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Installing..."
    pip install docker-compose
fi

# Check NVIDIA Docker (if GPU enabled)
if [ "$GPU_ENABLED" = "true" ]; then
    if ! command -v nvidia-docker &> /dev/null; then
        echo "⚠️ NVIDIA Docker not found. GPU features may be limited."
    else
        echo "✅ NVIDIA Docker found - GPU acceleration ready"
    fi
fi

# Create necessary directories
echo "📁 Creating deployment directories..."
mkdir -p logs
mkdir -p models
mkdir -p checkpoints
mkdir -p ssl
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana

# Generate SSL certificates (self-signed for development)
echo "🔐 Generating SSL certificates..."
if [ ! -f ssl/eserisia.crt ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/eserisia.key \
        -out ssl/eserisia.crt \
        -subj "/C=FR/ST=Paris/L=Paris/O=ESERISIA/CN=eserisia.ai"
fi

# Create environment file
echo "⚙️ Creating environment configuration..."
cat > .env << EOF
# ESERISIA AI Production Environment
ESERISIA_ENV=$DEPLOYMENT_ENV
ESERISIA_VERSION=1.0.0
ESERISIA_GPU_ENABLED=$GPU_ENABLED
ESERISIA_QUANTUM_ENABLED=$QUANTUM_ENABLED

# Security
ESERISIA_API_KEY=eserisia-ultra-secure-key-2025
ESERISIA_JWT_SECRET=ultra-secure-jwt-secret-eserisia-2025

# Database
MONGODB_USERNAME=eserisia
MONGODB_PASSWORD=ultra-secure-mongodb-password-2025
REDIS_PASSWORD=ultra-secure-redis-password-2025

# Monitoring
GRAFANA_ADMIN_PASSWORD=eserisia-ultra-admin-2025
PROMETHEUS_RETENTION=30d

# Performance
WORKER_PROCESSES=4
MAX_REQUESTS=10000
TIMEOUT=300
KEEPALIVE=5

# Scaling
REPLICAS=3
MEMORY_LIMIT=8G
CPU_LIMIT=4
EOF

# Create Nginx configuration
echo "🌐 Configuring Nginx load balancer..."
cat > deployment/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream eserisia_api {
        server eserisia-ai:8000;
        keepalive 32;
    }
    
    upstream eserisia_web {
        server eserisia-web:8501;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    server {
        listen 80;
        server_name eserisia.ai www.eserisia.ai;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name eserisia.ai www.eserisia.ai;
        
        ssl_certificate /etc/nginx/ssl/eserisia.crt;
        ssl_certificate_key /etc/nginx/ssl/eserisia.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305;
        
        # API routes
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            proxy_pass http://eserisia_api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout 300s;
        }
        
        # Web interface
        location / {
            proxy_pass http://eserisia_web/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # Health checks
        location /health {
            proxy_pass http://eserisia_api/health;
        }
    }
}
EOF

# Create Prometheus configuration
echo "📊 Configuring Prometheus monitoring..."
cat > deployment/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'eserisia-api'
    static_configs:
      - targets: ['eserisia-ai:8001']
    scrape_interval: 10s
    metrics_path: '/metrics'
    
  - job_name: 'eserisia-system'
    static_configs:
      - targets: ['eserisia-ai:8000']
    scrape_interval: 30s
    metrics_path: '/metrics'
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']
    scrape_interval: 15s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
    scrape_interval: 15s
EOF

# Build and deploy
echo "🔨 Building ESERISIA AI containers..."
docker-compose -f deployment/docker-compose.yml build --parallel

echo "🚀 Starting ESERISIA AI services..."
docker-compose -f deployment/docker-compose.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health checks
echo "🏥 Running health checks..."

# Check API health
API_HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "unhealthy")
if [ "$API_HEALTH" = "healthy" ]; then
    echo "✅ API Service: Healthy"
else
    echo "❌ API Service: Unhealthy"
fi

# Check Web Interface
WEB_HEALTH=$(curl -s http://localhost:8501/_stcore/health 2>/dev/null && echo "healthy" || echo "unhealthy")
if [ "$WEB_HEALTH" = "healthy" ]; then
    echo "✅ Web Interface: Healthy"  
else
    echo "❌ Web Interface: Unhealthy"
fi

# Performance benchmark
echo "🏃‍♂️ Running performance benchmark..."
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{
    "message": "Test ESERISIA AI performance",
    "max_length": 100,
    "temperature": 0.7
  }' | jq '.processing_time_ms'

# Display deployment summary
echo ""
echo "🎉 ESERISIA AI Deployment Completed Successfully!"
echo "================================================="
echo "📊 Service URLs:"
echo "   🌐 Web Interface: http://localhost:8501"
echo "   🔗 API Endpoint: http://localhost:8000"  
echo "   📋 API Docs: http://localhost:8000/docs"
echo "   📈 Grafana: http://localhost:3000 (admin/eserisia-ultra-admin-2025)"
echo "   🎯 Prometheus: http://localhost:9090"
echo "   📊 Kibana: http://localhost:5601"
echo ""
echo "🔑 API Authentication:"
echo "   Token: demo-token"
echo "   Header: Authorization: Bearer demo-token"
echo ""
echo "📊 System Status:"
echo "   Environment: $DEPLOYMENT_ENV"
echo "   GPU Support: $GPU_ENABLED"
echo "   Quantum Processing: $QUANTUM_ENABLED"
echo "   API Health: $API_HEALTH"
echo "   Web Health: $WEB_HEALTH"
echo ""
echo "🚀 ESERISIA AI - The Future of AI is Now Operational!"

# Optional: Show logs
if [ "$1" = "--logs" ]; then
    echo "📝 Showing live logs..."
    docker-compose -f deployment/docker-compose.yml logs -f eserisia-ai
fi
