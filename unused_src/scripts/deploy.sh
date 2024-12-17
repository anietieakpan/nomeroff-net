#!/bin/bash

# Configuration
CONFIG_FILE="deploy_config.env"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Default values
DEPLOY_ENV=${DEPLOY_ENV:-"production"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"registry.example.com"}
VERSION=${VERSION:-$(git describe --tags --always)}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    }
    
    # Check PostgreSQL client
    if ! command -v psql &> /dev/null; then
        warn "PostgreSQL client is not installed. Some database operations may not be available."
    }
}

# Build application
build_application() {
    log "Building application..."
    docker build -t ${DOCKER_REGISTRY}/license-plate-detector:${VERSION} .
    
    if [ $? -ne 0 ]; then
        error "Build failed"
        exit 1
    fi
}

# Push to registry
push_to_registry() {
    if [ "$DEPLOY_ENV" = "production" ]; then
        log "Pushing to registry..."
        docker push ${DOCKER_REGISTRY}/license-plate-detector:${VERSION}
        
        if [ $? -ne 0 ]; then
            error "Push to registry failed"
            exit 1
        fi
    fi
}

# Deploy application
deploy_application() {
    log "Deploying application..."
    
    # Create necessary directories
    mkdir -p logs output/detections output/videos
    
    # Generate docker-compose override file for environment
    cat > docker-compose.override.yml <<EOF
version: '3.8'
services:
  detector:
    image: ${DOCKER_REGISTRY}/license-plate-detector:${VERSION}
    environment:
      - DEPLOY_ENV=${DEPLOY_ENV}
EOF
    
    # Deploy using docker-compose
    docker-compose pull
    docker-compose up -d
    
    if [ $? -ne 0 ]; then
        error "Deployment failed"
        exit 1
    fi
}

# Database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    for i in {1..30}; do
        if docker-compose exec -T db pg_isready; then
            break
        fi
        echo "Waiting for database... ($i/30)"
        sleep 1
    done
    
    # Run migrations
    docker-compose exec -T detector python -m alembic upgrade head
    
    if [ $? -ne 0 ]; then
        error "Database migration failed"
        exit 1
    fi
}

# Health check
health_check() {
    log "Running health checks..."
    
    # Wait for application to start
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            break
        fi
        echo "Waiting for application... ($i/30)"
        sleep 1
    done
    
    # Check application health
    HEALTH_STATUS=$(curl -s http://localhost:8000/health)
    if [[ $HEALTH_STATUS != *"healthy"* ]]; then
        error "Health check failed"
        exit 1
    fi
}

# Main deployment flow
main() {
    log "Starting deployment for environment: $DEPLOY_ENV"
    
    check_prerequisites
    build_application
    push_to_registry
    deploy_application
    run_migrations
    health_check
    
    log "Deployment completed successfully!"
}

# Run main function
main "$@"