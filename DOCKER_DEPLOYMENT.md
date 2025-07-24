# Docker Deployment Guide

## Overview
The RIFE application can be deployed using Docker with optional authentication. Both the main LatentSync app and the standalone RIFE app support Docker deployment.

## Available Dockerfiles

### 1. Main LatentSync Application
- **File**: `/Dockerfile`
- **Port**: 8000
- **Application**: `gradio_app.py` (includes RIFE tab)

### 2. RIFE Standalone Application
- **File**: `/ECCV2022-RIFE/Dockerfile`
- **Port**: 7860
- **Application**: `rife_app/app.py` (RIFE only)

## Authentication Configuration

### Default Behavior (Local Development)
- **Authentication**: Disabled by default
- **Environment Variables**: `AUTH_USERNAME=""` and `AUTH_PASSWORD=""`
- **Access**: No login required

### Production Deployment
To enable authentication, override the environment variables:

```bash
# Docker run command
docker run -e AUTH_USERNAME=admin -e AUTH_PASSWORD=secure_password -p 8000:8000 your-image

# Docker Compose
services:
  latentsync:
    image: your-image
    ports:
      - "8000:8000"
    environment:
      - AUTH_USERNAME=admin
      - AUTH_PASSWORD=secure_password
```

## Build and Run Instructions

### Build Main LatentSync App
```bash
# Build the image
docker build -t latentsync-app .

# Run without authentication (local development)
docker run -p 8000:8000 latentsync-app

# Run with authentication (production)
docker run -e AUTH_USERNAME=admin -e AUTH_PASSWORD=your_password -p 8000:8000 latentsync-app
```

### Build RIFE Standalone App
```bash
# Navigate to RIFE directory
cd ECCV2022-RIFE

# Build the image
docker build -t rife-app .

# Run without authentication (local development)
docker run -p 7860:7860 rife-app

# Run with authentication (production)
docker run -e AUTH_USERNAME=admin -e AUTH_PASSWORD=your_password -p 7860:7860 rife-app
```

## Docker Compose Example

```yaml
version: '3.8'

services:
  latentsync:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AUTH_USERNAME=  # Empty = disabled auth
      - AUTH_PASSWORD=  # Empty = disabled auth
    volumes:
      - ./results:/app/results  # Persist results
      - ./temp:/app/temp        # Persist temp files

  rife:
    build: ./ECCV2022-RIFE
    ports:
      - "7860:7860"
    environment:
      - AUTH_USERNAME=  # Empty = disabled auth
      - AUTH_PASSWORD=  # Empty = disabled auth
    volumes:
      - ./temp_gradio:/app/temp_gradio  # Persist RIFE temp files
```

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_USERNAME` | `""` | Username for HTTP Basic Auth (empty = disabled) |
| `AUTH_PASSWORD` | `""` | Password for HTTP Basic Auth (empty = disabled) |
| `GRADIO_SERVER_NAME` | `"0.0.0.0"` | Server bind address |

## Security Considerations

### Local Development
- Authentication is disabled by default for ease of development
- No credentials required to access the applications

### Production Deployment
- **Always set strong credentials** for production deployments
- Use environment variables or secrets management for credentials
- Consider using HTTPS proxy (nginx, Traefik) in front of the applications
- Regularly rotate passwords

### Platform-Specific Deployment

#### Coolify
```bash
# Set in Coolify environment variables
AUTH_USERNAME=your-username
AUTH_PASSWORD=your-secure-password
```

#### Railway/Render/Heroku
Set environment variables in your platform's dashboard:
- `AUTH_USERNAME=your-username`
- `AUTH_PASSWORD=your-secure-password`

#### Kubernetes
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: auth-credentials
data:
  AUTH_USERNAME: <base64-encoded-username>
  AUTH_PASSWORD: <base64-encoded-password>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: latentsync
spec:
  template:
    spec:
      containers:
      - name: app
        image: latentsync-app
        envFrom:
        - secretRef:
            name: auth-credentials
```

## Troubleshooting

### Authentication Issues
- **Problem**: Can't access app even with credentials set
- **Solution**: Check that both `AUTH_USERNAME` and `AUTH_PASSWORD` are non-empty

### Container Won't Start
- **Problem**: Container exits immediately
- **Solution**: Check that required model files are present and accessible

### Port Conflicts
- **Problem**: Port already in use
- **Solution**: Use different host ports: `-p 8001:8000` or `-p 7861:7860`

## Health Checks

Both Dockerfiles include health checks:
- **Main App**: HTTP check on port 8000
- **RIFE App**: HTTP check on port 7860 with curl

Monitor container health:
```bash
docker ps  # Shows health status
docker inspect <container-id>  # Detailed health info
```