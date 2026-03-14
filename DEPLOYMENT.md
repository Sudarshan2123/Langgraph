# Production Deployment Guide

## Quick Start

### Local Production Mode
```bash
# Install dependencies
pip install -r requirements-prod.txt

# Set environment variables
cp .env.example .env
# Edit .env with your credentials

# Run server
python server.py
```

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## API Usage

### Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "lang": "en-US", "thread_id": "user123"}'
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Cloud Deployment Options

### AWS (Recommended)
1. **ECS/Fargate**: Push Docker image to ECR, deploy via ECS
2. **Lambda + API Gateway**: Use AWS Lambda Web Adapter
3. **EC2**: Run docker-compose on EC2 instance

### Google Cloud
- Deploy to Cloud Run (serverless containers)
- Use GKE for Kubernetes deployment

### Azure
- Azure Container Instances
- Azure App Service (containers)

## Production Checklist
- [ ] Set all environment variables securely
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring (CloudWatch, Datadog, etc.)
- [ ] Configure rate limiting
- [ ] Set up logging aggregation
- [ ] Enable authentication/API keys
- [ ] Configure CORS properly
- [ ] Set up CI/CD pipeline
- [ ] Enable auto-scaling
- [ ] Set up backup/disaster recovery
