# TennisFlow API Deployment Guide

This guide explains how to deploy the TennisFlow API processing service on a cloud server.

## Prerequisites

- A cloud server (DigitalOcean, AWS, Google Cloud, etc.)
- Ubuntu 20.04 or newer
- Docker and Docker Compose installed
- Supabase account with credentials

## Server Setup

### 1. Set Up a Cloud Server

#### DigitalOcean Option:
1. Create a Droplet with at least 4GB RAM / 2CPUs
2. Select Ubuntu 20.04
3. Choose a data center region close to your users
4. Add SSH keys for secure access
5. Create the Droplet

#### AWS EC2 Option:
1. Launch an EC2 instance (t3.medium or larger recommended)
2. Select Ubuntu Server 20.04 LTS
3. Configure instance details
4. Add storage (at least 30GB)
5. Add security group (open ports 22, 80, 443)
6. Launch and connect using SSH

### 2. Install Docker and Docker Compose

Connect to your server via SSH and run:

```bash
# Update package index
sudo apt update

# Install prerequisites
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Update package index again
sudo apt update

# Install Docker
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Add your user to the docker group
sudo usermod -aG docker ${USER}

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Apply group changes
newgrp docker
```

### 3. Clone Repository

```bash
# Clone the repository
git clone https://github.com/lucaschinaglia/tennisflow.git
cd tennisflow/api
```

### 4. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit the .env file with your credentials
nano .env
```

Update the .env file with your Supabase credentials:

```
SUPABASE_URL=https://mmjpyrqiemwpoidbmcdg.supabase.co
SUPABASE_KEY=your_service_role_key
```

### 5. Set Up Database Tables

Run the SQL queries from `supabase/schema.sql` in your Supabase SQL editor to create the necessary tables and policies.

### 6. Build and Start the Services

```bash
# Build and start in detached mode
docker-compose up -d --build
```

### 7. Set Up Nginx as a Reverse Proxy (Optional for Production)

```bash
# Install Nginx
sudo apt install -y nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/tennisflow-api
```

Add the following configuration:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/tennisflow-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 8. Set Up SSL with Let's Encrypt (Optional for Production)

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.yourdomain.com
```

## Monitoring and Maintenance

### Check Service Status

```bash
# Check running containers
docker ps

# View logs
docker-compose logs -f
```

### Update the Service

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Backup Data

Redis data is ephemeral and doesn't need to be backed up. All persistent data is stored in Supabase.

## Scaling Considerations

- The worker service can be horizontally scaled by adjusting the `replicas` parameter in a Docker Swarm or Kubernetes setup.
- For higher throughput, you can run multiple workers across different servers.

## Troubleshooting

### API Service Not Starting

Check logs for errors:
```bash
docker-compose logs api
```

### Worker Not Processing Videos

Check Redis connection:
```bash
docker-compose logs redis
```

Check worker logs:
```bash
docker-compose logs worker
```

### OpenPose Issues

Ensure OpenPose is correctly built in the Docker container:
```bash
docker exec -it tennisflow_worker bash
cd /openpose
./build/examples/openpose/openpose.bin --help
```

## Mobile App Configuration

After deploying the API, update your Expo app's environment variables:

1. Create an `.env` file in your Expo app root:
```
EXPO_PUBLIC_API_URL=https://api.yourdomain.com
EXPO_PUBLIC_SUPABASE_URL=https://mmjpyrqiemwpoidbmcdg.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
```

2. Rebuild and publish your Expo app:
```bash
eas build --platform all
eas submit
```