# Authentication Configuration for Deployment

## Overview
Both the main Gradio app and the RIFE interpolation tool support authentication using environment variables. This allows secure credential management in deployment environments like Coolify.

## Environment Variables

Set these environment variables in your deployment:

- `AUTH_USERNAME`: Username for authentication (default: "" - disabled)
- `AUTH_PASSWORD`: Password for authentication (default: "" - disabled)

**Note**: Both applications now default to **disabled authentication** for local development. Authentication is only enabled when both username and password are provided.

## Coolify Configuration

In your Coolify application settings:

1. Go to your application's Environment Variables section
2. Add the following variables:
   ```
   AUTH_USERNAME=your_desired_username
   AUTH_PASSWORD=your_secure_password
   ```

## Docker Configuration

If using Docker directly:

```bash
docker run -e AUTH_USERNAME=myuser -e AUTH_PASSWORD=mypassword ...
```

Or in docker-compose.yml:

```yaml
services:
  app:
    environment:
      - AUTH_USERNAME=myuser
      - AUTH_PASSWORD=mypassword
```

## Security Notes

- If both variables are not set or are empty, authentication will be disabled
- Default credentials are provided for development but should be changed in production
- Use strong, unique passwords in production environments
- Consider using Coolify's secret management features for sensitive credentials

## Application Behavior

- **gradio_app.py**: Runs on port 8000 with authentication
- **rife_app/app.py**: Runs on port 7860 with authentication
- Both apps will print authentication status on startup

## Testing

To test locally with environment variables:

```bash
export AUTH_USERNAME=testuser
export AUTH_PASSWORD=testpass
python gradio_app.py
```