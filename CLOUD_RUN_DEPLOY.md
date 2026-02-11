# Cloud Run Deploy (No Docker)

This folder is now prepared for Cloud Run source deploy using Python buildpacks.

## Required Cloud Run Settings

- Build context directory: `/`
- Entrypoint: leave blank (uses `Procfile`)
- Function target: leave blank

## Startup Command

Cloud Run will use:

`web: uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}`

## Required Environment Variables

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_DB_URL` (or `DATABASE_URL`)

## Common Runtime Variables

- `APP_DATA_DIRECTORY=/tmp/surge-pptx`
- `TEMP_DIRECTORY=/tmp/surge-pptx`
- `ALLOW_SQLITE_FALLBACK=false`
- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_CALLS=100`
- `RATE_LIMIT_WINDOW=3600`

## Optional Provider/API Variables

- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `PIXABAY_API_KEY`
- `PEXELS_API_KEY`
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `SMTP_FROM`
- `FEEDBACK_EMAIL_TO`
