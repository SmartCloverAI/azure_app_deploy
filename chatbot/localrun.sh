docker build -t localrun .
docker run --rm -v ./_cache:/ai_fastapi_app/_cache localrun