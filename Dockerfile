# Gauntlet web playground. Build once, run anywhere with a CPU:
#   docker build -t gauntlet .
#   docker run --rm -p 7860:7860 gauntlet          # http://127.0.0.1:7860
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    GAUNTLET_CACHE=/app/data/models GAUNTLET_PUBLIC=1 GAUNTLET_TRUST_PROXY=1 PORT=7860
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gauntlet gauntlet
COPY reports/report.html reports/report.html
# train + cache the three linear defenders at build time, so the container answers within a second of starting
RUN python -m gauntlet warm
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health',timeout=4)"
CMD ["sh", "-c", "python -m gauntlet serve --host 0.0.0.0 --port ${PORT:-7860}"]
