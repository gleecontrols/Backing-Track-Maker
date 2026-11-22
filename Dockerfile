FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 timgm6mb-soundfont \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir setuptools fastapi "uvicorn[standard]" python-multipart \
    librosa numpy scipy soundfile pretty_midi pedalboard \
    && pip install --no-cache-dir --no-deps tinysoundfont
COPY . .
RUN mkdir -p data
ENV PORT=8000
ENV PIANO_SF2=/usr/share/sounds/sf2/TimGM6mb.sf2
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT}"]
