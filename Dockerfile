FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 timgm6mb-soundfont \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir "setuptools<81" fastapi "uvicorn[standard]" python-multipart \
    librosa numpy scipy soundfile pretty_midi pedalboard mido \
    && pip install --no-cache-dir --no-deps tinysoundfont
COPY . .
RUN mkdir -p data
ENV PIANO_SF2=/usr/share/sounds/sf2/TimGM6mb.sf2
# Fail the BUILD (with a readable error) if anything is missing, instead of crash-looping at runtime.
RUN cd /app && python -c "import sys; sys.path.insert(0,'backend'); import main, render, tinysoundfont; print('startup check ok, soundfont:', render.find_soundfont())"
CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
