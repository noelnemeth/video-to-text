# Video to Text Transcriber

Ez egy egyszerű eszköz videó fájlok hanganyagának szöveggé alakításához az OpenAI Whisper modell (Faster-Whisper implementáció) segítségével.

## Előfeltételek

- Python 3.12+ (Ajánlott: **arm64** verzió Apple Silicon-on)
- `ffmpeg` (Homebrew-val telepíthető: `brew install ffmpeg`)

## Telepítés

1. Lépj be a mappába:
   ```bash
   cd ~/programozas/video-to-text
   ```
2. Aktiváld a virtuális környezetet:
   ```bash
   source venv/bin/activate
   ```
3. Telepítsd a függőségeket:
   ```bash
   pip install faster-whisper rich gradio
   ```

## Használat

### Grafikus felület (GUI) - Ajánlott
Indítsd el a böngészős felületet:
```bash
python app.py
```
Ezután nyisd meg a terminálban megjelenő URL-t. Itt egyszerűen behúzhatod a videó fájlokat.

### Parancssori felület (CLI)
```bash
python transcribe.py video.mp4 --model small --beam_size 1
```

## Teljesítmény és Apple Silicon (M1/M2/M3/M4)

Ha Apple Silicon chipet használsz és a program lassúnak tűnik, ellenőrizd, hogy a Python **arm64** (Native) vagy **x86_64** (Rosetta) módban fut-e:
```bash
python -c "import platform; print(platform.machine())"
```
Ha `x86_64`-et látsz, a program Rosetta emulációval fut, ami **5-10x lassabb**. A maximális sebességhez javasolt egy natív `arm64` Python és virtuális környezet használata.

### Új funkciók:
- **Gyors mód**: A `beam_size=1` beállítással az átírás jelentősen gyorsabb, minimális pontosságcsökkenés mellett.
- **Szálkezelés**: A program automatikusan kihasználja az összes CPU magot.
- **Állapotjelző**: A terminálban és a felületen is láthatod a folyamat haladását.
