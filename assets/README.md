# Assets

This directory contains static assets for the AqarSense application.

## Background Image

Place your background image here:
- **Filename:** `AqarSense bg.jpg`
- **Recommended dimensions:** 1920×1080px or higher
- **Format:** JPG/JPEG
- **File size:** Keep under 2MB for optimal performance

The background image will be:
- Automatically loaded and embedded via base64 encoding
- Displayed with a 78% dark overlay for text readability
- Replaced with a gradient fallback if not found

## Usage

After placing your image:
```bash
python -m streamlit run app.py
```

The image will be applied automatically on app startup.
