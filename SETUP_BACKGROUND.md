# Background Image Setup

To add a custom background image to the AqarSense app:

1. **Prepare your image:**
   - Name it: `AqarSense bg.jpg`
   - Recommended size: 1920x1080px or higher
   - Format: JPG or JPEG

2. **Place the image:**
   - Copy `AqarSense bg.jpg` to the `assets/` directory:
     ```
     Rent_Prediction_UAE/
     ├── assets/
     │   └── AqarSense bg.jpg  ← Place your image here
     ├── app.py
     ├── data/
     ├── src/
     └── ...
     ```

3. **Restart the app:**
   ```bash
   python -m streamlit run app.py
   ```

The app will automatically:
- Load and embed the image with 78% dark overlay for text readability
- Fall back to a dark gradient background if the image is not found

**Note:** The image is base64-encoded at app startup for optimal performance.
