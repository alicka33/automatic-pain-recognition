# Pain Detector Mobile App

A React Native mobile application built with Expo for pain detection and monitoring.

---

## Quick Start Guide

Follow these steps to run the app on your device:

### Step 1: Install Node.js

1. Go to [nodejs.org](https://nodejs.org/)
2. Download and install the **LTS version** (v18 or higher recommended)
3. Verify installation by opening a terminal/command prompt and running:
   ```bash
   node --version
   npm --version
   ```

### Step 2: Clone and Navigate to the App

1. Clone the repository:
   ```bash
   git clone https://github.com/alicka33/automatic-pain-recognition.git
   ```
2. Navigate to the app folder:
   ```bash
   cd automatic-pain-recognition/pain_detection_app
   ```

### Step 3: Install Project Dependencies

Run this command in the project folder:
```bash
npm install
```

This will download all required packages (may take a few minutes).

**Note:** If `npm install` fails due to dependency conflicts, you may need to use:
```bash
npm install --force
```

### Step 4: Configure the API Token (Required)

The pain detection server is private and requires authentication. You need to create a `.env` file with the access token:

1. Create a file named `.env` in the `pain_detection_app` folder
2. **Reach out to me to get the API token** if you want to clone and run this project
3. Add the token to your `.env` file:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

> **Note:** The server is kept private for security reasons. Contact the repository owner for access credentials.

### Step 5: Install Expo Go on Your Phone

Download the **Expo Go** app on your mobile device:

- **Android**: [Get it on Google Play](https://play.google.com/store/apps/details?id=host.exp.exponent)
- **iOS**: [Download from App Store](https://apps.apple.com/app/expo-go/id982107779)

### Step 6: Start the Development Server

In your terminal (inside the project folder), run:
```bash
npm run start
```

A QR code will appear in your terminal.

### Step 7: Run the App on Your Phone

1. **Make sure your phone and computer are on the same WiFi network**
2. Open the **Expo Go** app on your phone
3. Scan the QR code:
   - **Android**: Use the Expo Go app's built-in QR scanner
   - **iOS**: Use your Camera app to scan the QR code, then tap the notification to open in Expo Go

The app will load on your device!

---

## Recommended: Use a Real Phone

**We strongly recommend using Expo Go on a physical device** rather than an emulator for the best experience. The app looks and performs much better on real hardware, especially for camera and media features.

### Alternative: Emulator/Simulator (Not Recommended)

If you don't have access to a physical device, you can use an emulator, but note that the design and performance may not be optimal:

#### For Android Emulator

1. Install [Android Studio](https://developer.android.com/studio)
2. Set up an Android Virtual Device (AVD) through Android Studio
3. Start the emulator
4. Run: `npm run android`

#### For iOS Simulator (macOS only)

1. Install Xcode from the Mac App Store
2. Run: `npm run ios`

> Note: Emulators may not display the app with the intended design quality and some features may not work properly.

---

## App Features

- Camera functionality for capturing images
- Video recording capabilities
- Media library access
- Audio recording support
- Themed UI with dark/light mode support

---

## Project Structure

```
pain_detection_app/
├── app/                     # Application screens (Expo Router)
│   ├── (pages)/            # Tab navigation screens
│   │   ├── _layout.tsx     # Pages layout
│   │   ├── index.tsx       # Home screen
│   │   ├── camera.tsx      # Camera screen
│   │   ├── record.tsx      # Video recording & pain analysis
│   │   └── library.tsx     # Media library screen
│   └── _layout.tsx         # Root layout
├── assets/                 # Images, fonts, icons
├── components/             # Reusable UI components
│   └── ThemedText.tsx      # Themed text component
├── constants/              # App configuration
│   ├── Colors.ts           # Color themes
│   └── api.ts              # API URLs and tokens
├── services/               # External service integrations
│   └── videoAnalysis.ts    # Pain detection API calls
└── package.json            # Dependencies and scripts
```

---
