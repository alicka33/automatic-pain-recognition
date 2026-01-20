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

### Step 2: Download the Project

1. Download or clone this repository to your computer
2. Open a terminal/command prompt
3. Navigate to the project folder:
   ```bash
   cd mobile_pain_reco
   ```

### Step 3: Install Project Dependencies

Run this command in the project folder:
```bash
npm install
```

This will download all required packages (may take a few minutes).

### Step 4: Install Expo Go on Your Phone

Download the **Expo Go** app on your mobile device:

- **Android**: [Get it on Google Play](https://play.google.com/store/apps/details?id=host.exp.exponent)
- **iOS**: [Download from App Store](https://apps.apple.com/app/expo-go/id982107779)

### Step 5: Start the Development Server

In your terminal (inside the project folder), run:
```bash
npm run start
```

A QR code will appear in your terminal.

### Step 6: Run the App on Your Phone

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

## Available Commands

| Command | Description |
|---------|-------------|
| `npm start` | Start the Expo development server |
| `npm run android` | Run on Android emulator/device |
| `npm run ios` | Run on iOS simulator (macOS only) |
| `npm run web` | Run in web browser |
| `npm test` | Run tests |
| `npm run lint` | Check code quality |

---

## Project Structure

```
mobile_pain_reco/
├── app/                     # Application screens (Expo Router)
│   ├── (pages)/            # Tab navigation screens
│   │   ├── index.tsx       # Home screen
│   │   ├── camera.tsx      # Camera screen
│   │   ├── record.tsx      # Recording screen
│   │   └── library.tsx     # Media library screen
│   └── _layout.tsx         # Root layout
├── assets/                 # Images, fonts, icons
├── components/             # Reusable UI components
├── constants/              # Colors and constants
└── android/                # Native Android code
```

---

## Required Permissions

The app will request these permissions when you first use the features:

- **Camera** - To take photos and videos
- **Microphone** - To record audio
- **Photo Library** - To save and access media files

