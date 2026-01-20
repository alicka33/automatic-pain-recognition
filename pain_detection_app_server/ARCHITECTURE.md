# Architecture Overview

## Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API LAYER (app.py)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  GET /       │  │ GET /health  │  │POST /upload  │  │
│  │   (root)     │  │              │  │   -video     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   SERVICE LAYER                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │        PainDetectionService                       │   │
│  │  • analyze_video(file_path) → results           │   │
│  │  • get_service_info() → config                  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────┐         ┌────────────────────┐   │
│  │  ModelLoader     │         │  FileHandler       │   │
│  │  • load_assets() │         │  • save_video()    │   │
│  │  • is_loaded()   │         │  • validate()      │   │
│  │  • get_helper()  │         │  • cleanup()       │   │
│  └──────────────────┘         └────────────────────┘   │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  INFERENCE LAYER                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │        VideoInferenceHelper                       │   │
│  │  • predict(video_path) → class, probs, frames   │   │
│  │  • process_video() → tensor                     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────┐         ┌────────────────────┐   │
│  │  Model Classes   │         │  Video Processing  │   │
│  │  • BiLSTM        │         │  • MediaPipe       │   │
│  │  • AttentionLSTM │         │  • Frontalization  │   │
│  └──────────────────┘         └────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Request Flow

```
1. Client uploads video
        │
        ▼
2. FastAPI endpoint receives request
        │
        ▼
3. FileHandler validates & saves video
        │
        ▼
4. PainDetectionService.analyze_video()
        │
        ▼
5. VideoInferenceHelper.predict()
        │
        ├─► Extract features (MediaPipe)
        ├─► Preprocess (Euclidean, padding)
        ├─► Run model inference
        └─► Return predictions
        │
        ▼
6. Service formats results + description
        │
        ▼
7. FileHandler cleans up temp file
        │
        ▼
8. FastAPI returns JSON response to client
```
