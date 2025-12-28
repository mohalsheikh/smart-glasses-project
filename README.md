# 👓 VisionAssist AI

### Intelligent Smart Glasses System for the Visually Impaired

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/AI-Powered-purple?style=for-the-badge" alt="AI Powered">
  <img src="https://img.shields.io/badge/Type-Capstone%20Project-orange?style=for-the-badge" alt="Capstone">
</p>

<p align="center">
  <em>A comprehensive AI-powered assistive technology system designed to enhance independence and safety for blind and visually impaired users through real-time environmental awareness, intelligent navigation, and natural voice interaction.</em>
</p>

---

## 🎯 Project Overview

**VisionAssist AI** is a capstone project that transforms ordinary smart glasses into an intelligent assistant capable of understanding and describing the world around visually impaired users. The system combines state-of-the-art computer vision, natural language processing, and spatial awareness technologies to provide real-time guidance and environmental information through natural voice interaction.

Unlike simple object detection apps, VisionAssist AI acts as a **true intelligent companion**—understanding context, remembering scenes, providing proactive safety alerts, and engaging in natural conversation about the user's surroundings.

---

## ✨ Key Features

### 🔍 Real-Time Scene Understanding

- **Intelligent Object Detection** — Identifies and tracks objects in real-time using advanced AI models, with smart prioritization of what matters most
- **Scene Description** — Provides natural, conversational descriptions of the environment ("You're in a kitchen. There's a table in front of you with a mug on it, and the refrigerator is to your left")
- **Person Detection & Analysis** — Detects people, estimates their distance, analyzes body language, and describes their general appearance and activities
- **Context-Aware Processing** — Understands the difference between indoor/outdoor environments and adjusts its behavior accordingly

### 📖 Advanced Text Recognition

- **Multi-Mode OCR System** — Three reading modes available:
  - _Offline Mode_ — Fast, local text recognition for quick reads
  - _Hybrid Mode_ — Combines local processing with AI enhancement for better accuracy
  - _AI Mode_ — Full scene-aware text understanding for complex documents
- **Document Reader** — Paragraph-by-paragraph reading with navigation (next, repeat, summarize)
- **Smart Text Processing** — Handles signs, labels, documents, screens, and handwritten text
- **Reading Order Intelligence** — Understands proper reading flow for multi-column layouts

### 🧭 Accessible Navigation System

- **Multi-Modal Transport Support**:
  - 🚶 Walking directions with pedestrian-focused guidance
  - ♿ Wheelchair-accessible routing
  - 🚌 Public transit integration
  - 🚗 Rideshare/taxi directions
- **Smart Destination Selection** — Automatically finds the closest option when multiple exist ("Navigate to Starbucks" finds the nearest one)
- **Landmark-Based Directions** — Uses recognizable landmarks for orientation instead of just street names
- **Turn-by-Turn Guidance** — Clear, spoken directions optimized for audio ("In about 50 meters, turn right at the crosswalk")
- **Live GPS Integration** — Real-time position tracking with graceful fallback handling

### 🛡️ Safety & Obstacle Avoidance

- **Real-Time Hazard Detection** — Identifies obstacles, stairs, curbs, poles, and other trip hazards
- **Depth Estimation** — Uses AI-powered depth analysis to estimate object distances
- **Proximity Alerts** — Escalating warnings based on distance ("Chair on your left" → "Chair on your left, close" → "Chair on your left, very close")
- **Trip Hazard Detection** — Specialized detection for step-downs and uneven surfaces
- **Configurable Alert Profiles**:
  - _Indoor Mode_ — Focus on furniture, doors, stairs
  - _Outdoor Mode_ — Emphasis on traffic, curbs, obstacles
  - _Quiet Mode_ — Only critical danger alerts

### 🎙️ Natural Voice Interaction

- **Conversational AI Brain** — Understands natural questions and commands, not just keywords
- **Multi-Intent Understanding** — Handles complex requests ("What's the weather and give me directions to the library")
- **Context Retention** — Remembers conversation history for follow-up questions
- **Smart Transcription** — AI-powered speech correction for noisy environments
- **Natural Speech Output** — Human-like spoken responses optimized for clarity

### 🧠 Intelligent Assistant Capabilities

- **Weather Information** — Current conditions and forecasts with natural descriptions
- **Time & Date** — Spoken time in natural format
- **Question Answering** — Answers questions about visible content (math problems, labels, etc.)
- **Translation Support** — Reads and translates visible text to other languages
- **General Knowledge** — Answers questions using conversational AI

### 💾 Scene Memory System

- **Spatial Memory** — Remembers scenes and objects using AI embeddings
- **"Have I Seen This?" Queries** — Can recall if similar objects or scenes were encountered before
- **Location Tagging** — Associates memories with locations for better recall
- **Scene Classification** — Automatically categorizes scenes (kitchen, office, street, etc.)
- **Context Summaries** — Provides summaries of recent observations

### 📊 Advanced Analytics & Monitoring

- **Real-Time Telemetry** — Logs performance metrics for analysis
- **FPS Monitoring** — Ensures smooth real-time performance
- **Detection Confidence Tracking** — Monitors AI accuracy over time
- **System Health Logging** — Tracks resource usage and errors

---

## 🏗️ System Architecture

The system is built on a modular, event-driven architecture with several interconnected components:

```
┌─────────────────────────────────────────────────────────────────┐
│                      MAIN CONTROLLER                            │
│         Orchestrates all subsystems and manages state           │
└──────────────┬───────────────────────────────────┬──────────────┘
               │                                   │
    ┌──────────▼──────────┐           ┌───────────▼───────────┐
    │   VISION PIPELINE   │           │   VOICE INTERFACE     │
    │  • Camera Handler   │           │  • Voice Listener     │
    │  • Object Detector  │           │  • Speech Engine      │
    │  • Depth Estimator  │           │  • Transcription      │
    │  • Human Analyzer   │           └───────────┬───────────┘
    └──────────┬──────────┘                       │
               │                                  │
    ┌──────────▼──────────┐           ┌───────────▼───────────┐
    │    AI FEATURES      │           │   ASSISTANT BRAIN     │
    │  • Scene AI Client  │◄─────────►│  • Intent Detection   │
    │  • OCR Engine       │           │  • Query Handlers     │
    │  • Scene Memory     │           │  • Context Manager    │
    │  • Proactive Assist │           └───────────────────────┘
    └─────────────────────┘
               │
    ┌──────────▼──────────┐           ┌───────────────────────┐
    │   SAFETY LAYER      │           │   EXTERNAL SERVICES   │
    │  • Obstacle Layer   │           │  • Weather Client     │
    │  • Guidance Engine  │           │  • Navigation Client  │
    │  • Alert System     │           │  • GPS Server         │
    └─────────────────────┘           └───────────────────────┘
```

### Core Design Principles

- **Real-Time Performance** — Optimized for smooth operation on portable hardware
- **Graceful Degradation** — Falls back to simpler methods when advanced features unavailable
- **Thread Safety** — Carefully managed concurrency to prevent audio overlaps and race conditions
- **Battery Awareness** — Efficient processing to maximize mobile battery life
- **Network Resilience** — Handles connectivity issues without crashing

---

## 🎮 User Interaction

### Voice Commands (Examples)

The system understands natural language, so users can speak naturally:

| Category       | Example Commands                                                              |
| -------------- | ----------------------------------------------------------------------------- |
| **Scene**      | "What's in front of me?" / "Describe my surroundings" / "What do you see?"    |
| **Reading**    | "Read this" / "What does this say?" / "Next paragraph" / "Summarize the page" |
| **People**     | "Is anyone here?" / "How many people?" / "Describe the person in front of me" |
| **Navigation** | "Directions to the nearest pharmacy" / "Take me to Central Park" / "Continue" |
| **Safety**     | "What's on my left?" / "Any obstacles ahead?"                                 |
| **Weather**    | "What's the weather like?" / "Will it rain today?"                            |
| **General**    | "What time is it?" / "Solve this math problem" / "What color is this?"        |

### Reading Modes

- **Mode 1 (Offline)** — Fastest, works without internet
- **Mode 2 (Hybrid)** — Balanced speed and accuracy
- **Mode 3 (AI)** — Best accuracy for difficult text

---

## 🔬 Technical Highlights

### AI & Machine Learning

- Multi-model object detection with confidence-based filtering
- Vision-language models for scene understanding
- Pose estimation for human analysis (17-point body, 21-point hands, 468-point face mesh)
- Semantic embeddings for scene memory and similarity matching
- Intent classification using large language models

### Computer Vision

- Real-time video processing with frame skipping optimization
- Multi-engine OCR with reading order reconstruction
- Monocular depth estimation for distance awareness
- Region-of-interest preprocessing for improved accuracy

### Audio Processing

- Voice activity detection and noise handling
- Speech-to-text with context-aware prompting
- Text-to-speech with natural prosody
- Intelligent audio queue management (no overlapping)

---

## 📈 Performance Considerations

The system is designed to balance accuracy with real-time performance:

- Processes key frames selectively to maintain responsiveness
- Caches results to avoid redundant API calls
- Uses progressive disclosure (simple to detailed) based on user needs
- Implements smart cooldowns to prevent alert fatigue
- Gracefully handles network latency and failures

---

## 🎓 Academic Context

This project was developed as a **Capstone Project** demonstrating the application of:

- Computer Vision and Deep Learning
- Natural Language Processing
- Human-Computer Interaction Design
- Assistive Technology Principles
- Real-Time Systems Engineering
- API Integration and Cloud Services

The goal was to create a practical, user-centered solution that could genuinely improve independence and quality of life for visually impaired individuals.

---

## 🔮 Future Roadmap

Potential areas for continued development:

- 🏠 Indoor mapping and spatial memory
- 🔊 Spatial audio for directional cues
- 👥 Face recognition for familiar people
- 📱 Mobile app companion for configuration
- 🌐 Multi-language support expansion
- 🤖 Edge AI optimization for fully offline operation

---

## 📜 License & Usage

This project is an academic capstone submission. The codebase demonstrates proprietary techniques and architectures developed specifically for this research.

---

## 👤 Author

Developed as a capstone project showcasing expertise in AI, computer vision, and assistive technology development.

---

<p align="center">
  <strong>VisionAssist AI</strong> — Seeing the World Through Intelligence
</p>

<p align="center">
  <em>"Technology should enhance human capability, not replace human agency."</em>
</p>
