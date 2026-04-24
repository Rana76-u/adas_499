# adas_499

A new Flutter project.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## Commit - 

- Road Signs:
  - Set bounding box color to Blue for predefined traffic signs
  - Display contextual driver guidance messages based on detected sign
  - Integrate GPS-based speed detection and show current speed on left side of screen
  - Add overspeed warning logic (e.g., alert when driver exceeds detected speed limit)

- Vehicles & Pedestrians:
  - Apply Red bounding box for high-risk objects
  - Apply Orange for medium-risk objects (existing behavior retained)
  - Apply White (thin box, no camera line) for low-risk/other objects

- Road Damage:
  - Set bounding box color to Yellow for potholes and cracks
  - Display warning message describing damage type and recommended action

- UI/Detection Adjustments:
  - Disable distance, direction, speed, trajectory prediction, and TTC for road signs and road damage
  - Restrict display to bounding box and label only for these categories