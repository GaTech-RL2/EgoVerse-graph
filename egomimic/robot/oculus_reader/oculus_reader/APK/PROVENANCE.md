# Bundled world-frame Quest APK

`teleop-debug.apk` is imported unchanged from the Yam station's pinned RAIL
Quest fork:

- Repository: `https://github.com/rohan-bansal/rohan-gello.git`
- Commit: `e23154aa625fea10d934ffcec6c72c31db95da29`
- Commit subject: `fix apk to be world frame`
- Source path: `oculus_reader/APK/teleop-debug.apk`
- SHA-256: `cc990542fb539d541b0927a7dcce7ede4c3e6c3cb53990ad42c305bb42a05876`
- Size: `7496215` bytes

The corresponding source change is in that commit's
`app_source/Src/OculusTeleop.cpp` and is mirrored in this repository. It emits
the raw controller hand pose in the Quest tracking-origin/world frame under the
established `wE9ryARX` Android log tag. It does not transform the hand pose into
headset coordinates.

The APK is tracked by Git LFS under this repository's existing `*.apk` rule.
