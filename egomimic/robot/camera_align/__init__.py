"""Live front-camera alignment monitor for the YAM rollout station.

``python -m egomimic.robot.camera_align.monitor`` serves a loopback dashboard that
compares the rollout dashboard's own front-camera stream against a saved reference
frame and saved ArUco tag poses, and separates camera motion from desk motion.
It never commands a robot; see README.md in this package.
"""
