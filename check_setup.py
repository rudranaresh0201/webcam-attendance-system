"""Verify the environment before running anything else.

    python check_setup.py              # packages + webcam
    python check_setup.py --no-camera  # packages only
"""
import sys


def main():
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        sys.exit("FAIL: Python 3.10+ required.")

    try:
        import cv2
        import numpy
        import pandas
    except ImportError as e:
        sys.exit(f"FAIL: {e}. Run: pip install -r requirements.txt")
    print(f"opencv {cv2.__version__}, numpy {numpy.__version__}, pandas {pandas.__version__}")

    if not hasattr(cv2, "face") or not hasattr(cv2, "CascadeClassifier"):
        sys.exit(
            "FAIL: this OpenCV build has no cv2.face / CascadeClassifier.\n"
            "You probably have plain opencv-python installed. Inside the venv run:\n"
            "  pip uninstall -y opencv-python opencv-python-headless\n"
            "  pip install -r requirements.txt"
        )

    import utils
    from engines import create_engine

    engines = {"classic": create_engine("classic")}
    print("OK: classic backend (Haar cascade + LBPH)")
    try:
        engines["dnn"] = create_engine("dnn")
        print("OK: dnn backend (YuNet + SFace)")
    except RuntimeError as e:
        print(f"--: dnn backend not available. {e}")

    if "--no-camera" in sys.argv:
        return

    try:
        cap = utils.open_camera()
    except RuntimeError as e:
        sys.exit(f"FAIL: {e}")
    try:
        # the first frames are often black while the sensor warms up
        for _ in range(10):
            ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        sys.exit("FAIL: camera opened but returned no frame.")
    frame = utils.prepare_frame(frame)
    print(f"OK: camera frame {frame.shape[1]}x{frame.shape[0]}")
    for name, engine in engines.items():
        print(f"    faces detected by {name}: {len(engine.detect(frame))}")


if __name__ == "__main__":
    main()
