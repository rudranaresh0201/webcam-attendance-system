"""Download the ONNX models used by the dnn backend (YuNet detector + SFace recognizer).

Not needed for the classic (Haar + LBPH) backend.
"""
import hashlib
import sys
import urllib.request

import config

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    config.YUNET_PATH: (
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    config.SFACE_PATH: (
        f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    for path, (url, expected) in MODELS.items():
        if path.exists() and sha256(path) == expected:
            print(f"OK       {path.name}")
            continue
        print(f"Download {path.name} ...")
        urllib.request.urlretrieve(url, path)
        if sha256(path) != expected:
            path.unlink()
            sys.exit(f"FAIL: checksum mismatch for {path.name}. Try again.")
        print(f"OK       {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
