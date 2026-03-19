from setuptools import setup, find_packages

sub_packages = find_packages(where="src")

# Build package_dir mapping: each sub-package must be explicitly mapped
# e.g. "deeptalk_asd.face_detector" -> "src/face_detector"
package_dir = {"deeptalk_asd": "src"}
for pkg in sub_packages:
    package_dir[f"deeptalk_asd.{pkg}"] = f"src/{pkg.replace('.', '/')}"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deeptalk_asd",
    version="0.3.0",
    description="DeepTalk Active Speaker Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir=package_dir,
    packages=["deeptalk_asd"] + [f"deeptalk_asd.{pkg}" for pkg in sub_packages],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv",
        "termcolor",
        "tornado",
        "python_speech_features",
        "numpy==1.26.4",
        "opencv-python",
        "scipy",
        "pandas",
        "tqdm",
        "sherpa-onnx",
    ],
)
