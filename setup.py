from setuppy import setup, find_packages

setup(
    name="license_plate_detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'nomeroff-net',
        'click',
        'pyyaml',
        'tqdm'
    ]
)