License Plate Detection System
============================

A robust system for real-time license plate detection and recognition.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   configuration
   development
   troubleshooting

Features
--------
- Real-time license plate detection
- Multi-format video and image processing
- Database storage for detections
- Performance monitoring and statistics
- Configurable visualization options
- CLI and programmatic interfaces

Quick Installation
----------------
.. code-block:: bash

   git clone https://github.com/yourusername/license-plate-detector.git
   cd license-plate-detector
   pip install -r requirements.txt

Quick Usage
----------
.. code-block:: bash

   # Process an image
   python main.py --mode image --source path/to/image.jpg

   # Process a video
   python main.py --mode video --source path/to/video.mp4

   # Use camera feed
   python main.py --mode camera --camera-id 0

Getting Started
-------------
Check out the :doc:`quickstart` guide for more detailed instructions on getting
started with the system.

Contributing
-----------
Interested in contributing? Check out the :doc:`development` guide.