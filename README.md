# vision

A small Python project: a vision model for PowerPoint and shortcuts.

Overview

This repository contains code and models aimed at extracting information from PowerPoint slides and providing keyboard shortcuts or automated actions based on visual content. The project is experimental and intended for research and prototyping.

Features

- Analyze slides (images) and detect layout elements (titles, bullets, images, tables).
- Suggest keyboard shortcuts and automation for common slide-editing tasks.
- Export analysis results to JSON for further processing.

Requirements

- Python 3.8+
- Install dependencies from requirements.txt if present: pip install -r requirements.txt

Quick start

1. Clone the repository:

   git clone https://github.com/gurkebaui/vision.git
   cd vision

2. Create a virtual environment and install dependencies:

   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt

3. Run an example script (if available):

   python examples/run_inference.py --input path/to/slide.png --output results.json

If there is no examples/run_inference.py yet, check the repository for scripts or adapt the code in the src/ directory.

Contributing

Contributions are welcome. Please open issues to discuss features or file pull requests with clear descriptions and tests where appropriate.

License

No license is currently specified for this repository. If you want others to reuse this code, consider adding a LICENSE file such as the MIT License.

Contact

Owner: @gurkebaui

Notes

This README is a starter template. I can customize it further to include specific usage instructions, examples, badges, or a more detailed architecture section if you point me to key files or scripts in the repo.