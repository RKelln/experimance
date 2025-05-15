"""Setup script for the Experimance package."""
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="experimance",
        version="0.1.0",
        author="Ryan Kelln",
        author_email="ryan.kelln@gmail.com",
        description="Interactive sand-table art installation with AI-generated satellite imagery",
        packages=find_packages(include=["*"]),
        package_data={
            "": ["*.json", "*.toml", "*.yaml", "*.yml"]
        },
        install_requires=[
            "numpy>=1.23",
            "pyzmq>=25.1.1",
            "pydantic>=2.5.0",
            "toml>=0.10.2",
            "python-dotenv>=1.0.0",
            "asyncio>=3.4.3",
            "aiohttp>=3.9.0",
            "uuid>=1.30",
            "Pillow>=9.5",
            "opencv-python>=4.8",
        ],
        extras_require={
            'dev': [
                "pytest>=7.4",
                "pytest-asyncio>=0.23",
                "pytest-mock>=3.12",
                "pytest-cov>=4.1",
                "mypy>=1.7",
                "ruff>=0.1.6",
            ],
            'display': [
                "pyglet>=2.0",
                "PyOpenGL>=3.1.7",
                "PyOpenGL-accelerate>=3.1.7",
                "pysdl2>=0.9.14",
                "pysdl2-dll>=2.0.0",
            ],
            'depth': [
                "pyrealsense2>=2.54",
            ],
            'audio': [
                "python-osc>=1.8",
                "sounddevice>=0.4.6",
            ],
        },
        python_requires=">=3.11",
    )
