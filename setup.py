from setuptools import setup, find_packages

setup(
    name="syntorch",
    version="0.1.0",
    description="LLM 추론을 위한 합성 트레이스 생성 프레임워크",
    author="Jimmy Lee",
    author_email="jimmylee871124@gmail.com",
    url="https://github.com/jimmylegendary/syntorch",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "networkx>=2.5",
        "matplotlib>=3.3.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
