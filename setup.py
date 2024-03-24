from setuptools import setup, find_packages

setup(
    name='CNN_LSTM',  # Replace with your package name
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.0.0',
        'torchvision',
        'Pillow',
        'gdown',
        'os',
    ],
    author='walid',
    author_email='acdsteamwalid@gmail.com',
    description='This pacakge loads a CNN LSTM pretrained on the tst storm and predicts the future specified frames',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-walid/tree/main',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
