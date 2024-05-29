from setuptools import setup, find_packages

setup(
    name='HammingEncoder',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'scikit-learn',
        'pandas',
        'tqdm'
    ],
    author='Junjie Dong',
    author_email='jd445@qq.com',
    description='A Hamming Encoder package',
    url='https://github.com/jd445/HammingEncoder',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
