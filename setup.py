from setuptools import setup, find_packages

setup(
    name='discodec',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'einops',
        'flax',
        'jax',
        'numpy',
        'optax',
        'orbax',
        'Requests'
    ],
    author='Amir Dellali',
    author_email='dellalia@ethz.ch',
    description='Discodec: A neural audio codec for latent music representations.',
    url='https://github.com/ETH-DISCO/discodec',
    classifiers=[
        'Intended Audience :: Developers'
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.9',
)
