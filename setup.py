from setuptools import setup, find_packages

setup(name='excitationbp',
      version='0.1',
      description='A minimal implementation of excitation backprop for PyTorch',
      author='Yuechao Hou',
      author_email=['yuechaohou@gmail.com'],
      license='MIT',
      packages=find_packages(),
      keywords=['deep-learning', 'excitation-backprop', 'visualization', 'interpretability'],
      zip_safe=False)