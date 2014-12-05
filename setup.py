from codecs import open as codecs_open
from setuptools import setup


# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(name='fntools',
      description='Functional programming tools for data processing',
      author=u'Taurus Olson',
      author_email=u'taurusolson@gmail.com',
      url='https://github.com/TaurusOlson/fntools',
      version='1.0',
      py_modules=['fntools'],
      packages=['fntools'],
      zip_safe=False
      )

