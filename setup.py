from setuptools import setup, find_packages

setup(
    name="youtube_video_commments_sentiment_analyzer",
    version="0.1",
    package_dir={'': 'src'},
    packages=find_packages(where='src'), 
    entry_points={
        'console_scripts': [
            'start = main:main',
        ],
    },
    install_requires=[
        'googleapiclient.errors', 
        'googleapiclient.discovery', 
        'streamlit'
    ],
)