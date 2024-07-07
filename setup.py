import os

BRANCH = 'r2.0.0rc0'
os.system('pip install wget')
os.system('apt-get -y install sox libsndfile1 ffmpeg')
os.system('pip install text-unidecode')
os.system('pip install "matplotlib>=3.3.2"')
os.system('pip install aiohttp==3.9.2')
os.system('pip install boto3 --upgrade')
os.system(f'python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]')