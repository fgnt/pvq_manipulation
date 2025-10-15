from setuptools import setup, find_packages

setup(
    name='pvq_manipulation',
    version='0.0.1',
    author='Department of Communications Engineering, Paderborn University',
    author_email='sek@nt.upb.de',
    license='MIT',
    keywords='audio speech',
    packages=find_packages(),  # <-- falls du Module im Projekt hast
    install_requires=[
        'torchdiffeq',
        'TTS==0.22.0',
        'padertorch',
        'onnxruntime',
        'creapy @ git+https://gitlab.tugraz.at/speech/creapy.git',
        'ipywidgets',
    ],
)
