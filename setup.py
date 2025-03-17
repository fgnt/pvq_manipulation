from distutils.core import setup

setup(
    name='pvq_manipulation',
    version='0.0.0',
    author='Department of Communications Engineering, Paderborn University',
    author_email='sek@nt.upb.de',
    license='MIT',
    keywords='audio speech',
    install_requires=[
        'torchdiffeq',
        'paderbox @ git+http://github.com/fgnt/paderbox',
        'padertorch @ git+http://github.com/fgnt/padertorch',
        'TTS @ git+https://github.com/coqui-ai/TTS.git@dev#egg=TTS',
    ],
)
