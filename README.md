[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

# ZigZag-LLM

This repository provides a framework to model Large Language Models on dedicated, single-core accelerators and facilitates early identification of energy bottlenecks within the hardware architecture. It is built upon the [ZigZag](https://github.com/KULeuven-MICAS/zigzag/tree/master) Design Space Exploration tool and inherits its hardware definition format.

> **Energy Cost Modelling for Optimizing Large Language Model Inference on Hardware Accelerators**\
> Robin Geens, Man Shi, Arne Symons, Chao Fang, Marian Verhelst\
> Paper: https://ieeexplore.ieee.org/document/10737844

## Getting started
```
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ python main.py
```

The result should look like this:
![image](https://github.com/user-attachments/assets/453ce332-f3a9-4cfd-af98-bbb578768c9e)


## Speculative decoding

To run speculative decoding simulations you have to run:
python main_speculative_v2.py
python parse_v2_results.py
python plot_v2_results.py

If you want to include standard decoding in the plot to compare it to speculative decoding, you also need to run a simulation and parsing for the standard case by running:
python main_standard.py
python parse_standard_results.py
