# MEBeauty: Facial Beauty Analysis using CNNs

This repository contains the implementation of our paper on Facial Beauty Analysis Using Distribution Prediction and CNN Ensembles.

## Installation

1. Download the MEBeauty.zip file:
   ```
   https://drive.google.com/file/d/1W7T6Ed3iH3vzfrlZ0r-FuV4OC9pM1Mw4/view?usp=sharing
   ```

2. Extract it inside the project root folder

3. Set up the Python environment:
   ```bash
   # Create a conda environment (Python 3.7 recommended)
   conda create --name beauty python=3.7
   conda activate beauty
   
   # Install required packages
   pip install -r requirements.txt
   ```

## Usage

The `train_test.ipynb` notebook contains both training and inference code for the facial beauty analysis models.

## Model Architecture

Our approach employs convolutional neural networks and ensemble learning techniques to predict facial beauty scores as probability distributions rather than single values. This method more effectively captures the subjective nature of beauty perception.

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@INPROCEEDINGS{10387332,
  author={Ibrahim, Ahmed Aman and Ugail, Noah Hassan and Jayatileke, Tazkia Hoodh and Saffery, Millie Hope and Ugail, Hassan},
  booktitle={2023 15th International Conference on Software, Knowledge, Information Management and Applications (SKIMA)}, 
  title={Facial Beauty Analysis Using Distribution Prediction and CNN Ensembles}, 
  year={2023},
  volume={},
  number={},
  pages={130-135},
  keywords={Deep learning;Computational modeling;Transfer learning;Predictive models;Convolutional neural networks;Task analysis;Faces;Facial Beauty Prediction;Discrete Probability Distribution;Convolutional Neural Networks;Ensemble Learning;Earth Mover's Distance},
  doi={10.1109/SKIMA59232.2023.10387332}
}
```

## Contact

For any questions about these models or the paper, please contact:
- Ahmed Aman Ibrahim - ahmedamanibrahim@gmail.com