# CNN-on-the-prediction-of-image

A Convolutional Neural Network (TensorFlow / Keras) that classifies animal photos into **5 classes**:
**Cat, Dog, Horse, Elephant, Butterfly**.

Extended from the original 2-class (Cat / Dog) version.

## Dataset

All photos come from the Kaggle dataset
[Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10).

| Class     | Train | Validation | Total | Test (unseen) |
|-----------|------:|-----------:|------:|--------------:|
| Cat       | 160   | 40         | 200   | 10            |
| Dog       | 160   | 40         | 200   | 10            |
| Horse     | 160   | 40         | 200   | 10            |
| Elephant  | 160   | 40         | 200   | 10            |
| Butterfly | 160   | 40         | 200   | 10            |

```
Dataset/
├── train/       Cat/ Dog/ Horse/ Elephant/ Butterfly/   (160 each)
├── validation/  Cat/ Dog/ Horse/ Elephant/ Butterfly/   (40 each)
└── test/        cat_00.jpg ... butterfly_09.jpg         (10 each)
```

No photo appears in more than one split. Test file names start with the true class, so the notebook
can check the test accuracy automatically.

## Changes from the 2-class version

- 3 new classes (Horse, Elephant, Butterfly), with 200 photos per class.
- `class_mode='categorical'` + `Dense(5, activation='softmax')` + `categorical_crossentropy`
  (instead of binary / sigmoid, which only works for 2 classes).
- **Transfer learning with MobileNetV2**: a CNN pretrained on ImageNet is used as a frozen feature
  extractor, and only a new 5-class output layer is trained. A CNN trained from scratch on 160 photos
  per class only reached about 60% accuracy.
- Data augmentation (rotation, shift, zoom, flip) and Dropout to reduce overfitting.
- EarlyStopping that keeps the best weights.
- Fixed data leakage: in the old dataset, every validation and test photo was also in the training set.
- Fixed prediction: test images are now rescaled by 1/255 like the training images, and the class is
  picked with `argmax`.
- Accuracy is now reported on the validation set (it was measured on the training set before).
- Added a confusion matrix and a classification report.

## Results

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| CNN from scratch | ~60% | 35/50 (70%) |
| **MobileNetV2 transfer learning** | **98.5%** | **50/50 (100%)** |

## How to run

TensorFlow needs Python 3.9 – 3.13.

```bash
pip install -r requirements.txt
jupyter notebook CNN.ipynb
```

Run all cells. The trained model is saved to `models/CNN_predict_model_5class.keras`.
