Diabetic Retinopathy Classification with Retinal Vessel Segmentation
👁️ Project Overview

Diabetic Retinopathy (DR) is a leading cause of blindness among working-age adults, caused by progressive damage to retinal blood vessels. In this project, we explore whether explicit retinal vessel segmentation features can enhance DR classification performance.

We design a two-pronged deep learning pipeline:

Baseline Classification – Robust models trained directly on fundus images.

Feature-Augmented Classification – Incorporating retinal vessel maps as an additional feature.

Our experiments were conducted on the APTOS 2019 dataset, using EfficientNet-B3, EfficientNet-B4, and Swin Transformer architectures.

⚙️ Methodology
🔹 1. Retinal Vessel Segmentation

Baseline Model: U-Net with Dice + Focal + Weighted BCE loss → Dice = 0.679.

Improved Model: U-Net++ with EfficientNet-B5 encoder → Dice = 0.8214.

These vessel maps were later fused with fundus images for classification.

📌 Released Dataset:
We also provide our segmentation outputs for the APTOS 2019 dataset, generated with our U-Net++ model:
👉 Segmentation Dataset on Kaggle

This dataset contains pre-computed vessel masks that can be directly used by other researchers for classification or further segmentation experiments.

🔹 2. Classification Pipelines

Baseline Pipeline

Weighted Ordinal Focal Loss to handle class imbalance and ordinal labels.

Strong augmentations + Mixup for generalization.

Fine-tuning with a hybrid loss: Smooth Kappa + Ordinal Loss (direct QWK optimization).

Feature-Augmented Pipeline

Fundus image + vessel segmentation mask.

Fusion strategies:

4-channel input (RGB + vessel mask).

Dual-stream architecture (parallel processing + late fusion).

🔹 3. Evaluation Metric

Quadratic Weighted Kappa (QWK) – industry standard for ordinal DR grading.

📊 Results
Model	Segmentation	QWK	Accuracy	Precision	Recall	F1-score
EfficientNet-B3	No	0.92	0.83	0.85	0.83	0.84
EfficientNet-B4	No	0.91	0.81	0.85	0.81	0.81
Swin Transformer	No	0.91	0.77	0.84	0.77	0.79
EfficientNet-B3	Yes	0.91	-	-	-	-
EfficientNet-B4	Yes	0.88	0.70	0.82	0.70	0.72
Swin Transformer	Yes	0.89	0.75	0.81	0.75	0.76

➡️ Best Single Model: EfficientNet-B3 (QWK = 0.921, Accuracy = 83.1%).
➡️ Segmentation features showed promise but did not surpass the baseline due to segmentation quality limits (Dice = 0.8214).

🔬 Key Insights

Vessel segmentation is a clinically grounded feature, but classifier improvements depend heavily on segmentation fidelity.

Larger models (EfficientNet-B4, Swin) were data-hungry and underperformed on APTOS compared to EfficientNet-B3.

Ensembling did not improve results due to high correlation between models.

Our released segmentation dataset enables other researchers to build upon this work without retraining segmentation models.

📚 Benchmarking Against Literature
Method/Model	QWK	Accuracy
Our Best Model (EfficientNet-B3)	0.921	83.1%
Wahab Sait A.R. et al. (2023)	0.911	98.0%
Karki et al. (2021)	0.901	89.1%
Kobat et al. (2022)	0.864	84.9%
Ishtiaq et al. (2023)	0.856	95.2%

➡️ Our model ranks at the top tier of QWK performance on APTOS 2019.

📌 References

Wahab Sait A.R. A Lightweight Diabetic Retinopathy Detection Model Using a Deep-Learning Technique. Diagnostics, 2023.

Karki S.S., Kulkarni P. Diabetic Retinopathy Classification using a Combination of EfficientNets. IEEE ESCI, 2021.

Tariq M., Palade V., Ma Y. Transfer Learning based Classification of Diabetic Retinopathy. MICAD, 2022.

Kobat S.G. et al. Automated Diabetic Retinopathy Detection Using DenseNet. Diagnostics, 2022.

Ishtiaq U. et al. Hybrid Technique for Diabetic Retinopathy Detection. Diagnostics, 2023.

🚀 Future Work

Improve segmentation model beyond Dice = 0.82 to unlock full feature potential.

Explore larger datasets to better leverage Swin Transformer & EfficientNet-B4.

Investigate multi-task learning (joint segmentation + classification).

✨ Developed by: Abdallah Yasser Abdallah, Begad Mohamed, Youssef Khaled Amer

https://www.kaggle.com/begadmohamedhamdy
https://www.kaggle.com/youssefkhaled04
https://www.kaggle.com/abdallahshady
