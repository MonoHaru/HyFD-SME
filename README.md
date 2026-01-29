# Deep Hybrid Model for Fault Diagnosis of Ship's Main Engine

Official implementation of:

📄 [Deep Hybrid Model for Fault Diagnosis of Ship's Main Engine](https://www.mdpi.com/2077-1312/13/8/1398)

📰 JMSE (Journal of Marine Science and Engineering), MDPI, 2025

[\[PDF\]](src/jmse-13-01398-v2.pdf)

## 🧑‍🤝‍🧑 Authors

**First Authors (Equal Contribution)**
- [Tae-Gyeong Kim](https://github.com/MonoHaru), Department of Artificial Intelligence, Sejong University, Seoul 05006, Republic of Korea, [ktk23114418@sju.ac.kr](mailto:ktk23114418@sju.ac.kr)
- [Se-Ha Kim](https://github.com/), Department of Artificial Intelligence, Sejong University, Seoul 05006, Republic of Korea, [23114417@sju.ac.kr](mailto:23114417@sju.ac.kr)

**Second Author**
- [Junseok Lee](https://github.com/), Artificial Intelligence Laboratory, Okestro Co., Ltd., Seoul 07335, Republic of Korea, [js.lee6@okestro.com](mailto:js.lee6@okestro.com)

**Third Author**
- [Hyoung-Kyu Song](https://github.com/), Department of Information and Communication Engineering, Sejong University, Seoul 05006, Republic of Korea, [songhk@sejong.ac.kr](mailto:songhk@sejong.ac.kr)

**Fourth Author**
- [Hyoung-Kyu Song](https://github.com/), Department of Convergence Engineering for Intelligent Drone, Sejong University, Seoul 05006, Republic of Korea, [songhk@sejong.ac.kr](mailto:songhk@sejong.ac.kr)

**Fifth Author**
- [Hyeonjoon Moon](https://github.com/), Department of Computer Science and Engineering, Sejong University, Seoul 05006, Republic of Korea, [hmoon@sejong.ac.kr](mailto:hmoon@sejong.ac.kr)

**Corresponding Author**
- [Chang-Jae Chun](https://github.com/), Department of Data Science and Artificial Intelligence, Sejong University, Seoul 05006, Republic of Korea, [cchun@sejong.ac.kr](mailto:cchun@sejong.ac.kr)


## 💡 Abstract
##### Ships play a crucial role in modern society, serving purposes such as marine transportation, tourism, and exploration. Malfunctions or defects in the main engine, which is a core component of ship operations, can disrupt normal functionality and result in substantial financial losses. Consequently, early fault diagnosis of abnormal engine conditions is critical for effective maintenance. In this paper, we propose a deep hybrid model for fault diagnosis of ship main engines, utilizing exhaust gas temperature data. The proposed model utilizes both time-domain features (TDFs) and time-series raw data. In order to effectively extract features from each type of data, two distinct feature extraction networks and an attention module-based classifier are designed. The model performance is evaluated using real-world cylinder exhaust gas temperature data collected from the large ship low-speed two-stroke main engine. The experimental results demonstrate that the proposed method outperforms conventional methods in fault diagnosis accuracy. The experimental results demonstrate that the proposed method improves fault diagnosis accuracy by 6.146% compared to the best conventional method. Furthermore, the proposed method maintains superior performanceeven in noisy environments under realistic industrial conditions. This study demonstrates the potential of using exhaust gas temperature using a single sensor signal for data-driven fault detection and provides a scalable foundation for future multi-sensor diagnostic systems.

##### Keywords: attention mechanism; deep learning; degradation; exhaust gas temperature; fault diagnosis; feature fusion; hybrid model; marine main engine; time-domain feature


## ✨ Contributions
- ##### We propose a hybrid model for fault diagnosis of a ship’s main engine. Since the proposed hybrid model consists of two separate feature extractors for time-series raw data and TDF, it can effectively extract features that lead to achieving high fault diagnosis accuracy.
- ##### We analyzed the performance of the proposed model by additionally considering the environment with noise signals. We demonstrated through simulation that the performance of the proposed model is better than the existing methods even in noisy environments.
- ##### In order to evaluate the performance of the proposed hybrid model, we created training data by simulating six main engine abnormal classes according to the degree of equipment degradation based on the actual data collected from a two-stroke ship diesel engine. We trained and verified our proposed model using the data created based on the actual collected data.


## 🧭 Overview
<img src="https://github.com/MonoHaru/HyFD-SME/blob/main/assets/overview.png" alt="overview" width="800">

## 📁 Datasets
1. `0_percent_overlapping.csv`
2. `10_percent_overlapping.csv`
3. `20_percent_overlapping.csv`
4. `30_percent_overlapping.csv`
5. `40_percent_overlapping.csv`
6. `50_percent_overlapping.csv`


## 🚀 Train
`python train.py --overlap_percentage [OVERLAP_%] --snr [SNR_dB]`

## 🧪 Test
`python test.py --overlap_percentage [OVERLAP_%] --snr [SNR_dB] --model_name [MODEL_NAME]`

**📌 Note**
- `--overlap_percentage`: Indicates the percentage of overlapping segments in the dataset, which is determined based on the degree of equipment degradation in the main engine.
- `--snr` : Specifies the Signal-to-Noise Ratio (SNR) level, representing the amount of noise added to the data to simulate realistic industrial conditions.


## 🎯 Results
#### Table 1. Fault diagnosis accuracy (%) results for four conventional models and six overlap percentages.
| Model                    |     0%     |     10%    |     20%    |     30%    |     40%    |     50%    |
| :----------------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| Random Forest Classifier |   98.426   |   96.775   |   95.037   |   92.219   |   89.902   |   88.234   |
| CNN+BiGRU                |   97.842   |   95.215   |   92.379   |   87.154   |   85.151   |   79.682   |
| CNN+BiLSTM+Attention     |   98.345   |   95.140   |   92.218   |   91.378   |   88.584   |   83.713   |
| RSCB+ViT                 |   97.113   |   94.264   |   91.671   |   84.737   |   81.430   |   73.781   |
| **HyFD-SME (Ours)**      | **99.148** | **98.976** | **98.293** | **97.574** | **96.857** | **94.380** |


#### Table 2. Fault dignosis accuracy (%) results with different SNRs (dB) in 10% overlapping percentage.
| Model                    |    -4 dB   |    -2 dB   |    0 dB    |    2 dB    |    4 dB    |
| :----------------------- | :--------: | :--------: | :--------: | :--------: | :--------: |
| Random Forest Classifier |   80.727   |   81.791   |   83.535   |   85.125   |   87.334   |
| CNN+BiGRU                |   80.110   |   81.505   |   83.471   |   85.083   |   86.800   |
| CNN+BiLSTM+Attention     |   81.693   |   83.646   |   86.019   |   87.846   |   89.586   |
| RSCB+ViT                 |   79.339   |   85.955   |   82.150   |   85.739   |   86.551   |
| **HyFD-SME (Ours)**      | **97.292** | **97.323** | **97.584** | **97.880** | **98.337** |


## 📜 License
The code in this repository is released under the MIT License.


## 📖 BibTex
```
@article{kim2025deep,
  title={Deep Hybrid Model for Fault Diagnosis of Ship’s Main Engine},
  author={Kim, Se-Ha and Kim, Tae-Gyeong and Lee, Junseok and Song, Hyoung-Kyu and Moon, Hyeonjoon and Chun, Chang-Jae},
  journal={Journal of Marine Science and Engineering},
  volume={13},
  number={8},
  pages={1398},
  year={2025},
  publisher={MDPI}
}
```
