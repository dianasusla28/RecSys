

# oos-rs
Semi-personalized recsys
```
python3 install -r requirements.txt
cd spr/src
PYTHONPATH=. run python3 scripts/preprocess_ml1m.py --data=PATH_TO_ML1M
PYTHONPATH=. run python3  scripts/test_your_function.py --data=PATH_TO_ML1M --db_path=../results/
PYTHONPATH=. run python3  scripts/get_report.py --db_path=../results/
```


## Сравнение моделей и подбор гиперпараметров

## ml_1m
| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| RandomInductiveRecommender |   0.01 |   0.02 |   0.01 |   0.02 | 
TopPopularInductiveRecommender |   0.20 |   0.23 |   0.19 |   0.23 | 
UserKNNRecommender |   0.17 |    0.20 |    0.17 |    0.21 |
ALSRecommender |   0.27 |   0.32 |   0.27 |   0.33 | 
IALSInductiveRecommender |    0.36 |   0.43 |   0.36 |  0.44 |
EASEInductiveRecommender |   **0.42** |   **0.48** |   **0.41** |   **0.49** | 
XGBoostInductiveRecommender |    0.21 |   0.30 |   0.22 |   0.32 |
MultiVAE  | 0.36 |  0.42 |  0.35 |  0.43 |

ALSRecommender: alpha: 26, factors: 61, similarity: 1

IALSInductiveRecommender: alpha: 12, factors: 84

EASEInductiveRecommender: l2_norm: 1698.228

XGBoostInductiveRecommender: num_leaves: 43



## Сравнение метрик близости

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| braycurtis |  0.25 |   0.34 |   0.26 |   **0.36** | 
canberra |   **0.30** |   0.34 |   **0.29** |   0.34 | 
chebyshev |   0.05 |    0.07 |    0.04 |    0.07 |
cityblock |  0.28  |  0.33  |  0.28  |  0.34  | 
correlation |  0.26  |  0.34  |  0.27  |  **0.36**  |
cosine | 0.25   |  0.33  |  0.25  |  0.35  | 
dice |  0.24  |  0.33  |  0.25  |  0.35  |
euclidean  |  0.29  |  0.34  |  0.28  |  0.34  |
hamming  |  **0.30**  |  **0.35**  |  **0.29**  |  0.35  |
jaccard  |  0.22  |  0.32  |  0.24  |  0.34  |
jensenshannon  |  0.25  |  0.33  |  0.26  |  0.35  |
kulsinski  |  0.05  |  0.09  |  0.05  |  0.09  |
kulczynski1  |  0.01  |  0.05  |  0.02  |  0.04  |
matching  |  0.29  |  0.33  |  0.28  |  0.34  |
minkowski  |  0.28  |  0.33  |  0.28  |  0.34  |
rogerstanimoto  |  0.28  |  0.33  | 0.28   |  0.34  |
russellrao  |  0.02  |  0.05  |  0.03  |  0.05  |
seuclidean  |  0.24  |  0.29  |  0.24  |  0.30  |
sokalmichener  |  **0.30**  |  0.34  |  **0.29**  |  0.35  |
sokalsneath  |  0.24  |  0.33  |  0.26  |  **0.36**  |
sqeuclidean  |  0.28  |  0.33  |  0.27  |  0.34  |
yule  |  0.27  |  0.32  |  0.26  |  0.32  |

Гиперпараметры

braycurtis: alpha: 11, factors: 54

canberra: alpha: 6, factors: 58

chebyshev: alpha: 17, factors: 119

cityblock: alpha: 17, factors: 62

correlation: alpha: 8, factors: 54

cosine: alpha: 9, factors: 42

dice: alpha: 19, factors: 33

euclidean: alpha: 14, factors: 55

hamming: alpha: 7, factors: 33

jaccard: alpha: 22, factors: 48

jensenshannon: alpha: 6, factors: 106

kulsinski: alpha: 17, factors: 110

kulczynski1: alpha: 42, factors: 86

matching: alpha: 12, factors: 79

minkowski: alpha: 18, factors: 41

rogerstanimoto: alpha: 17, factors: 54

russellrao: alpha: 6, factors: 40

seuclidean: alpha: 28, factors: 32

sokalmichener: alpha: 8, factors: 37

sokalsneath: alpha: 14, factors: 56

sqeuclidean: alpha: 20, factors: 41

yule: alpha: 12, factors: 34

## Объединение рекомендательных списков

1. Объединение

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommender: canberra, hamming |   0.29 |   0.34 |   0.28 |   0.24 | 
Лучший результат с одной метрикой |  **0.30**  |  **0.35**  |  **0.29**  |  **0.36**  |


| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommender: canberra, hamming, sokalmichener |   **0.30** |   0.34 |   **0.29** |   0.35 | 
Лучший результат с одной метрикой |  **0.30**  |  **0.35**  |  **0.29**  |  **0.36**  |


| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommender: canberra, hamming, sokalmichener, sokalsneath |   0.26 |   0.34 |   0.26 |   **0.36** | 
Лучший результат с одной метрикой |  **0.30**  |  **0.35**  |  **0.29**  |  **0.36**  |

2. Усреднение 

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommender: canberra, hamming, sokalmichener, sokalsneath |   **0.32** |   **0.37** |   **0.31** |   **0.37** | 
Лучший результат с одной метрикой |  0.30  |  0.35  |  0.29  |  0.36  |

Все лучшие метрики по каждому результату

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommender: braycurtis, correlation, canberra, hamming, sokalmichener, sokalsneath |   **0.33** |   **0.39** |   **0.33** |   **0.40** | 
Лучший результат с одной метрикой |  0.30  |  0.35  |  0.29  |  0.36  |

4 метрики лучшие метрики по одному результату

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommender: sokalmichener, hamming, canberra, braycurtis |   **0.31** |   **0.36** |   **0.30** |   **0.36** | 
Лучший результат с одной метрикой |  0.30  |  0.35  |  0.29  |  0.36  |


## Кластеризация

Лучшая комбинация метрик из предыдущего пункта

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommenderKMeans: braycurtis, correlation, canberra, hamming, sokalmichener, sokalsneath  |   0.24  |   0.29 |   0.24 |   0.29 | 

## Сравнение метрик близости для кластеризации

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| braycurtis |  0.29 |   0.35 |   0.29 |   0.37 | 
canberra |   0.29 |   0.34 |   0.28 |   0.35 | 
chebyshev |   0.28 |    0.33 |    0.27 |    0.33 |
cityblock |  0.31  |  0.36  |  0.30  |  0.36  | 
correlation |  0.32  |  0.38  |  0.31  |  0.39  |
cosine | **0.33**   |  **0.39**  |  **0.32**  |  **0.40**  | 
dice |  0.08  |  0.12  |  0.08  |  0.11  |
euclidean  |  0.31  |  0.36  |  0.30  |  0.37  |
hamming  |  0.06  |  0.08  |  0.06  |  0.07  |
jaccard  |  0.12  |  0.16  |  0.13  |  0.17  |
jensenshannon  |  0.06  |  0.08  |  0.06  |  0.07  |
kulsinski  |  0.07  |  0.10  |  0.07  |  0.10  |
matching  |  0.09  |  0.15  |  0.10  |  0.15  |
minkowski  |  0.28  |  0.33  |  0.27  |  0.34  |
rogerstanimoto  |  0.07  |  0.11  | 0.07   |  0.10  |
russellrao  |  0.11  |  0.16  |  0.12  |  0.17  |
seuclidean  |  0.26  |  0.31  |  0.25  |  0.31  |
sokalmichener  | 0.07 |  0.12  |  0.07  |  0.10  |
sokalsneath  |  0.10  |  0.15  |  0.11  |  0.16  |
sqeuclidean  |  0.28  |  0.33  |  0.27  |  0.33  |
yule  |  0.08  |  0.12  |  0.06  |  0.09  |

Гиперпараметры

braycurtis: alpha: 10, factors: 74

canberra: alpha: 7, factors: 85

chebyshev: alpha: 9, factors: 40

cityblock: alpha: 7, factors: 41

correlation: alpha: 8, factors: 49

cosine: alpha: 7, factors: 42

dice: alpha: 43, factors: 62

euclidean: alpha: 7, factors: 35

hamming: alpha: 43, factors: 95

jaccard: alpha: 31, factors: 108

jensenshannon: alpha: 48, factors: 108

kulsinski: alpha: 34, factors: 117

matching: alpha: 44, factors: 39

minkowski: alpha: 11, factors: 67

rogerstanimoto: alpha: 32, factors: 110

russellrao: alpha: 37, factors: 100

seuclidean: alpha: 16, factors: 68

sokalmichener: alpha: 32, factors: 56

sokalsneath: alpha: 49, factors: 104

sqeuclidean: alpha: 14, factors: 49

yule: alpha: 22, factors: 57

## Объединение рекомендательных списков

Лучшие функции близости по всем метрикам

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommenderKMeans: cosine, correlation, euclidean, cityblock  |   0.32  |   0.38 |   **0.32** |   0.39 | 
Лучший результат с одной метрикой |  **0.33**   |  **0.39**  |  **0.32**  |  **0.40**  | 

## Подбор количества кластеров

## Метод K ближайших соседей

Перебор метрик близости

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommender + KNN annoy: angular |  **0.27** |  **0.36** | **0.29** | **0.39** |
| ALSRecommender + KNN annoy: euclidean |  0.22 | 0.27 |  0.21 |  0.27 |
| ALSRecommender + KNN annoy: manhattan |  0.22 | 0.27 |  0.21 |  0.27 |
| ALSRecommender + KNN annoy: hamming |  0.22 | 0.27 |  0.21 |  0.27 |
| ALSRecommender + KNN annoy: dot |  0.04 | 0.10 |  0.05 |  0.09 |


## Подбор K_neighbors

ALSRecommender + KNN annoy: angular

![alt text](https://psv4.userapi.com/c237131/u169642165/docs/d31/39c574eb5bf0/KNN_r50.png?extra=BJ_2yZcl55m3esmj7T9mg9PzUM6hIPzwkPbSTUQn11mog04EwbkmNucRxHnoRRNvLA5sch9GjygKfSZNEDz4uyu6ULiTAKdYNK5_yrLsmKGdWo7YJGf-z9Oyfrp3UZey8gYOitc-60STOdU_3OP6cJSNMA)

![alt text](https://psv4.userapi.com/c237131/u169642165/docs/d16/c8fb690c38b3/KNN_r20.png?extra=oDF0LKYVJbJRm4a5Yg5g7W5uoyqwzV8tP8bpTqJ3aZPdqI1GLEbNuJNjVWyoEMkVDvPybX2oTUqD7RRSkrs52KFv1wQgpMIINI_kcKdY9hT4IR8vX625cC6Z_kPjS1Akpfhd5mJ18tMg5xtDnXPjWV9Hmg)

![alt text](https://psv4.userapi.com/c237131/u169642165/docs/d5/cf734589eadb/KNN_n20.png?extra=x9XljNDkxEr9635sSKmjAE6jO9Ils2oARi33XFB_T0atjLY7fHkE-AO3JEIGpOL_pVas40gSr_HagLyrCGH4u9euatcKZYZ0eddL3year-rNnCTOYFoRIFS4MfJPj6S26LGBwg35JEoWffmo4ORywjQ9pA)

![alt text](https://psv4.userapi.com/c237131/u169642165/docs/d14/be2240c98f82/KNN_n100.png?extra=qEC5pSRwi1XOy6V86rylZ1E0bsDeCTUbHKXbXl2nEilVE9tM2nNpmsVsbHO3suzJjQW8mOwZlpvvaHhdx3n-41SEhljoFdUOw75plxYPFvlNPcT2oo79S5csFtjBJihwU6L-q6VOjdndDRgFcAawZvkseQ)


| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| Лучший результат ALSRecommender + KNN annoy: angular |  **0.35** |  **0.41** | **0.34** | **0.42** |
