

# oos-rs
Semi-personalized recsys
```
python3 install -r requirements.txt
cd spr/src
PYTHONPATH=. run python3 scripts/preprocess_ml1m.py --data=PATH_TO_ML1M
PYTHONPATH=. run python3  scripts/test_your_function.py --data=PATH_TO_ML1M --db_path=../results/
PYTHONPATH=. run python3  scripts/test_your_function.py --data=PATH_TO_OTHER_DADASET --db_path=../results/ --N_users=5000
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

# Объединение

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

# Усреднение 

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

# Сравнение метрик близости

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

# Объединение рекомендательных списков

Лучшие функции близости по всем метрикам

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommenderKMeans: cosine, correlation, euclidean, cityblock  |   0.32  |   0.38 |   **0.32** |   0.39 | 
Лучший результат с одной метрикой |  **0.33**   |  **0.39**  |  **0.32**  |  **0.40**  | 

# Подбор количества кластеров

![alt text](https://psv4.userapi.com/c235131/u169642165/docs/d10/be589b4e23a6/KMeans_all_metric_1.png?extra=zABM6evPmazoQlkXocSa6QVWK9bo0-mzLQHrOr35wQYm61gNi5ciJB06Zx4wTiAK-54po9J24cmwRkh_QbiNDik48q5rRYH2rpsXZmjycS2RhRInfnkv4pVlb-QGqU76aLCMU1-snMXKfg59MAYwRKqdS5k)

## Эксперименты с несколькими наборами данных 
| Amazon-book | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
|RandomInductiveRecommender| 0.0000| 0.0005| 0.0000 |0.0007|
|TopPopularInductiveRecommender| 0.0090| 0.0174 |0.0118| 0.0202|
|UserKNNRecommender| 0.0067| 0.0012| 0.0088| 0.0124|
|ALSRecommender| 0.0329| 0.0539| 0.0377| 0.0653|
|MultVAE| 0.0358| 0.0579| 0.0419| 0.0706|

| Gowalla | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
|RandomInductiveRecommender| 0.0003 |0.0008| 0.0004 |0.0013|
|TopPopularInductiveRecommender| 0.0398| 0.0493| 0.0406| 0.0553|
|UserKNNRecommender| 0.0271| 0.0319| 0.0259| 0.0320|
|ALSRecommender| 0.1017| 0.1444| 0.1189| 0.1727|
|VAEInductiveRecommender| 0.1048| 0.1463 |0.1197|0.1839|

| YELP | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
|RandomInductiveRecommender| 0.0004 |0.0010 |0.0007| 0.0013|
|TopPopularInductiveRecommender| 0.0192| 0.0368 |0.0259 |0.0472|
|UserKNNRecommender| 0.0125| 0.0251| 0.0176 |0.0324|
|ALSRecommender |0.0459 |0.0796 |0.0555| 0.0999|
|MultVAE| 0.0472| 0.0818| 0.0570| 0.1003|

## Метод K ближайших соседей

Перебор метрик близости

| | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| ALSRecommender + KNN annoy: angular |  **0.27** |  **0.36** | **0.29** | **0.39** |
| ALSRecommender + KNN annoy: euclidean |  0.22 | 0.27 |  0.21 |  0.27 |
| ALSRecommender + KNN annoy: manhattan |  0.22 | 0.27 |  0.21 |  0.27 |
| ALSRecommender + KNN annoy: hamming |  0.22 | 0.27 |  0.21 |  0.27 |
| ALSRecommender + KNN annoy: dot |  0.04 | 0.10 |  0.05 |  0.09 |

# Подбор числа ближайших соседей

Лучшие результаты ALSRecommender + KNN: angular

| MovieLens dataset | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| Лучший результат ALSRecommender + KNN annoy |  **0.349** |  **0.411** | **0.344** | **0.420** |

| Amazon-book dataset | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| Лучший результат ALSRecommender + KNN annoy |  **0.041** |  **0.067** | **0.045** | **0.073** |

| Gowalla dataset | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| Лучший результат ALSRecommender + KNN annoy |  **0.116** |  **0.160** | **0.128** | **0.190** |

| YELP dataset | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
| Лучший результат ALSRecommender + KNN annoy |  **0.057** |  **0.091** | **0.070** | **0.115** |

# Взвешивание скоров слижайших соседей

|  | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
|MovieLens dataset |0.349 |	0.409 |	0.342 |	0.416|
|Amazon-book dataset| 0.041| 0.063| 0.045| 0.074|
|Gowalla dataset| 0.113| 0.149 |0.124 |0.175|
|YELP dataset| 0.053| 0.092 |0.065 |0.115|

# Блендинг по ближайшим соседям 

|  | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
|MovieLens dataset |0.0612 |	0.1295 |	0.0735 |	0.1402|
|Amazon-book dataset| 0.0199 |   0.0411 |   0.0249 |   0.0511|
|Gowalla dataset| 0.0522 |	0.1007 |	0.0742 |	0.1417|
|YELP dataset|  0.0226 |  0.0590 |   0.0320 |   0.0759|

# Градиентный бустинг

|  | ndcg_20| ndcg_100| recall_20| recall_50 |
|---| ---| ---| ---| --- |
|MovieLens dataset |0.365 |0.423 |	0.364 |0.4356|
|Amazon-book dataset| 0.041 |	0.063 |	0.045 |	0.074 |
|Gowalla dataset| 0.109 |	0.153 |	0.121 |	0.181|
|YELP dataset| 0.046 |	0.084 |	0.061 |	0.110|
