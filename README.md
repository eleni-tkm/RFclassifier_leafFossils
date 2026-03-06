# Leaf Classification Using Random Forests
*A **synthetic** morphometric dataset for leaf identification*

---

## 📌 Overview

This repository contains a **synthetic leaf morphometric dataset** (`fossil_leaf_dummy_dataset.csv`) and an accompanying **Random Forest classification pipeline** (`randomForesClassifier.py`).  
The project demonstrates how machine‑learning models can help classify leafs using **quantitative shape descriptors**, **leaf architectural features**, and **encoded botanical qualitative traits**.

This README explains:

- The dataset  
- All included morphological features  
- How normalization addresses the *leaf size problem*  
- Assumptions and limitations  
- How to use the dataset with a Random Forest (RF) classifier
- Key RF metrics
- Improvements

---

## 📁 Dataset Description

The dataset contains **110 synthetic leaf samples** (10 classes × 11 samples per class)
Each row includes **29 morphological and architectural features**, designed to reflect real leaf characters used in scientific research

Because this is synthetic data, the ranges and correlations are designed to **mimic realistic biological patterns**, such as:
- Serrate species having more teeth  
- Lobed species having deeper sinuses  
- Narrow willow‑like leaves showing high aspect ratios  
- LMA and thickness showing positive correlation  

---

## 🌿 Morphological Features


### **A. Basic Morphometry**
| Feature | Description |
|--------|-------------|
| **leaf_length_cm** | Blade length from apex to base |
| **leaf_width_cm** | Maximum blade width — used with length to determine shape indices |
| **leaf_area_cm2** | Estimated leaf area |
| **perimeter_cm** | Estimated outline perimeter -> increases with lobing and serration complexity |
| **aspect_ratio** | Length ÷ width -> distinguishes narrow vs elliptical vs broad leaves |

---

### **B. Petiole & Vein Attributes**
| Feature | Description |
|--------|-------------|
| **petiole_length_cm** | Proportional petiole length often correlates with functional leaf architecture |
| **secondary_vein_count** | Number of major secondary veins|
| **vein_density_mm_per_mm2** | Vein density reflects water transport capacity |

---

### **C. Margin, Lobe & Tooth Morphology (CLAMP‑derived variables)**


| Feature | Description |
|--------|-------------|
| **tooth_count** | Number of teeth along margin - correlates with environmental gradients |
| **tooth_mean_height_mm** | Height of serrations - reflects “degree of serration” |
| **lobe_count** | Number of leaf lobes |
| **sinus_depth_mm** | Depth of indentations between lobes |

---

### **D. Leaf Physical/Economic Traits**
| Feature | Description |
|--------|-------------|
| **thickness_mm** | Leaf thickness — used as a functional ecological trait |
| **LMA_g_m2** | Leaf Mass per Area  |

---

### **E. Encoded Qualitative Traits**

| Feature | Code Meaning |
|--------|--------------|
| **margin_type_code** | 0 Entire, 1 Crenate, 2 Dentate, 3 Denticulate, 4 Double Serrate, 5 Serrate, 6 Lobate, 7 Ciliate, 8 Sinuate, 9 Undulate |
| **apex_type_code** | 0 Acute, 1 Acuminate, 2 Obtuse, 3 Emarginate |
| **base_type_code** | 0 Cuneate, 1 Rounded, 2 Cordate, 3 Truncate |
| **venation_type_code** | 0 Pinnate, 1 Palmate, 2 Parallel |
| **teeth_regularity_code** | 0 Irregular, 1 Regular (CLAMP variable) |
| **teeth_closeness_code** | 0 Distant, 1 Close (CLAMP variable) |
| **lobing_degree_code** | 0 None, 1 Weak, 2 Strong |
| **shape_class_code** | 0 Elliptic, 1 Ovate, 2 Lanceolate, 3 Oblanceolate, 4 Cordate |

---

### **F. Size‑Normalized Features (to remove the size problem)**

Real leaf datasets contain leaves of many sizes (juvenile vs mature leaves).  
To prevent Random Forests from learning “species = size,” we normalize size‑dependent metrics.

| Feature | Formula | Purpose |
|---------|---------|---------|
| **teeth_per_cm** | tooth_count ÷ perimeter | Removes leaf size influence on tooth count |
| **tooth_height_rel** | tooth_height ÷ leaf_length | Normalizes serration size |
| **sinus_depth_rel** | sinus_depth ÷ leaf_width | Size‑independent lobing measure |
| **petiole_to_blade_ratio** | petiole_length ÷ leaf_length | Removes absolute scaling of the petiole |
| **form_factor** | (4πA) ÷ P² | Shape compactness index used in physiognomic analyses |

These ensure that small vs large versions of the same species (e.g., small *Betula*, large *Betula*) do **not** confuse the classifier.

---

## 📘 Important Literature

- [Digital Atlas of Ancient Life](https://www.digitalatlasofancientlife.org/learn/paleoecology/paleoclimate/clamp/)
- https://www.researchgate.net/publication/368787071_Deep_-_Morpho_Algorithm_DMA_for_medicinal_leaves_features_extraction
- https://www.slideserve.com/ctrujillo/leaf-morphology-powerpoint-ppt-presentation
- https://northernontarioflora.ca/?m=L&t=F

## ⚖️ Limitations of This Dataset

This synthetic dataset is designed for **demonstration, model testing, and method development only**.  
Limitations include:

### **1. Synthetic (not empirical)**
- The dataset is algorithmically generated and **does not represent real leafs** exactly

### **2. No leaf preservation artifacts**
Real leaf samples may suffer:
- Margin erosion  
- Compression distortion  
- Incomplete laminae  
- Midrib shearing  

None of these are simulated here

### **3. No missing data**
Real datasets often have:
- Missing apex/base  
- Partial lobes  
- Unmeasurable veins

None of these are simulated here

### **4. No class imbalance**
Real assemblages may contain:
- Many *Quercus* leaves  
- Only a few *Platanus* leaves  
- Biases from depositional environment

This set uses equal sample sizes per species (11 each)

### **5. No ontogenetic or environmental variation in trait plasticity**
Real leaves vary with:
- Light environment  
- Temperature  
- Water availability  

This synthetic set only simulates trait variation statistically

---

## 🚀 Using This Dataset for Random Forest Classification

Example in Python:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("fossil_leaf_dummy_dataset.csv")
X = df.drop(columns=["id", "species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    random_state=42
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```
## Key Metrics
### MODEL PERFORMANCE 
| Metric               | value |
|-----------------------|-----------|
| **Accuracy** | 0.4545 |
| **F1 Macro** | 0.4186 |
| **F1 Weighted** | 0.3935 |

### Poor numbers possible explanations:
- Too many classes for the available data
- High dimensionality vs. small data (~29 features vs. 110 rows)
- Biologically realistic overlap (Many leafs share similar length/width, margins, and tooth metrics)
- Random Forest behavior in multi‑class settings
  - RF favors features that yield the biggest impurity drop (often continuous size features)
  - Some classes end up with no distinct decision region → never predicted (precision/recall = 0)

### CLASSIFICATION REPORT

| label                 | precision | recall | f1-score | support |
|-----------------------|-----------|--------|----------|---------|
| Acer_fossil_A         | 0.00      | 0.00   | 0.00     | 2       |
| Betula_fossil_A       | 0.50      | 1.00   | 0.67     | 2       |
| Ficus_fossil_A        | 0.33      | 0.50   | 0.40     | 2       |
| Lauraceae_fossil_A    | 0.50      | 0.50   | 0.50     | 2       |
| Liquidambar_fossil_A  | 0.00      | 0.00   | 0.00     | 3       |
| Platanus_fossil_A     | 0.50      | 1.00   | 0.67     | 2       |
| Populus_fossil_A      | 1.00      | 0.50   | 0.67     | 2       |
| Quercus_fossil_A      | 0.25      | 0.33   | 0.29     | 3       |
| Salix_fossil_A        | 1.00      | 1.00   | 1.00     | 2       |
| Ulmus_fossil_A        | 0.00      | 0.00   | 0.00     | 2       |

### The Betula case—explained
In the report above, Betula_fossil_A has:
- precision = 0.50
- recall = 1.00
- f1 = 0.67
- support = 2 (only 2 true Betula samples exist in the test set)

This means:
- **Recall 1.00 (100%)** → The model **found all real Betula** samples in the test set (both of them). None were missed
- Precision 0.50 (50%) → Whenever the model said “this is Betula,” it was correct only **half** of the time.
_How is that possible together?_ Because the model **predicted Betula more times than the real Betula** count. For example, if the model predicted Betula 4 times but only 2 of those were truly Betula (the other 2 belonged to other classes), then:
  - True Positives (TP) = 2
  - False Positives (FP) = 2
  - Precision = TP / (TP + FP) = 2 / 4 = 0.5
  - Recall = TP / (TP + FN) = 2 / 2 = 1.0
 
  # Improvements
  - Use stratified K‑fold CV to get a more stable estimate than a single split
  - Reduce the number of classes (e.g., group to family or morphotype groups)
  - Increase samples per class (even synthetic augmentation) so trees learn stable boundaries
  - Constrain size dominance by de-emphasizing raw size features and rely more on normalized architectural traits

# Related Sources:
- [**CLAMP (Climate Leaf Analysis Multivariate Program)**](https://www.digitalatlasofancientlife.org/learn/paleoecology/paleoclimate/clamp/)
- [**Morphyll**](https://www.researchgate.net/publication/322513713_MORPHYLL_A_database_of_fossil_leaves_and_their_morphological_traits)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8702526/
- https://www.sciencedirect.com/science/article/pii/S1574954125003383
- https://onlinelibrary.wiley.com/doi/abs/10.1002/gj.70007
- https://figshare.utas.edu.au/articles/thesis/Old_plants_new_tricks_machine_learning_and_the_conifer_fossil_record/23246396/1?file=40965215
- https://d1wqtxts1xzle7.cloudfront.net/108463103/joc.692120231206-1-68epyu-libre.pdf?1701888050=&response-content-disposition=inline%3B+filename%3DApplication_of_Machine_Learning_Methods.pdf&Expires=1772790953&Signature=OpWLXMMeqop1IiL483Z40z~k0ozVWPclDuwaHNRvc-miHpkcC8UU0~ICAT2brId4Xuh4n96xrU~~Q59m88fkb~NDQ4qFddJ~Ws-On-dyaIheXcksPzBFFt~NHf8xOzS6cEwYJwc~PdDcJ0ju06SPMxVBmAJiPZUS9PZd5drSr0miTkLxM6Kmq8BvjADVJim265yjmnkYKcG9m8QQ7gxy58eZBgyZfj2-ul5wU4nThhaWhGkJF0Ub6e4G970X5Of937zKaPVfJpuPmlwUaky4Qw672h2EYL8v~qTEnwlRlVpkdBmEl16ze8Cp9mmgcQP89DMMtmoDdnvY6Drr1zYHyw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA


