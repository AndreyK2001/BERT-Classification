# Repository for experiments for (thesis) project "Language complexity in multilingual language models"

## Tasks
1. (done) Fine-tune predictor on [NPlus1](https://tatianashavrina.github.io/taiga_site/segments) and on [Cambridge English Readability Dataset](https://ilexir.co.uk/datasets/index.html#tab-60623fc2c01923cc84b)
Datasets are to different (NPlus1 contains texts for native speakers and in Cambridge Dataset texts are for learners). This caused biased predictions (model trained on english texts predicted all russian text to be harder and opposit situation on model trained on russian texts).

2. (done) Fine-tune predictor on one language (english) from ReadMe++ and predict on other.
Success

3. (done) Count ANC on each pair of languages with english
Success. Plots of ANC are considerably different from ones for XNLI task in article [Cross-lingual Similarity of Multilingual Representations Revisited](https://aclanthology.org/2022.aacl-main.15/)

4. Consider other multilingual classification tasks using BERT and perform compare ANC on them with Language complexity and XNLI tasks

5. Test the hypothesis "The higher f1 metric on the prediction, the larger metric ANC in BERT layers"

## Articles and datasources
1. [ReadMe++](https://github.com/tareknaous/readme) - Multilingual dataset on Language complexity marked with CEFR
2. [Cross-lingual Similarity of Multilingual Representations Revisited](https://aclanthology.org/2022.aacl-main.15/) - artical on XNLI where ANC (metric of similarity) was introduced
