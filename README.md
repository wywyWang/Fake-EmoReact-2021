# Fake-EmoReact-2021

Source code of competition in Fake-EmoReact 2021, the shared task of SocialNLP 2021 (in conjunction with NAACL 2021). We won the first place.

## Challenge
Given an unlabeled tweet and its GIF response, the model should predict the label of tweet as fake or real.
The positive samples are tweets with the hashtag #FakeNews where the Covid related ones are included, and the negative samples are from the previous SocialNLP challenge [EmotionGIF](https://sites.google.com/view/emotiongif-2020/).
## Dataset
- train.json
    - 168,521 samples with gold labels, to be used for training the model.
- dev.json
    - 40,487 in Round 1.
    - 45,036 in Round 2 (main track).
- eval.json
    - 88,664 in Round 1.
    - 110,492 in Round 2 (main track).
## Metric
- The metric that will be used to evaluate entries are Macro-Precision, Macro-Recall, and Macro-F1 from Scikit-learn.