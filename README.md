# Hate Speech Detection
Fine-tuning a Llama 3.1 model for hate speech detection

**Warning: Most notebooks and images in this repository contain hateful and offensive language.**

## Overview

This project aims to demonstrate the application of Large Language Models in content moderation, showcasing the effectiveness of a fine-tuned Llama-3 model in a multiclass classification task, where the model distinguishes between hate speech, offensive language and normal (non-offensive) language. 

The baseline model performs modestly on the test set, achieving an overall accuracy of 0.52. The model was then fine-tuned with [LoRA](https://arxiv.org/abs/2106.09685) and showed a significant increase in performance, achieving an overall accuracy of 0.89.

After fine-tuning, we quantized the model using the [GPTQ algorithm](https://arxiv.org/abs/2210.17323), leading to a slight drop in accuracy (0.87).

Ultimately, hate speech detection remains a difficult and nuanced task, largely because the term "hate speech" has varied meanings, with no consistent, clear-cut definition. Furthermore, hate speech is typically considered offensive, but not all offensive language is considered hate speech. This overlap adds another layer of complexity when distinguishing between hate speech and offensive language.

## Dataset

As a result of the many interpretations of what constitutes hate speech, I followed the United Nation's [definition](https://www.un.org/en/hate-speech/understanding-hate-speech/what-is-hate-speech) outlined below:

Hate speech is *“any kind of communication in speech, writing or behaviour, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of who they are, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.”*

The data used for this project was sourced from two datasets which closely aligned with this definition of hate speech. The first dataset was obtained from the *Automated Hate Speech Detection and the Problem of Offensive Language* study by Thomas Davidson et al (Github repository [here](https://github.com/t-davidson/hate-speech-and-offensive-language)). The automated hate speech detection dataset consists of more than 25K tweets categorized as hate speech, offensive language or neither, with tweets with racist and homophobic connotations being more likely to be classified as hate speech and sexist comments as offensive language.

The first dataset is summarized below:

| Column Name | Description |
|-|-|
| count | number of CrowdFlower (CF) users who coded each tweet (minimum is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF)|
| hate_speech | number of CF users who judged the tweet to be hate speech|
| offensive_language | number of CF users who judged the tweet to be offensive.|
| neither |number of CF users who judged the tweet to be neither offensive nor non-offensive.|
| class  | class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither |

For the scope of this project, only the `class` column will be used. The `class` column contains 1430, 4163, and 19190 tweets classified as hate speech, offensive language and neither respectively. The hate speech class is augmented with more tweets from a second dataset to make up for the underrepresentation of the class.

The second dataset, the *measuring hate speech* dataset, was sourced from the UC Berkeley D-lab (dataset link [here](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)). According to this [paper](https://aclanthology.org/2022.nlperspectives-1.11/), this dataset adopts the legal definition of hate speech in the US and uses this for classifying hate speech. The dataset of 39,565 comments was annotated by 8,472 annotators from Amazon Mechanical Turk (a crowdsourcing marketplace), with the annotators rating each comment on 10 different features (e.g. respect, violence, hate), and identifying targeted identity groups (e.g., race, gender, religion). It consists of several columns, but we are most interested in the `hatespeech` column, which contains 3 classes - 0, 1, and 2. 2 is assigned to the most hateful tweets and 0 is neutral (but not necessarily harmless/inoffensive). 

The hate speech class from the automated hate speech detection dataset is augmented with more tweets from a second dataset to make up for the underrepresentation of the class. For this, we select the comments labeled as ‘2’ in the second dataset containing hateful comments. 

The final dataset is of the structure below:

| Column Name | Description |
|-|-|
| Sentiment | Contains 3 sentiment labels: hate, offensive & normal |
| Text | Contains the tweets |

The combined dataset contains 19,190 posts/tweets labeled as “offensive,” 16,342 labeled as “hate,” and 4,163 – “normal.” A visual representation can be seen below:

![class_distribution_of_sentiment_labels](https://github.com/user-attachments/assets/97066726-e0c9-4095-8d7a-111f92297b40)

## Results

### Baseline Model
The test set consists of 300 samples from each of the three classes: hate speech, offensive language, and normal speech. Overall, the base Llama 3.1 model achieves an overall accuracy of 52%. Surprisingly, the model performs decently in classifying normal speech with an F1 – score of 71%, but significantly worse in the other categories. This alludes to the fact that normal speech is easier to identify than hateful and offensive speech mainly due to the absence of vulgarity. Pertinently, the offensive speech class has a high recall of 95%, capturing most of the actual offensive speech texts. However, this comes at the cost of misclassifying any text with a hint of vulgarity or aggressiveness as offensive – including hate speech as seen in the image below:

![confusion_matrix_of_base_llama_3 1_before_finetuning](https://github.com/user-attachments/assets/da04c4e1-f1e9-46bf-8612-0a39af8320df)

An overview of the classification report can be seen below:

| Class           | Precision | Recall | F1-Score | Sample Size |
|----------------|-----------|--------|----------|--------------|
| Hate           | 0.65      | 0.04   | 0.07     | 300          |
| Offensive      | 0.41      | 0.95   | 0.57     | 300          |
| Normal         | 0.93      | 0.57   | 0.71     | 300          |
| **Macro Avg**  | 0.66      | 0.52   | 0.45     | 900          |

**Accuracy:** 0.52 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Total Samples:** 900

### Fine-tuned-only Model

As expected, after the fine-tuning, the trained model achieved strong classification performance as seen in Figure: Confusion Matrix After Fine-Tuning, Table: Fine-Tuned Model. The precision of the “normal” class remains the same as the base model, while the recall of the “offensive” class drops 5%. Nevertheless, there is a significant bump in every other metric and class, including the overall accuracy of the model which jumps from 52% to 89%.
It is also worth noting that most of the misclassified examples are concentrated in the “hate speech” and “offensive” classes. While the fine-tuned model showed a significant improvement in classification of these classes over the base model, it might be easy to assume that there is still some difficulty in distinguishing these two classes. 

![confusion_matrix_after_finetuning](https://github.com/user-attachments/assets/8d484b61-82a1-42f2-b9a5-ef134698ba3a)

### Quantized Model
Lastly, the model quantized with GPTQ shows a slight downgrade in the overall accuracy of the model from 89% to 87%. Likewise, a similar percentage decrease in performance is observed across the other metrics except for the precision of the normal class. This decrease across the board will lead to an increase in false positives and negatives during the classification of hate speech which could be detrimental depending on how sensitive the application is. 

![confusion_matrix_after_quantization](https://github.com/user-attachments/assets/277c4445-a59a-4dec-8b41-1f2bccfd1a6a)

| Class          | Precision (Pre-GPTQ) | Precision (GPTQ) | Recall (Pre-GPTQ) | Recall (GPTQ) | F1-Score (Pre-GPTQ) | F1-Score (GPTQ) |
|----------------|----------------------|------------------|--------------------|----------------|----------------------|------------------|
| Hate           | 0.90                 | 0.88             | 0.82               | 0.78           | 0.86                 | 0.82             |
| Offensive      | 0.84                 | 0.80             | 0.90               | 0.92           | 0.87                 | 0.86             |
| Normal         | 0.93                 | 0.94             | 0.95               | 0.91           | 0.94                 | 0.93             |
| **Macro Avg**  | 0.89                 | 0.87             | 0.89               | 0.87           | 0.89                 | 0.87             |

**Accuracy:** 0.89 (Pre-GPTQ)  0.87 (GPTQ)

### Performance Summary of the Models

| Metrics                   | Model without Quantization | Quantized Model       | Change (%) |
|---------------------------|----------------------------|------------------------|------------|
| **Model Size**            | 16.17 GB                   | 5.34 GB                | -66.95% 🟩 |
| **Inference Time (900 posts)** | 6.73 Minutes                 | 4.56 Minutes              | -32.24% 🟩 |
| **Accuracy**              | 89%                        | 87%                    | -2.25% 🟥  |
| **Total Training Time**   | 154 Minutes                | 154 + 23 Minutes       | +12.99% 🟥 |

Quantization significantly reduces model size and inference time, making models more lightweight and efficient, which is ideal for hardware with limited resources. In this case, inference speed improved by over 70%, allowing faster classification. However, this comes at the cost of a ~2% drop in accuracy, which could be critical in sensitive tasks like hate speech detection, where both false negatives and false positives carry social implications. Additionally, quantization adds to the training time (~23 minutes) and may require more powerful hardware. Ultimately, the trade-offs between speed, size, accuracy, and fairness must be carefully weighed based on the application’s context and sensitivity.

