# YouTube Comments Sentiment Analysis and Visualization

## Overview
This project analyzes YouTube comments to derive insights into user sentiments and frequently used emojis. The analysis pipeline involves data cleaning, sentiment polarity computation, and visualization using tools like word clouds and bar charts.

## Dataset
The dataset used is `UScomments.csv`, which contains user comments from YouTube. Due to the large dataset size (~700K rows), sampling is employed where necessary to optimize processing.

## Libraries Used
- **Pandas**: For data manipulation and cleaning.
- **NumPy**: For numerical operations.
- **Matplotlib** & **Seaborn**: For visualizations.
- **TextBlob**: For sentiment analysis.
- **WordCloud**: For generating word clouds.
- **Emoji**: For extracting and analyzing emojis.

## Key Steps

### 1. Data Cleaning
- **Handling Missing Values**: Checked for null values using `.isnull().sum()` and dropped them with `.dropna()`.
- **Removing Duplicates**: Duplicated rows were removed using `.drop_duplicates()`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# If you just want to skip the problematic lines and continue reading the rest of the file, you can use: on_bad_lines='skip'
yt = pd.read_csv(r"UScomments.csv", on_bad_lines='skip')
yt

yt.head(10)

# Data Cleaning
yt.isnull() # It will shows the missing values in tables (Data Frame) as bool

# Here the data are more than 10000, hence isnull() is not use here
# So we can use sum with isnull
yt.isnull().sum() # It will shows how any missing values in each column

# If the missing values are less we can use dropna
yt.dropna(inplace=True)

# Now I am again checking for the missing value, this time is null
yt.isnull().sum()

# Checking rows X Columns now
yt

# Dropping duplicate
yt = yt.drop_duplicates()
yt
print(type(yt))
```

### 2. Sentiment Analysis
- **Polarity Calculation**: Sentiment polarity was calculated using `TextBlob`.
  - Positive: Polarity > 0
  - Neutral: Polarity = 0
  - Negative: Polarity < 0

```python
from textblob import TextBlob

# Single sentence analysis (Warm-up)
TextBlob("Logan Paul it's yo big day ‼︌").sentiment

# To get the polarity alone
TextBlob("Logan Paul it's yo big day ‼︌").sentiment.polarity

# Checking total shape (rows and columns of the given dataset)
yt.shape

# If you want to analyze only a particular range from the given data
sample_df = yt[0:1000] # It takes only 1000 rows from 700K rows of dataset
sample_df.shape
type(yt)

# To execute sentiment polarity for values in dataset we need to use a FOR function
polarity = []  # Reinitialize the list

for comment in yt["comment_text"]:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)

# Assign the polarity values back to the DataFrame
yt['polarity'] = polarity
len(polarity)

# Now need to add this polarity in DF (yt)
yt['polarity'] = polarity

print(type(yt))  # This should return <class 'pandas.core.frame.DataFrame'>
print(type(yt))
yt.head()
```

### 3. Word Cloud Visualization
- **Positive Comments**: Generated a word cloud for comments with positive polarity.
- **Negative Comments**: Generated a word cloud for comments with negative polarity.
- **Stopwords Removal**: Common words like "is", "are" were excluded from the visualization.

```python
from wordcloud import WordCloud, STOPWORDS

# Wordcloud Analysis
# Helps to display all the possible +ve keywords / keywords have +ve polarity (like: Super, Good, Nice, Awesome..)
filter1 = yt["polarity"] == 1
positive_comment = yt[filter1] # it will show only the positive value

filter2 = yt["polarity"] == -1
negative_comment = yt[filter2]

# To install wordcloud
!pip install wordcloud

# Generating the wordcloud for positive comments
total_positive_comment = ''.join(positive_comment["comment_text"])
wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_positive_comment)
```

![Positive_Comments](https://github.com/SriSurya-DA/Senimental_Analysis_For_YouTube_Comments/blob/main/Best_Comments.png)


```python
plt.imshow(wordcloud)
plt.axis('off')

# Now wordcloud for bad comments
total_negative_comment = ''.join(negative_comment["comment_text"])
wordcloud_neg = WordCloud(stopwords=set(STOPWORDS)).generate(total_negative_comment)

plt.imshow(wordcloud_neg)
plt.axis('off')
```

![negative_comments](https://github.com/SriSurya-DA/Senimental_Analysis_For_YouTube_Comments/blob/main/negative_comments.png)

### 4. Emoji Analysis
- Extracted emojis from comments using the `emoji` library.
- Counted occurrences of each emoji using `Counter` from the `collections` module.
- Displayed the top 5 most common emojis in a bar chart.

```python
import emoji
from collections import Counter

emoji_final = []

# Iterating through all comments (for-loop)
for comment in yt["comment_text"]:
    for char in comment:
        if char in emoji.EMOJI_DATA:  # If character emoji list contains
            emoji_final.append(char)  # Append to list

# Now emoji_final contains all emojis from all comments
emoji_counts = Counter(emoji_final).most_common(5)

# Plotting the bar chart
emojis, counts = zip(*emoji_counts)

# Set the font to 'DejaVu Sans' which supports emojis
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(figsize=(10, 6))
plt.bar(emojis, counts, color='skyblue')

# Adding labels and title
plt.xlabel('Emoji', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Top 10 Most Common Emojis', fontsize=15)

plt.show()
```

![Top_Emojies](https://github.com/SriSurya-DA/Senimental_Analysis_For_YouTube_Comments/blob/main/Top_Emojis.png)

### 5. Visualization Highlights
- Word clouds provide insights into frequently used positive and negative terms.
- Bar charts reveal the most frequently used emojis in the comments.

## Installation
To run this project, install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn textblob wordcloud emoji
```


