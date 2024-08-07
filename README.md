# Amazon-Sentiment-Analyzer

!["C:\Users\hp\Desktop\Olailemi Portfolio\html5up-solid-state\images\Sentiment.png"]

## Introduction:

In today's digital age, e-commerce platforms face the challenge of analyzing vast amounts of customer feedback to accurately understand product sentiment. This understanding is crucial for businesses to make informed decisions about product improvements, marketing strategies, and overall customer satisfaction. However, manually analyzing thousands of product reviews is both time-consuming and inefficient.
The goal of this project is to develop an automated sentiment analysis solution to efficiently process and interpret Amazon e-commerce reviews. By leveraging Natural Language Processing (NLP) techniques, we aim to classify customer reviews into sentiment categories such as positive, neutral, and negative. This automated approach will help businesses quickly gauge customer opinions, identify trends, and respond proactively to customer needs.
This report details the methodologies and processes undertaken by Team Bravo to achieve this goal, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment. The result is a robust sentiment analysis tool that provides valuable insights into customer sentiment for Amazon e-commerce reviews.

## Project Workflow Overview:

### Jupyter Notebook:

Jupyter Notebook was employed for the initial stages of the project, including data preprocessing, exploratory data analysis (EDA), sentiment extraction, and model training.

#### Importing Libraries:

The following libraries were imported to facilitate data manipulation, visualization, and model training:
-	Pandas: For data manipulation and analysis.
-	NumPy: For numerical operations.
-	Matplotlib and Seaborn: For data visualization.
-	TextBlob: For sentiment analysis.
-	Scikit-learn: For machine learning model implementation.
-	Joblib: For saving the trained model.

#### Making a copy of the Data:

In the course of data analysis, preserving the integrity and safety of the data is of utmost importance. To achieve this, an initial step involved creating a duplicate of the original dataset. This precautionary measure ensures that the original data remains unaltered, allowing for modifications and preprocessing to be carried out on the copied version.
The process began with loading the dataset into a DataFrame. Once the data was loaded, a copy of the DataFrame was created. This copy served as a backup and was used for various stages of the analysis, including preprocessing and exploratory data analysis (EDA). By working with the duplicate, any changes or transformations applied to the data did not affect the original dataset, preserving its initial state for reference and potential future use.
To confirm the accuracy of the copied dataset, basic checks were performed to compare the dimensions and summary statistics of both the original and the copied DataFrames. This verification process ensured that the copy was a faithful representation of the original data, maintaining data consistency.
The copied dataset was then saved to a new file, providing a reliable backup that could be accessed or restored if needed. This step was crucial for data management, ensuring that the original dataset remained intact while enabling further analysis and experimentation on the duplicate.
Additionally, implementing version control for datasets was considered, especially for larger projects, to track changes over time and facilitate reverting to previous versions if necessary. It was also essential to ensure compliance with data privacy regulations, protecting any sensitive information in the dataset.
Automating the data copying and backup process as part of the overall data processing workflow was recommended. Regular backups provide a safeguard against unexpected data loss or corruption, ensuring that the analysis remains robust and the data secure.

#### Understanding the Data:

The dataset under analysis comprises 46,100 rows and 7 columns, each providing different aspects of customer reviews. Here is a detailed description of each column and observations regarding the data:
-	ID: This column contains 46,100 entries with a data type of int64. Each entry represents a unique identifier for a review, ensuring that each row can be distinctly referenced.
-	Username: This column includes 46,096 entries with a data type of object. It records the usernames of the reviewers. A small number of entries (4 missing) are not provided, indicating some data gaps in this field.
-	Location: This column has 46,088 entries and a data type of object, capturing the geographical location of the reviewers. It also exhibits missing data, with 12 entries absent.
-	Total Review: This column contains 46,100 rows and is currently stored as an object data type. It was expected to be in integer format to accurately reflect the total number of reviews. This inconsistency needs to be addressed to facilitate proper numerical analysis.
-	Date of Experience: This column, which has 46,100 rows, is recorded as an object data type. It represents the date when the experience occurred. Ideally, this column should be converted to a datetime format to enable time-based analysis.
-	Content: This column has 36,717 entries and is of object data type. It holds the textual content of the reviews. Notably, there is a significant amount of missing data in this column, with 9,383 entries missing.
-	Rating: This column features 46,100 entries with an int64 data type, capturing the rating given by the reviewers. This column is complete and correctly formatted for numerical analysis.
Upon reviewing the dataset, several key observations were made:
-	Missing Data: There are gaps in the 'Content', 'Username', and 'Location' columns. Addressing these missing values is essential to ensure the completeness and accuracy of the analysis.
-	Data Type Inconsistencies: The 'Total Review' column is currently of object data type but should be converted to int for accurate analysis. Similarly, the 'Date of Experience' column needs conversion from object to a datetime format to facilitate time-based operations.
-	Completeness: The 'Content' column has a substantial amount of missing data, which may impact the quality of sentiment analysis and other textual analyses.

#### Data Cleaning:

Data cleaning was a critical step in preparing the dataset for analysis. The objective was to address missing values, correct data type inconsistencies, and preprocess textual data to ensure its quality and usability. Below is a detailed description of the cleaning procedures applied to the dataset:
•	Handling Missing Values
The 'Content' column, which contains textual reviews, had several missing entries. To address this, the missing values in this column were filled with the placeholder string 'no review'. This approach provided a uniform value for entries where reviews were not available, ensuring that the dataset remained complete and consistent.
•	Data Type Conversion
The 'Date of Experience' column was initially in an object data type, which included mixed formats. To facilitate time-based analysis, this column was converted to a datetime format. This conversion handled the varying date formats present in the dataset, enabling accurate chronological operations and analyses.
•	Cleaning the 'Total Review' Column
The 'Total Review' column initially contained textual representations such as "review" and "reviews," which were inconsistent with the numeric nature of the data. These words were removed, and the numbers were converted to a numeric format. This conversion ensured that the data type was consistent with the intended use for quantitative analysis.
•	Text Preprocessing
The 'Content' column underwent extensive text preprocessing to enhance the quality of the textual data. The following steps were performed:
-	Contractions Expansion: Expanded contractions (e.g., "won't" to "will not") to standardize the text.
-	Stop Words Handling: Added new stop words and removed specific ones (e.g., "not," "no," "but," "won't") to refine the text processing.
-	Text Cleaning: Removed HTML tags, URLs, and punctuation marks from the text. This step involved converting the text to lowercase to maintain consistency.
-	Tokenization and Lemmatization: Tokenized the text into individual words and lemmatized them to their base forms, which helps in reducing variations of words to a single form.
-	Text Joining: After tokenization and lemmatization, the words were joined back into a single string for uniformity.
A new column named cleaned_review was created to store the preprocessed text. The original 'Content' column was then dropped to finalize the cleaning process.
By applying these cleaning techniques, the dataset was effectively prepared for subsequent analysis. The cleaned text ensured higher accuracy in sentiment analysis and other textual analyses, while the consistency in data types facilitated more reliable and efficient data processing.

#### Data Preprocessing:

Here we will be executing some steps which will effectively transformed raw review data into actionable insights, providing a deeper understanding of customer sentiment and enabling more targeted analysis and decision-making.
Sentiment extraction was a crucial phase in analyzing customer reviews to gauge overall sentiment. This process involved leveraging the TextBlob library to analyze the textual content of reviews, creating a sentiment column, and conducting various analyses to derive actionable insights. Here is a detailed description of the sentiment extraction process and subsequent analyses:
•	Sentiment Analysis Using TextBlob
TextBlob was employed to perform sentiment analysis on the 'cleaned_review' column. TextBlob's sentiment analysis capabilities allowed us to extract sentiment scores, which include polarity (ranging from -1 for negative to 1 for positive) and subjectivity (ranging from 0 for objective to 1 for subjective). These scores were used to categorize each review into sentiment labels: positive, neutral, or negative.
•	Creation of Sentiment Column
Based on the sentiment scores generated by TextBlob, a new column named sentiment was created. This column mapped the polarity and subjectivity scores to categorical sentiment labels:
-	Positive: Reviews with high polarity scores.
-	Neutral: Reviews with scores around zero.
-	Negative: Reviews with low polarity scores.
This categorization facilitated a clearer understanding of the sentiment distribution within the dataset.
•	Univariate Analysis of Sentiments
A univariate analysis was conducted to explore the distribution of sentiment labels across the dataset. This analysis provided insights into the overall sentiment trends, such as the proportion of positive, neutral, and negative reviews. It helped in understanding the general sentiment of customer feedback and identifying predominant sentiment trends.
•	Bivariate Analysis 
-	Date of Experience by Sentiments
A bivariate analysis was performed to examine how sentiments varied over time. By analyzing the 'Date of Experience' in relation to sentiment labels, trends and patterns in sentiment over different periods were identified. This analysis provided insights into how customer sentiment might have shifted over time and revealed any temporal patterns or anomalies.
-	Sentiments by Ratings
The sentiment data was also analyzed in conjunction with the ratings given by reviewers. This analysis explored how sentiment varied with different rating levels, helping to understand whether higher ratings corresponded with more positive sentiments and lower ratings with more negative sentiments.
-	Average Total Review by Sentiment
An analysis was conducted to determine the average 'Total Review' count associated with each sentiment label. This metric provided insights into the volume of reviews corresponding to different sentiment categories and helped assess the relationship between sentiment and review volume.
•	Final Data Preparation
After performing the sentiment extraction and related analyses, all columns except cleaned_review and sentiment were dropped. This streamlined dataset focused solely on the preprocessed review content and the derived sentiment labels, preparing it for model training.

#### Model Training and Building:

Model Training and Building
In this phase, the objective was to develop and evaluate machine learning models for sentiment analysis based on the preprocessed review data. The focus was on training models to accurately predict sentiment from the cleaned review text. The following outlines the steps taken in model training and evaluation:
•	Data Preparation
The dataset used for model training comprised the cleaned_review column as the feature and the sentiment column as the target variable. The text data was converted into numerical features using TfidfVectorizer. This technique converts the text into a matrix of Term Frequency-Inverse Document Frequency (TF-IDF) features, which reflects the importance of words in the reviews relative to their frequency across the dataset.
•	Model Building
Three distinct machine learning models were constructed to evaluate their performance in sentiment classification:
-	Logistic Regression: This model was trained to perform binary and multiclass classification. It achieved an accuracy of 0.915, with both precision and recall also at 0.915. Logistic Regression is well-suited for text classification tasks and provided robust performance in this context.
-	Support Vector Classification (SVC): The SVC model demonstrated the highest performance among the models, with an accuracy of 0.924, precision of 0.924, and recall of 0.924. SVC is effective in handling high-dimensional data and was particularly well-suited for the TF-IDF feature set used in this analysis.
-	Naive Bayes: Although Naive Bayes showed lower performance compared to the other models, it served as a useful baseline. This model achieved an accuracy of 0.753, precision of 0.804, and recall of 0.753. The simplicity of Naive Bayes, with its assumption of feature independence, influenced its performance in this task.
•	Model Evaluation and Selection
The models were evaluated using accuracy, precision, and recall metrics. The SVC model was identified as the best performer due to its superior scores across all evaluation metrics. This model's higher accuracy, precision, recall and F1Score indicated its effectiveness in accurately classifying sentiments.
•	Model Production for Deployment
The SVC model, being the top-performing model, was selected for deployment. It was serialized into a pickle file using the joblib library. This process allowed the model to be saved as TBsentiment_mod.pkl, facilitating its use in production environments. The pickle file enables the model to be easily loaded and utilized for sentiment analysis on new review data.
Through these steps, the model training and building phase established a reliable sentiment analysis tool, leveraging advanced machine learning techniques to deliver accurate sentiment predictions and support informed decision-making.









