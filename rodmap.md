# SpamDetection

Certainly! Below is an example of how you can implement a Naive Bayes classifier for spam detection in R using the e1071 package. We'll use the SMS Spam Collection Dataset, which contains a set of SMS messages labeled as "ham" (not spam) or "spam".

Note: Before running the code, make sure to install the necessary packages. Also, ensure you have an active internet connection to download the dataset directly.

# Install required packages (if not already installed)
install.packages(c("e1071", "tm", "SnowballC", "caTools", "gmodels"))

# Load the required libraries
library(e1071)
library(tm)
library(SnowballC)
library(caTools)
library(gmodels)
Step 1: Load the Dataset
We'll download the dataset from a public repository and read it into R.

# Download the dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
download.file(url, destfile = "smsspamcollection.zip")

# Unzip the dataset
unzip("smsspamcollection.zip")

# Read the dataset
sms_data <- read.delim("SMSSpamCollection", header = FALSE, sep = "\t", stringsAsFactors = FALSE)

# Assign column names
colnames(sms_data) <- c("label", "text")

# View the first few rows
head(sms_data)
Step 2: Prepare the Data
We'll preprocess the text data by converting to lowercase, removing numbers, stopwords, punctuation, and applying stemming.

# Convert labels to factors
sms_data$label <- factor(sms_data$label)

# Create a corpus
sms_corpus <- VCorpus(VectorSource(sms_data$text))

# Preprocess the text data
sms_corpus_clean <- sms_corpus %>%
  tm_map(content_transformer(tolower)) %>%     # Convert to lowercase
  tm_map(removeNumbers) %>%                    # Remove numbers
  tm_map(removePunctuation) %>%                # Remove punctuation
  tm_map(removeWords, stopwords("english")) %>%# Remove English stopwords
  tm_map(stemDocument) %>%                     # Perform stemming
  tm_map(stripWhitespace)                      # Remove extra whitespace
Step 3: Create a Document-Term Matrix
We'll convert the corpus into a Document-Term Matrix (DTM), which is essential for text classification tasks.

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# Inspect the DTM
sms_dtm
Step 4: Split the Data into Training and Testing Sets
We'll split the data into training and test sets using a 75-25 split.

# Split the data into training and test sets
set.seed(123)
train_indices <- sample.split(sms_data$label, SplitRatio = 0.75)

sms_train <- sms_dtm[train_indices, ]
sms_test  <- sms_dtm[!train_indices, ]

sms_train_labels <- sms_data$label[train_indices]
sms_test_labels  <- sms_data$label[!train_indices]
Step 5: Reduce Sparse Terms
To improve performance and reduce noise, we'll remove sparse terms that appear in less than 1% of documents.

# Remove sparse terms
sms_dtm_train <- removeSparseTerms(sms_train, 0.99)
sms_dtm_test  <- sms_test[, sms_dtm_train$dimnames$Terms]
Step 6: Convert Counts to Factors
Naive Bayes can handle categorical data effectively. We'll convert term frequencies to categorical variables indicating the presence ("Yes") or absence ("No") of a term.

# Define a function to convert counts to "Yes"/"No"
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# Apply the function to training and test matrices
sms_train_matrix <- apply(sms_dtm_train, MARGIN = 2, convert_counts)
sms_test_matrix  <- apply(sms_dtm_test, MARGIN = 2, convert_counts)
Step 7: Train the Naive Bayes Model
We'll train the model using the training data and the naiveBayes function.

# Train the Naive Bayes classifier
sms_classifier <- naiveBayes(sms_train_matrix, sms_train_labels)
Step 8: Test the Model
We'll use the trained model to make predictions on the test data.

# Predict on test data
sms_test_pred <- predict(sms_classifier, sms_test_matrix)
Step 9: Evaluate the Model
We'll evaluate the model's performance by creating a confusion matrix and calculating accuracy.

# Create a confusion matrix
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('Predicted', 'Actual'))

# Calculate accuracy
accuracy <- sum(sms_test_pred == sms_test_labels) / length(sms_test_labels)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))
Step 10: Improve the Model with Laplace Smoothing (Optional)
We can improve the model by applying Laplace smoothing to handle zero probabilities.

# Train the model with Laplace smoothing
sms_classifier2 <- naiveBayes(sms_train_matrix, sms_train_labels, laplace = 1)

# Predict on test data
sms_test_pred2 <- predict(sms_classifier2, sms_test_matrix)

# Evaluate the improved model
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('Predicted', 'Actual'))

# Calculate new accuracy
accuracy2 <- sum(sms_test_pred2 == sms_test_labels) / length(sms_test_labels)
print(paste("Accuracy with Laplace smoothing:", round(accuracy2 * 100, 2), "%"))
Explanation:
Data Preprocessing: Cleaning and preprocessing text data is crucial in text mining. We convert text to lowercase, remove numbers, punctuation, stopwords (common words that may not contribute to the model), and perform stemming to reduce words to their root forms.
Document-Term Matrix (DTM): A DTM represents the frequency of terms in documents. Each row represents a document, and each column represents a term.
Sparse Terms Removal: We remove terms that appear infrequently across the documents to reduce the dimensionality of the data and improve model performance.
Converting Counts to Factors: Converting term frequencies to categorical variables (presence or absence) helps Naive Bayes to handle the data effectively since it calculates probabilities for categorical features.
Model Training: The Naive Bayes classifier is trained on the training data and learns the probabilities of terms appearing in spam and ham messages.
Model Evaluation: The confusion matrix helps visualize the performance, showing true positives, true negatives, false positives, and false negatives. Accuracy is calculated as the proportion of correct predictions.
Conclusion:
Using Naive Bayes for spam detection is effective due to its simplicity and efficiency with high-dimensional data like text. By converting the text data into a suitable format and removing noise through preprocessing, we can achieve a good level of accuracy in classifying spam messages.

Additional Notes:
Dataset License: The SMS Spam Collection Dataset is available from the UCI Machine Learning Repository and is intended for educational and research purposes.
Experimentation: You can experiment with different thresholds for sparse term removal, n-grams, or using TF-IDF weighting to potentially improve the model.
Advanced Techniques: For further improvement, consider exploring other machine learning algorithms, such as Support Vector Machines (SVM), or deep learning models specialized for text data.
