# Install required packages (if not already installed)
install.packages(c("e1071", "tm", "SnowballC", "caTools", "gmodels"))

# Load the required libraries
library(e1071)
library(tm)
library(SnowballC)
library(caTools)
library(gmodels)

# Step 1: Load the Dataset
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

# Step 2: Prepare the Data
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

# Step 3: Create a Document-Term Matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

# Inspect the DTM
sms_dtm

# Step 4: Split the Data into Training and Testing Sets
# Split the data into training and test sets
set.seed(123)
train_indices <- sample.split(sms_data$label, SplitRatio = 0.75)

sms_train <- sms_dtm[train_indices, ]
sms_test  <- sms_dtm[!train_indices, ]

sms_train_labels <- sms_data$label[train_indices]
sms_test_labels  <- sms_data$label[!train_indices]

# Step 5: Reduce Sparse Terms
# Remove sparse terms
sms_dtm_train <- removeSparseTerms(sms_train, 0.99)
sms_dtm_test  <- sms_test[, sms_dtm_train$dimnames$Terms]

# Step 6: Convert Counts to Factors
# Define a function to convert counts to "Yes"/"No"
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# Apply the function to training and test matrices
sms_train_matrix <- apply(sms_dtm_train, MARGIN = 2, convert_counts)
sms_test_matrix  <- apply(sms_dtm_test, MARGIN = 2, convert_counts)

# Step 7: Train the Naive Bayes Model
# Train the Naive Bayes classifier
sms_classifier <- naiveBayes(sms_train_matrix, sms_train_labels)

# Step 8: Test the Model
# Predict on test data
sms_test_pred <- predict(sms_classifier, sms_test_matrix)

# Step 9: Evaluate the Model
# Create a confusion matrix
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('Predicted', 'Actual'))

# Calculate accuracy
accuracy <- sum(sms_test_pred == sms_test_labels) / length(sms_test_labels)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Step 10: Improve the Model with Laplace Smoothing (Optional)
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
