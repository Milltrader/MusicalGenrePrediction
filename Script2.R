################################################################################
#       COMPLETE CODE: AUDIO GENRE PROJECT + FOCUSED EDA ON TOP FEATURES
################################################################################

# ----------------------------
# 1. Load Required Packages
# ----------------------------
# (Uncomment install.packages if needed)
# install.packages('ggplot2')
# install.packages('dplyr')
# install.packages('tidyr')
# install.packages('corrplot')
# install.packages('readr')
# install.packages('ggcorrplot')
# install.packages('FactoMineR')
# install.packages('factoextra')
# install.packages('caret')
# install.packages('randomForest')
# install.packages('e1071')
# install.packages('nnet')
# install.packages('xgboost')
# install.packages('class')
# install.packages('GGally')

library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(readr)
library(ggcorrplot)
library(FactoMineR)
library(factoextra)
library(caret)
library(randomForest)
library(e1071)
library(nnet)
library(xgboost)
library(class)
library(GGally)

# ----------------------------
# 2. Set Working Directory
# ----------------------------
setwd("D:/Coding/Projects 2/Audio Genres")

# ----------------------------
# 3. Load Data
# ----------------------------
data <- read_csv("data/features_30_sec.csv/features_30_sec.csv")
data <- data %>% select(-length)

# ----------------------------
# 4. Initial Data Overview
# ----------------------------
cat("\nDATA OVERVIEW\n")
cat("Dimensions (rows x columns):\n")
print(dim(data))

cat("\nStructure of the dataset:\n")
print(str(data))

cat("\nSummary statistics:\n")
print(summary(data))

cat("\nMissing values per column:\n")
print(colSums(is.na(data)))

# ----------------------------
# 5. Distribution of Genres
# ----------------------------
ggplot(data, aes(x = label)) + 
  geom_bar(fill = "steelblue") + 
  theme_minimal() + 
  labs(title = "Distribution of Musical Genres", x = "Genre", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ----------------------------
# 6. Correlation 
# ----------------------------
# Exclude non-numeric columns
numeric_features <- data %>% select(-label, -filename)
cor_matrix <- cor(numeric_features, use = "complete.obs")

corrplot(cor_matrix, method = "color", type = "upper",
         tl.cex = 0.7, tl.col = "black", 
         title = "Correlation Heatmap")

numeric_data <- data %>% select(-label, -filename)

# Identify highly correlated pairs (|correlation| > 0.9)
cor_matrix <- cor(numeric_data, use = "complete.obs")
highly_correlated <- findCorrelation(cor_matrix, cutoff = 0.8)

# Remove one feature from each highly correlated pair
data_reduced <- data %>%
  select(-filename) %>%
  select(-all_of(names(numeric_data)[highly_correlated]))

cat("\nNumber of Features Removed Due to High Correlation:", length(highly_correlated), "\n")
cat("\nRemaining Features After Correlation Filter:", ncol(data_reduced) - 1, "\n")  # -1 for label
# ----------------------------
# 7. Basic Feature Distributions
# ----------------------------
# List of all numeric features
numeric_features <- setdiff(names(data), c("label", "filename"))

# Chunk size
chunk_size <- 10

# Function to plot a chunk of features
plot_feature_chunk <- function(feature_chunk) {
  data_long_chunk <- data %>%
    select(all_of(feature_chunk)) %>%
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "Value")
  
  ggplot(data_long_chunk, aes(x = Value)) +
    geom_histogram(fill = "steelblue", color = "black", bins = 30, alpha = 0.7) +
    facet_wrap(~ Feature, scales = "free", ncol = 2) +   # 2 columns for better spacing
    theme_minimal() +
    labs(title = paste("Feature Distributions (", paste(feature_chunk, collapse = ", "), ")"), 
         x = "Value", y = "Frequency") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Loop through all features in chunks
for (i in seq(1, length(numeric_features), by = chunk_size)) {
  feature_chunk <- numeric_features[i:min(i + chunk_size - 1, length(numeric_features))]
  print(plot_feature_chunk(feature_chunk))
}


# ----------------------------
# 8. Boxplots by Genre
# ----------------------------
# List all numeric features (excluding label and filename)
features_boxplot_all <- setdiff(names(data), c("label", "filename"))

# Reshape to long format for ggplot
data_boxplot_all <- data %>%
  select(all_of(features_boxplot_all), label) %>%
  pivot_longer(-label, names_to = "Feature", values_to = "Value")

# Plotting function for smaller chunks (same approach as with histograms to avoid overcrowding)
plot_boxplot_chunk <- function(feature_chunk) {
  data_boxplot_chunk <- data_boxplot_all %>% filter(Feature %in% feature_chunk)
  
  ggplot(data_boxplot_chunk, aes(x = label, y = Value, fill = label)) +
    geom_boxplot(alpha = 0.7, outlier.size = 0.5) +
    facet_wrap(~ Feature, scales = "free", ncol = 2) +   # 2 columns for better spacing
    theme_minimal() +
    labs(title = paste("Boxplots of Features by Genre (", paste(feature_chunk, collapse = ", "), ")"),
         x = "Genre", y = "Value") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Define chunk size (how many features per plot)
chunk_size <- 6

# Loop through and plot in chunks
for (i in seq(1, length(features_boxplot_all), by = chunk_size)) {
  feature_chunk <- features_boxplot_all[i:min(i + chunk_size - 1, length(features_boxplot_all))]
  print(plot_boxplot_chunk(feature_chunk))
}

# ----------------------------
# 9. PCA
# ----------------------------
# Exclude non-numeric columns directly from `data`
numeric_data <- data %>% 
  select(-label, -filename)

# Scale numeric data
numeric_features_scaled <- scale(numeric_data)

# Perform PCA
pca_result <- PCA(as.data.frame(numeric_features_scaled), graph = FALSE)

# Scree plot - variance explained
fviz_eig(pca_result) + 
  labs(title = "Scree Plot: Variance Explained by PCA Components")

# Define palette for genres (adjust if you have different number of genres)
colors <- c("red", "blue", "green", "purple", "orange", 
            "pink", "cyan", "brown", "yellow", "gray")

# PCA projection with points colored by genre
fviz_pca_ind(
  pca_result,
  geom.ind = "point",
  col.ind = data$label,  # Use label for colors
  palette = colors,
  addEllipses = TRUE,
  legend.title = "Genre"
) +
  theme_minimal() +
  labs(title = "PCA Projection of Musical Genres")

# Variable contribution to PCs
fviz_pca_var(pca_result, col.var = "contrib") +
  scale_color_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "PCA: Feature Contributions")

# ----------------------------
# 10. Train/Test Split
# ----------------------------
set.seed(123)
trainIndex <- createDataPartition(data$label, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data  <- data[-trainIndex, ]

# Prepare features and labels
train_features <- train_data %>% select(-label, -filename)
train_labels   <- as.factor(train_data$label)

test_features  <- test_data %>% select(-label, -filename)
test_labels    <- as.factor(test_data$label)

# ----------------------------
# 11. Feature Preprocessing
# ----------------------------
preproc <- preProcess(train_features, method = c("center", "scale"))
train_features <- predict(preproc, train_features)
test_features  <- predict(preproc, test_features)

# For formula-based models (SVM), combine X and y in one frame:
train_final <- cbind(train_features, label = train_labels)
test_final  <- cbind(test_features,  label = test_labels)

# ----------------------------
# 12. Random Forest
# ----------------------------
set.seed(123)
rf_model <- randomForest(x = train_features, y = train_labels, ntree = 100)
pred_rf <- predict(rf_model, test_features)
acc_rf  <- mean(pred_rf == test_labels)

cat("\nRandom Forest Accuracy:", acc_rf, "\n")
cat("\nRandom Forest Confusion Matrix:\n")
print(confusionMatrix(pred_rf, test_labels))
varImpPlot(rf_model, main = "Random Forest Feature Importance")

# ----------------------------
# 13. SVM
# ----------------------------
svm_model <- svm(label ~ ., data = train_final, kernel = "radial")
pred_svm  <- predict(svm_model, test_final)
acc_svm   <- mean(pred_svm == test_labels)

cat("\nSVM Accuracy:", acc_svm, "\n")
cat("\nSVM Confusion Matrix:\n")
print(confusionMatrix(pred_svm, test_labels))

# ----------------------------
# 14. KNN
# ----------------------------
k <- 5
pred_knn <- knn(train_features, test_features, cl = train_labels, k = k)
acc_knn  <- mean(pred_knn == test_labels)

cat("\nKNN Accuracy (k=5):", acc_knn, "\n")
cat("\nKNN Confusion Matrix:\n")
print(confusionMatrix(pred_knn, test_labels))

# ----------------------------
# 15. XGBoost
# ----------------------------
train_matrix <- xgb.DMatrix(as.matrix(train_features), label = as.numeric(train_labels) - 1)
test_matrix  <- xgb.DMatrix(as.matrix(test_features),  label = as.numeric(test_labels) - 1)

params <- list(
  objective = "multi:softmax",
  num_class = length(unique(train_labels)),
  eval_metric = "mlogloss"
)
set.seed(123)
xgb_model <- xgb.train(params, train_matrix, nrounds = 100)
pred_xgb <- predict(xgb_model, test_matrix)
acc_xgb  <- mean(pred_xgb == (as.numeric(test_labels) - 1))

cat("\nXGBoost Accuracy:", acc_xgb, "\n")

# Convert numeric predictions to factor to get confusion matrix
pred_xgb_factor <- factor(pred_xgb, levels = 0:(length(unique(test_labels)) - 1))
test_labels_factor <- factor(as.numeric(test_labels) - 1, 
                             levels = 0:(length(unique(test_labels)) - 1))

cat("\nXGBoost Confusion Matrix:\n")
print(confusionMatrix(pred_xgb_factor, test_labels_factor))

xgb_importance <- xgb.importance(model = xgb_model)
print(xgb_importance)
xgb.plot.importance(xgb_importance)

# ----------------------------
# 16. Compare Model Performance
# ----------------------------
table_results <- data.frame(
  Model = c("Random Forest", "SVM", "KNN (k=5)", "XGBoost"),
  Accuracy = c(acc_rf, acc_svm, acc_knn, acc_xgb)
)
cat("\nModel Performance Comparison:\n")
print(table_results)

# ----------------------------
# 17. Focused EDA on Top Features (XGBoost)
# ----------------------------
# Example: pick top 6 features from XGBoost importance
top_features <- xgb_importance$Feature[1:6]
cat("\nTop XGBoost Features:\n")
print(top_features)

# Make a long dataframe with only these top features + label
top_data_long <- data %>%
  select(all_of(top_features), label) %>%
  pivot_longer(-label, names_to = "Feature", values_to = "Value")

# Violin plots for distribution shape
ggplot(top_data_long, aes(x = label, y = Value, fill = label)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  facet_wrap(~ Feature, scales = "free", ncol = 3) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Violin Plots of Top XGBoost Features by Genre",
    x = "Genre",
    y = "Value"
  )

# (Optional) Pairwise relationships among these top features
ggpairs(
  data %>% select(all_of(top_features), label),
  aes(color = label, alpha = 0.5),
  columns = 1:length(top_features)
) +
  theme_minimal() +
  labs(title = "Pairwise Plots of Top XGBoost Features by Genre")

################################################################################
# END
################################################################################

