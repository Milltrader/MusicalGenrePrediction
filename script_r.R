# Chargement des bibliothèques
install.packages('xgboost')
install.packages('readr')

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


setwd("C:/Users/anas9/Documents/2024-2025/s5/data_mining/projet_git/MusicalGenrePrediction")

# Chargement des données
data <- read_csv("data/features_30_sec.csv/features_30_sec.csv")

# Aperçu des donnée
print(dim(data))  # Dimensions du dataset
print(str(data))  # Structure des données
print(summary(data))  # Statistiques descriptives

# Vérification des valeurs manquantes
print(colSums(is.na(data)))

# Distribution des genres musicaux
ggplot(data, aes(x = label)) + 
  geom_bar(fill = "steelblue") + 
  theme_minimal() + 
  labs(title = "Distribution des genres musicaux", x = "Genre", y = "Nombre d'exemples") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Calcul des corrélations entre les features numériques
# Exclure label et filename des features numériques
numeric_features <- data %>% select(-label, -filename)
# Calcul des corrélations
cor_matrix <- cor(numeric_features)
# Affichage de la heatmap de corrélation
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7, tl.col = "black")



# Affichage des distributions des principales variables
features_to_plot <- c("chroma_stft_mean", "rms_mean", "spectral_centroid_mean", "tempo")
data_long <- data %>% select(all_of(features_to_plot), label) %>% pivot_longer(-label, names_to = "Feature", values_to = "Value")

ggplot(data_long, aes(x = Value, fill = label)) +
  geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
  facet_wrap(~Feature, scales = "free") +
  theme_minimal() +
  labs(title = "Distribution des principales features par genre", x = "Valeur", y = "Fréquence")



# Boxplots pour visualiser la variation des features par genre
features_boxplot <- c("rms_mean", "spectral_centroid_mean", "tempo")
data_boxplot <- data %>% select(all_of(features_boxplot), label) %>% pivot_longer(-label, names_to = "Feature", values_to = "Value")

ggplot(data_boxplot, aes(x = label, y = Value, fill = label)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~Feature, scales = "free") +
  theme_minimal() +
  labs(title = "Boxplots des features par genre", x = "Genre", y = "Valeur") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# PCA - Normalisation des données
numeric_features_scaled <- scale(numeric_features)

# Appliquer PCA
pca_result <- PCA(numeric_features_scaled, graph = FALSE)

# Définition manuelle des couleurs pour les genres musicaux
colors <- c("red", "blue", "green", "purple", "orange", "pink", "cyan", "brown", "yellow", "gray")

# Visualisation des individus
fviz_pca_ind(pca_result, 
             geom.ind = "point", 
             col.ind = data$label, 
             palette = colors,  # Utilisation de couleurs définies manuellement
             addEllipses = TRUE, 
             legend.title = "Genre") +
  theme_minimal() +
  labs(title = "Visualisation PCA des genres musicaux")

# Visualisation des variables avec une palette continue
fviz_pca_var(pca_result, col.var = "contrib") +
  scale_color_gradient(low = "blue", high = "red") +  # Palette continue
  theme_minimal() +
  labs(title = "Contribution des variables à la PCA")



# Visualisation des genres dans l'espace PCA
fviz_pca_ind(pca_result, 
             geom.ind = "point", 
             col.ind = data$label, 
             palette = colors,  # Couleurs définies pour chaque genre
             addEllipses = TRUE, 
             legend.title = "Genre") +
  theme_minimal() +
  labs(title = "Projection des genres dans l'espace PCA")




# Séparation des données en train/test
set.seed(123)
trainIndex <- createDataPartition(data$label, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# Suppression des colonnes non numériques pour l'entraînement
train_features <- train_data %>% select(-label, -filename)
test_features <- test_data %>% select(-label, -filename)
train_labels <- as.factor(train_data$label)
test_labels <- as.factor(test_data$label)

# Normalisation des features
preproc <- preProcess(train_features, method = c("center", "scale"))
train_features <- predict(preproc, train_features)
test_features <- predict(preproc, test_features)

# Modèle Random Forest
rf_model <- randomForest(x = train_features, y = train_labels, ntree = 100)
predictions_rf <- predict(rf_model, test_features)
accuracy_rf <- mean(predictions_rf == test_labels)
print(paste("Random Forest Accuracy:", accuracy_rf))

# Modèle SVM
svm_model <- svm(train_labels ~ ., data = train_features, kernel = "radial")
predictions_svm <- predict(svm_model, test_features)
accuracy_svm <- mean(predictions_svm == test_labels)
print(paste("SVM Accuracy:", accuracy_svm))

# Modèle K-Nearest Neighbors (KNN)
k <- 5
predictions_knn <- knn(train_features, test_features, train_labels, k = k)
accuracy_knn <- mean(predictions_knn == test_labels)
print(paste("KNN Accuracy (k=5):", accuracy_knn))

# Modèle XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_features), label = as.numeric(train_labels) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(test_features), label = as.numeric(test_labels) - 1)
params <- list(objective = "multi:softmax", num_class = length(unique(train_labels)), eval_metric = "mlogloss")
xgb_model <- xgb.train(params, train_matrix, nrounds = 100)
predictions_xgb <- predict(xgb_model, test_matrix)
accuracy_xgb <- mean(predictions_xgb == (as.numeric(test_labels) - 1))
print(paste("XGBoost Accuracy:", accuracy_xgb))

# Comparaison des performances
table_results <- data.frame(
  Model = c("Random Forest", "SVM", "KNN", "XGBoost"),
  Accuracy = c(accuracy_rf, accuracy_svm, accuracy_knn, accuracy_xgb)
)
print(table_results)
