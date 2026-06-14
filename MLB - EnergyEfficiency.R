############################################################
# Energy Efficiency Analysis & Modeling in R
# Dataset Columns: X1–X8, Y1, Y2
############################################################

#############################
# 0. ENV RESET & PACKAGES
#############################

#rm(list = ls())
#graphics.off()

# Core packages
library(car)              # vif() for multicollinearity
library(neuralnet)        # neural networks
library(NeuralNetTools)   # plotnet(), garson() for NN visualization
library(gridExtra)        # grid.arrange() for multiple ggplots
library(dplyr)            # data manipulation
library(reshape2)         # melt(), dcast()

library(readxl)           # read_excel()
library(ggplot2)          # ggplot2 graphics

library(psych)            # describe()

# ML packages
library(e1071)            # svm()
library(class)            # knn()
library(naivebayes)       # naive_bayes()
library(rpart)            # decision trees
library(rpart.plot)       # rpart.plot()
library(randomForest)     # randomForest()
library(gbm)              # gbm.fit()
library(caret)

#############################
# DATA LOADING & LABEL MAPPING
#############################

df_raw <- EnergyEfficiency
colnames(df_raw) <- c("X1","X2","X3","X4","X5","X6","X7","X8","Y1","Y2")

var_labels <- c(
  X1 = "Relative Compactness",
  X2 = "Surface Area",
  X3 = "Wall Area",
  X4 = "Roof Area",
  X5 = "Overall Height",
  X6 = "Orientation",
  X7 = "Glazing Area",
  X8 = "Glazing Area Distribution",
  Y1 = "Heating Load",
  Y2 = "Cooling Load",
  HeatingLoad = "Heating Load",
  CoolingLoad = "Cooling Load"
)

label_var <- function(x) {
  unname(ifelse(x %in% names(var_labels), var_labels[x], x))
}

#############################
# CLEAN / RECODE DATA
#############################


# Convert to factors with labels
df_raw$Orientation <- factor(df_raw$X6,
                             levels = c(2, 3, 4, 5),
                             labels = c("North", "East", "South", "West"))

df_raw$GlazingDist <- factor(df_raw$X8,
                             levels = c(0, 1, 2, 3, 4, 5),
                             labels = c("Unknown", "Uniform", "North", "East", "South", "West"))

# Dummy variables for linear regression (removes 1 level automatically)
dummy_mat <- model.matrix(
  ~ Orientation + GlazingDist,
  data = df_raw
)

# Remove intercept column (optional for LR)
dummy_mat <- dummy_mat[, -1]  # remove "(Intercept)"


X_num <- df_raw[, c("X1","X2","X3","X4","X5","X7")]


data_ml <- cbind(
  X_num,
  dummy_mat,
  HeatingLoad = df_raw$Y1,
  CoolingLoad = df_raw$Y2
)

#df_plot <- data_ml

  
#############################
# 1a) Combined Scatterplots: Predictors vs Heating & Cooling (one plot)
#############################

num_predictors <- c("X1","X2","X3","X4","X5","X7")

df_plot = df_raw

# 1) Take only needed columns
df_scatter <- df_plot[, c(num_predictors, "Y1", "Y2")]

# 2) Melt predictors to long: Predictor + Xvalue
df_long_pred <- reshape2::melt(
  df_scatter,
  id.vars = c("Y1", "Y2"),
  measure.vars = num_predictors,
  variable.name = "Predictor",
  value.name = "Xvalue"
)

# 3) Melt Y1/Y2 to long: Target + Load
df_long <- reshape2::melt(
  df_long_pred,
  id.vars = c("Predictor", "Xvalue"),
  measure.vars = c("Y1", "Y2"),
  variable.name = "Target",
  value.name = "Load"
)

# Optional: nicer labels for Target
df_long$Target <- factor(
  df_long$Target,
  levels = c("Y1", "Y2"),
  labels = c(label_var("Y1"), label_var("Y2"))
)

# 4) Single combined plot: facets by predictor, color = Heating vs Cooling
p_combined <- ggplot(
  df_long,
  aes(x = Xvalue, y = Load, color = Target)
) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ Predictor, scales = "free_x",
             labeller = as_labeller(label_var)) +
  
  scale_color_manual(
    name = "Target",
    values = setNames(
      c("blue", "red"),
      c(label_var("Y1"), label_var("Y2"))
    )
  

  
  ) +
  xlab("Predictor value") +
  ylab("Load") +
  ggtitle("Predictors vs Heating & Cooling Loads (combined)") +
  theme_minimal()

print(p_combined)


#############################
# 1b) HISTOGRAMS FOR EACH VARIABLE
#############################

df_hist <- df_raw

for (col in names(df_hist)) {
  
  # If numeric → histogram
  if (is.numeric(df_hist[[col]])) {
    
    p <- ggplot(df_hist, aes_string(x = col)) +
      geom_histogram(
        bins = 30,
        fill = "blue",
        color = "white",
        alpha = 0.85
      ) +
      xlab(label_var(col)) +
      ggtitle(paste("Histogram of", label_var(col))) +
      theme_minimal()
    
    print(p)
    
  } else {
    
    # If categorical → bar plot
    p <- ggplot(df_hist, aes_string(x = col)) +
      geom_bar(
        fill = "blue",
        color = "white",
        alpha = 0.85
      ) +
      xlab(label_var(col)) +
      ggtitle(paste("Barplot of", label_var(col))) +
      theme_minimal()
    
    print(p)
  }
}


#############################
# 2.	Statistical analysis: DESCRIPTIVES, CORRELATION, LINEAR MODELS, VIF
#############################

cat("\n Descriptive statistics\n")
print(psych::describe(data_ml))

data_ml_num <- data_ml[, sapply(data_ml, is.numeric)]
cor_mat <- cor(data_ml_num)
cat("\nCorrelation matrix\n")
print(round(cor_mat, 2))


#c.	Linear regression for each Y-variable


# Maximum iterations = number of predictors

#############################
# 8. LINEAR MODELS (NO SCALING, NO VIF)
#############################

# Predictors = all columns except the two targets
#predictor_cols_all <- setdiff(colnames(data_ml), c("HeatingLoad", "CoolingLoad"))

# ----- 8.1 LM for Heating Load -----

#############################
# 8. LINEAR MODELS (LM ONLY, NO VIF, NO SCALING)
#############################

# Work on a copy with clear target names
df_lm <- df_raw
df_lm$HeatingLoad <- df_lm$Y1
df_lm$CoolingLoad <- df_lm$Y2

# ----- 8.1 LM for Heating Load -----
# Predictors: numeric X1, X2, X3, X4, X5, X7 + factors Orientation, GlazingDist
lm_heat <- lm(
  HeatingLoad ~ X1 + X2 + X3 + X4 + X5 + X7 + Orientation + GlazingDist,
  data = df_lm
)

cat("\n================ HEATING LOAD LM ==================\n")
print(summary(lm_heat))

# ----- 8.2 LM for Cooling Load -----
lm_cool <- lm(
  CoolingLoad ~ X1 + X2 + X3 + X4 + X5 + X7 + Orientation + GlazingDist,
  data = df_lm
)

cat("\n================ COOLING LOAD LM ==================\n")
print(summary(lm_cool))

# Optional: basic residual diagnostics
# par(mfrow = c(2, 2))
# plot(lm_heat)
# plot(lm_cool)
# par(mfrow = c(1, 1))


  

#############################
# 9. VIF ANALYSIS (Heating & Cooling)
#############################

#############################
# 9. VIF ANALYSIS (NUMERIC PREDICTORS ONLY)
#############################

# Work off the same data as your LM
#############################
# d. VIF ANALYSIS (NUMERIC PREDICTORS ONLY, X4 EXCLUDED)
#############################

# Use same base data as LM
df_lm <- df_raw
df_lm$HeatingLoad <- df_lm$Y1
df_lm$CoolingLoad <- df_lm$Y2

# Numeric predictors for VIF (exclude X4 because of collinearity)
num_preds <- c("X1","X2","X3","X5","X7")

# 1) VIF for Heating Load
lm_heat_vif <- lm(
  HeatingLoad ~ X1 + X2 + X3 + X5 + X7,
  data = df_lm
)

cat("\n================ VIF – HEATING LOAD (numeric only, X4 excluded) ==================\n")
print(car::vif(lm_heat_vif))

# 2) VIF for Cooling Load
lm_cool_vif <- lm(
  CoolingLoad ~ X1 + X2 + X3 + X5 + X7,
  data = df_lm
)

cat("\n================ VIF – COOLING LOAD (numeric only, X4 excluded) ==================\n")
print(car::vif(lm_cool_vif))


#.	Run modeling analysis using the techniques that you’ve learned so far


#############################
# a. Assign Categories A/B/C/D for Y1 & Y2
#############################

# HEATING LOAD (Y1)
qY1 <- quantile(df_raw$Y1, probs = c(0, 0.25, 0.50, 0.75, 1), na.rm = TRUE)

df_raw$HeatCat <- cut(
  df_raw$Y1,
  breaks = qY1,
  include.lowest = TRUE,
  labels = c("D", "C", "B", "A")   # D = lowest, A = highest
)

# COOLING LOAD (Y2)
qY2 <- quantile(df_raw$Y2, probs = c(0, 0.25, 0.50, 0.75, 1), na.rm = TRUE)

df_raw$CoolCat <- cut(
  df_raw$Y2,
  breaks = qY2,
  include.lowest = TRUE,
  labels = c("D", "C", "B", "A")   # D = lowest, A = highest
)

# Inspect category counts
table(df_raw$HeatCat)
table(df_raw$CoolCat)


#############################
# b. PERCEPTRONS FOR Y1 & Y2  (AB/CD → +1 / -1)
#############################

# 1) Assign +1 for A & B, -1 for C & D
df_raw$HeatBin <- ifelse(df_raw$HeatCat %in% c("A","B"),  1, -1)
df_raw$CoolBin <- ifelse(df_raw$CoolCat %in% c("A","B"),  1, -1)

# attach to ML dataset
data_ml$HeatBin <- df_raw$HeatBin
data_ml$CoolBin <- df_raw$CoolBin

# 2) Select predictors (same used by all models)
predictor_cols_all <- setdiff(
  colnames(data_ml),
  c("HeatingLoad","CoolingLoad","HeatBin","CoolBin")
)

# 3) Train/Test split
set.seed(123)
n <- nrow(data_ml)
train_idx <- sample(seq_len(n), size = floor(0.7*n))
test_idx  <- setdiff(seq_len(n), train_idx)

# TRAIN + TEST matrices
X_train <- as.matrix(data_ml[train_idx, predictor_cols_all])
X_test  <- as.matrix(data_ml[test_idx,  predictor_cols_all])

# Add bias term
X_train <- cbind(Intercept = 1, X_train)
X_test  <- cbind(Intercept = 1, X_test)

y_heat_train <- data_ml$HeatBin[train_idx]
y_heat_test  <- data_ml$HeatBin[test_idx]

y_cool_train <- data_ml$CoolBin[train_idx]
y_cool_test  <- data_ml$CoolBin[test_idx]

############################################
# 4) PERCEPTRON FUNCTION (your reference)
############################################
perceptron <- function(X, y, numEpochs) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize weights in [-10,10] like your reference
  w <- runif(p, -10, 10)
  
  for (epoch in 1:numEpochs) {
    predictedResult <- numeric(n)
    numIncorrect <- 0
    
    for (i in 1:n) {
      xi <- as.numeric(X[i, ])
      predictedResult[i] <- sign(sum(w * xi))
      if (predictedResult[i] == 0) predictedResult[i] <- 1
      
      # If incorrect → update weight
      if (predictedResult[i] != y[i]) {
        numIncorrect <- numIncorrect + 1
        w <- w + y[i] * xi      # w <- w + y_i * x_i
      }
    }
    
    cat("\nEpoch #: ", epoch)
    cat("\nNumber Incorrect: ", numIncorrect)
    cat("\nWeights: ", paste(round(w, 4), collapse = ", "))
  }
  
  return(w)
}

# Accuracy helper
perceptron_accuracy <- function(w, X, y_true) {
  preds <- sign(X %*% w)
  preds[preds == 0] <- 1
  mean(preds == y_true)
}

############################################
# 5) RUN 5 MODELS FOR HEATING & COOLING
############################################
numEpochs <- 10
n_reps <- 5

perc_results <- data.frame(
  Target = character(),
  ModelID = integer(),
  Accuracy = numeric(),
  stringsAsFactors = FALSE
)

for (rep in 1:n_reps) {
  cat("\n======================= HEATING MODEL", rep, "=======================\n")
  set.seed(123 + rep)
  w_heat <- perceptron(X_train, y_heat_train, numEpochs)
  acc_heat <- perceptron_accuracy(w_heat, X_test, y_heat_test)
  cat(sprintf("\nFinal Heating Weight (Model %d): %s\n",
              rep, paste(round(w_heat, 4), collapse=", ")))
  cat(sprintf("Heating Accuracy (Model %d): %.4f\n", rep, acc_heat))
  
  perc_results <- rbind(
    perc_results,
    data.frame(Target="HeatingLoad", ModelID=rep, Accuracy=acc_heat)
  )
  
  cat("\n======================= COOLING MODEL", rep, "=======================\n")
  set.seed(123 + rep)
  w_cool <- perceptron(X_train, y_cool_train, numEpochs)
  acc_cool <- perceptron_accuracy(w_cool, X_test, y_cool_test)
  cat(sprintf("\nFinal Cooling Weight (Model %d): %s\n",
              rep, paste(round(w_cool, 4), collapse=", ")))
  cat(sprintf("Cooling Accuracy (Model %d): %.4f\n", rep, acc_cool))
  
  perc_results <- rbind(
    perc_results,
    data.frame(Target="CoolingLoad", ModelID=rep, Accuracy=acc_cool)
  )
}

############################################
# 6) SHOW SUMMARY
############################################
cat("\n==================== PERCEPTRON RESULTS ====================\n")
print(perc_results)


#############################
# Convert perceptron results to matrix form
#############################

# perc_results must contain columns:
# Target ("HeatingLoad"/"CoolingLoad"), ModelID (1–5), Accuracy

# Create matrix: 2 rows (Heating, Cooling) × 5 columns (Model 1–5)
acc_matrix <- matrix(NA, nrow = 2, ncol = 5)
rownames(acc_matrix) <- c("HeatingLoad", "CoolingLoad")
colnames(acc_matrix) <- paste0("Model", 1:5)

for (i in 1:nrow(perc_results)) {
  tgt <- perc_results$Target[i]
  mid <- perc_results$ModelID[i]
  acc <- perc_results$Accuracy[i]
  
  acc_matrix[tgt, paste0("Model", mid)] <- round(acc, 4)
}

cat("\nPERCEPTRON ACCURACY MATRIX\n")
print(acc_matrix)


############################################################
# c. SVM for Heating & Cooling (A/B/C/D) + Plots + Accuracy
############################################################

library(e1071)

# Make sure categories are factors with levels A–D
df_raw$HeatCat <- factor(df_raw$HeatCat, levels = c("A","B","C","D"))
df_raw$CoolCat <- factor(df_raw$CoolCat, levels = c("A","B","C","D"))


set.seed(123)
n <- nrow(df_raw)
train_idx <- sample(seq_len(n), size = floor(0.7 * n))
test_idx  <- setdiff(seq_len(n), train_idx)

train_data <- df_raw[train_idx, ]
test_data  <- df_raw[test_idx, ]





##################################

############################################################
# c. SVM for Heating & Cooling (A/B/C/D) + Confusion Matrix
############################################################

library(e1071)

df_raw$HeatCat <- factor(df_raw$HeatCat, levels=c("A","B","C","D"))
df_raw$CoolCat <- factor(df_raw$CoolCat, levels=c("A","B","C","D"))


set.seed(123)
train_data <- df_raw[train_idx, ]
test_data  <- df_raw[test_idx, ]

#######################################
# SVM 1 — HEATING (A/B/C/D)
#######################################
set.seed(123)

cat("\n==================== SVM 1 (Heating) ====================\n")

svm1_heat <- svm(
  formula = HeatCat ~ X1 + X2 + X3 + X4 + X5 + X6 + X7,
  data    = train_data,
  kernel  = "radial",
  cost    = 1,
  scale   = TRUE
)

print(svm1_heat)

# Predictions
pred_heat <- predict(svm1_heat, newdata = test_data)

# Confusion Matrix
cm_heat <- table(
  Actual = test_data$HeatCat,
  Predicted = pred_heat
)

cat("\nConfusion Matrix (Heating):\n")
print(cm_heat)

# Accuracy
acc_heat <- mean(pred_heat == test_data$HeatCat)
cat(sprintf("\nSVM 1 (Heating: X1 + X2) Accuracy: %.4f\n", acc_heat))

# Plot
plot(
  svm1_heat,
  data=train_data,
  X1 ~ X2,
  main="SVM – Heating Load Categories (A/B/C/D)"
)


#######################################
# SVM 2 — COOLING (A/B/C/D)
#######################################

cat("\n==================== SVM 2 (Cooling) ====================\n")

svm2_cool <- svm(
  formula = CoolCat ~ X1 + X2 + X3 + X4 + X5 + X6 + X7,
  data    = train_data,
  kernel  = "radial",
  cost    = 1,
  scale   = TRUE
)

print(svm2_cool)

# Predictions
pred_cool <- predict(svm2_cool, newdata = test_data)

# Confusion Matrix
cm_cool <- table(
  Actual = test_data$CoolCat,
  Predicted = pred_cool
)

cat("\nConfusion Matrix (Cooling):\n")
print(cm_cool)

# Accuracy
acc_cool <- mean(pred_cool == test_data$CoolCat)
cat(sprintf("\nSVM 2 (Cooling: X1 + X2) Accuracy: %.4f\n", acc_cool))

# Plot
plot(
  svm2_cool,
  data=train_data,
  X1 ~ X2,
  main="SVM – Cooling Load Categories (A/B/C/D)"
)


############################################################
# d. Neural networks: 1–5 hidden nodes for Heating & Cooling
# Includes plot for every network
############################################################

library(neuralnet)


# Prepare dataset
df_nn <- df_raw
df_nn$HeatingLoad <- df_nn$Y1
df_nn$CoolingLoad <- df_nn$Y2

predictors_nn <- c("X1","X2","X3","X4","X5","X6","X7","X8")

# Train/test split (reuse if already defined)
set.seed(123)
n <- nrow(df_nn)
train_idx <- sample(seq_len(n), size = floor(0.7 * n))
test_idx  <- setdiff(seq_len(n), train_idx)

train_nn <- df_nn[train_idx, ]
test_nn  <- df_nn[test_idx, ]

# Store results
nn_results <- data.frame(
  Target      = character(),
  HiddenNodes = integer(),
  Accuracy    = numeric(),
  stringsAsFactors = FALSE
)

# Accuracy function: R²
nn_r2 <- function(y_true, y_pred) {
  sse <- sum((y_true - y_pred)^2)
  sst <- sum((y_true - mean(y_true))^2)
  1 - sse/sst
}

############################################################
# Loop over both targets and hidden node counts
############################################################

for (target in c("HeatingLoad", "CoolingLoad")) {
  
  formula_nn <- as.formula(
    paste(target, "~", paste(predictors_nn, collapse = " + "))
  )
  
  y_test <- test_nn[[target]]
  
  cat("\n================ NEURAL NETS FOR", target, "================\n")
  
  for (h in 1:5) {
    
    cat("\n---", target, "with", h, "hidden node(s) ---\n")
    
    # Train neural network
    set.seed(123 + ifelse(target=="HeatingLoad", 0, 100) + h)
    
    net_model <- neuralnet(
      formula_nn,
      data = train_nn,
      hidden = h,
      lifesign = "minimal",
      linear.output = TRUE,
      threshold = 0.01
    )
    
    #############################
    # 📌 PLOT neural network
    #############################
    plot(
      net_model,
      rep = "best",
      main = paste("Neural Network:", target, "- Hidden Nodes:", h)
    )
    
    #############################
    # Compute predictions
    #############################
    nn_pred <- neuralnet::compute(
      net_model,
      test_nn[, predictors_nn]
    )$net.result[, 1]
    
    
    acc <- nn_r2(y_test, nn_pred)
    
    cat(sprintf(
      "Test Accuracy (R²) for %s (hidden=%d): %.4f\n",
      target, h, acc
    ))
    
    nn_results <- rbind(
      nn_results,
      data.frame(
        Target = target,
        HiddenNodes = h,
        Accuracy = acc
      )
    )
  }
}

############################################################
# Summary Table
############################################################

cat("\n=========== NEURAL NETWORK ACCURACY SUMMARY (R^2) ===========\n")
print(nn_results)

# Optional: accuracy matrix
nn_acc_matrix <- xtabs(Accuracy ~ Target + HiddenNodes, data = nn_results)
nn_acc_matrix <- round(nn_acc_matrix, 4)

cat("\nAccuracy matrix (R²):\n")
print(nn_acc_matrix)


############################################################
# e. K-Nearest Neighbors for Heating & Cooling (A/B/C/D)
############################################################

library(class)

############################################################
# e. KNN for Heating & Cooling (A/B/C/D) — Reference Style
############################################################

library(class)
library(gmodels)          # CrossTable()
library(caret)            # confusionMatrix()

# Ensure categories exist from part (a)
df_raw$HeatCat <- factor(df_raw$HeatCat, levels = c("A","B","C","D"))
df_raw$CoolCat <- factor(df_raw$CoolCat, levels = c("A","B","C","D"))

# Train/test split
set.seed(123)
n <- nrow(df_raw)
train_idx <- sample(seq_len(n), size = floor(0.7 * n))
test_idx  <- setdiff(seq_len(n), train_idx)

train_knn <- df_raw[train_idx, ]
test_knn  <- df_raw[test_idx, ]

predictors_knn <- c("X1","X2","X3","X4","X5","X6","X7","X8")

X_train <- as.matrix(train_knn[, predictors_knn])
X_test  <- as.matrix(test_knn[,  predictors_knn])

# Standardize predictors
X_train_sc <- scale(X_train)
X_test_sc <- scale(
  X_test,
  center = attr(X_train_sc, "scaled:center"),
  scale  = attr(X_train_sc, "scaled:scale")
)

# Results storage
knn_results <- data.frame(
  Target = character(),
  k = integer(),
  Accuracy = numeric(),
  stringsAsFactors = FALSE
)

############################################################
# LOOP over k values
############################################################

for (k in k_values) {
  
  cat("\n====================== KNN (k =", k, ") ======================\n")
  
  ##############################
  # 1. Heating Load (A/B/C/D)
  ##############################
  cat("\n--- Heating Load ---\n")
  
  heat_pred <- knn(
    train = X_train_sc,
    test  = X_test_sc,
    cl    = train_knn$HeatCat,
    k     = k
  )
  
  # CrossTable evaluation
  heat_table <- CrossTable(
    x = test_knn$HeatCat,
    y = heat_pred,
    prop.chisq = FALSE
  )
  
  # Accuracy using reference method
  heat_acc <- sum(diag(heat_table$prop.tbl))
  
  cat(sprintf("Heating Accuracy (k=%d): %.4f\n", k, heat_acc))
  
  # Store result
  knn_results <- rbind(
    knn_results,
    data.frame(Target = "HeatingLoad", k = k, Accuracy = heat_acc)
  )
  
  # Caret confusion matrix (optional)
  print(confusionMatrix(heat_pred, test_knn$HeatCat))
  
  
  ##############################
  # ❄ 2. Cooling Load (A/B/C/D)
  ##############################
  cat("\n--- Cooling Load ---\n")
  
  cool_pred <- knn(
    train = X_train_sc,
    test  = X_test_sc,
    cl    = train_knn$CoolCat,
    k     = k
  )
  
  # CrossTable evaluation
  cool_table <- CrossTable(
    x = test_knn$CoolCat,
    y = cool_pred,
    prop.chisq = FALSE
  )
  
  # Accuracy
  cool_acc <- sum(diag(cool_table$prop.tbl))
  
  cat(sprintf("Cooling Accuracy (k=%d): %.4f\n", k, cool_acc))
  
  # Store result
  knn_results <- rbind(
    knn_results,
    data.frame(Target = "CoolingLoad", k = k, Accuracy = cool_acc)
  )
  
  # Caret confusion matrix (optional)
  print(confusionMatrix(cool_pred, test_knn$CoolCat))
}

############################################################
# Summary of all KNN models
############################################################

cat("\nKNN ACCURACY SUMMARY\n")
print(knn_results)


############################################################
# f. Naive Bayes for Heating & Cooling (A/B/C/D)
#     + Conditional Probability Tables Printed
############################################################

library(naivebayes)
library(caret)

set.seed(123)

# Ensure proper factor levels
df_raw$HeatCat <- factor(df_raw$HeatCat, levels = c("A","B","C","D"))
df_raw$CoolCat <- factor(df_raw$CoolCat, levels = c("A","B","C","D"))

set.seed(123)

idx <- sample(nrow(df_raw), 0.7 * nrow(df_raw))


train_nb <- df_raw[idx, ]
test_nb  <- df_raw[-idx, ]

predictors_nb <- c("X1","X2","X3","X4","X5","X6","X7","X8")

############################################################
# Helper: Print conditional probabilities in clean format
############################################################
print_nb_conditionals <- function(nb_model, title = "Model") {
  
  cat("\n======================================================\n")
  cat("Conditional Probability Tables:", title, "\n")
  cat("======================================================\n\n")
  
  # Prior probabilities
  cat("Class Prior Probabilities:\n")
  print(round(nb_model$prior, 4))
  cat("\n")
  
  # Conditional probability tables for each predictor
  for (var in names(nb_model$tables)) {
    cat("------------------------------------------------------\n")
    cat("P(", var, " | Class )\n", sep = "")
    cond <- nb_model$tables[[var]]
    print(round(cond, 4))
    cat("\n")
  }
}

############################################################
# 1) Naive Bayes — HEATING
############################################################

cat("\n==================== Naive Bayes — Heating ====================\n")

nb_heat <- naive_bayes(
  x = train_nb[, predictors_nb],
  y = train_nb$HeatCat,
  laplace = 1
)

# PRINT CONDITIONAL PROBABILITIES
print_nb_conditionals(nb_heat, title = "Heating Load (A/B/C/D)")

# Predict
pred_heat_nb <- predict(nb_heat, test_nb[, predictors_nb])

# Confusion Matrix
cat("\nConfusion Matrix — Heating:\n")
cm_heat_nb <- table(Actual = test_nb$HeatCat, Predicted = pred_heat_nb)
print(cm_heat_nb)

# Accuracy
acc_heat_nb <- mean(pred_heat_nb == test_nb$HeatCat)
cat(sprintf("\nNaive Bayes Heating Accuracy: %.4f\n", acc_heat_nb))


############################################################
# 2) Naive Bayes — COOLING
############################################################

cat("\n==================== Naive Bayes — Cooling ====================\n")

nb_cool <- naive_bayes(
  x = train_nb[, predictors_nb],
  y = train_nb$CoolCat,
  laplace = 1
)

# PRINT CONDITIONAL PROBABILITIES
print_nb_conditionals(nb_cool, title = "Cooling Load (A/B/C/D)")

# Predict
pred_cool_nb <- predict(nb_cool, test_nb[, predictors_nb])

# Confusion Matrix
cat("\nConfusion Matrix — Cooling:\n")
cm_cool_nb <- table(Actual = test_nb$CoolCat, Predicted = pred_cool_nb)
print(cm_cool_nb)

# Accuracy
acc_cool_nb <- mean(pred_cool_nb == test_nb$CoolCat)
cat(sprintf("\nNaive Bayes Cooling Accuracy: %.4f\n", acc_cool_nb))


############################################################
# Summary table
############################################################

nb_results <- data.frame(
  Target   = c("HeatingLoad", "CoolingLoad"),
  Model    = c("NaiveBayes", "NaiveBayes"),
  Accuracy = c(acc_heat_nb, acc_cool_nb)
)

cat("\n NAIVE BAYES ACCURACY SUMMARY \n")
print(nb_results)


############################################################
# g. Decision Tree (1 = A/B, 0 = C/D) for Heating & Cooling
############################################################

library(rpart)
library(rpart.plot)


############################################################
# g. Decision Tree (1 = A/B, 0 = C/D) for Heating & Cooling
############################################################

library(rpart)
library(rpart.plot)
library(caret)   # <-- for confusionMatrix()

# ----------------------------------------------------------
# 1. Correct binary assignment (EXPLICIT)
# ----------------------------------------------------------

df_raw$HeatBin <- ifelse(df_raw$HeatCat %in% c("A","B"), 1,
                         ifelse(df_raw$HeatCat %in% c("C","D"), 0, NA))

df_raw$CoolBin <- ifelse(df_raw$CoolCat %in% c("A","B"), 1,
                         ifelse(df_raw$CoolCat %in% c("C","D"), 0, NA))

df_raw$HeatBin <- factor(df_raw$HeatBin, levels = c(0,1))
df_raw$CoolBin <- factor(df_raw$CoolBin, levels = c(0,1))

set.seed(123)
idx <- sample(nrow(df_raw), 0.7 * nrow(df_raw))

# Train/test split
train_dt <- df_raw[idx, ]
test_dt  <- df_raw[-idx, ]

predictors <- c("X1","X2","X3","X4","X5","X6","X7","X8")

# ----------------------------------------------------------
# 2. Decision Tree — Heating (1 for A/B, 0 for C/D)
# ----------------------------------------------------------

cat("\n=================== DECISION TREE – HEATING ===================\n")

form_heat <- as.formula(
  paste("HeatBin ~", paste(predictors, collapse = " + "))
)

tree_heat <- rpart(
  form_heat,
  data = train_dt,
  method = "class"
)

rpart.plot(
  tree_heat,
  main = "Decision Tree – Heating Load (1 = A/B, 0 = C/D)"
)

pred_heat <- predict(tree_heat, newdata = test_dt, type = "class")

# Manual accuracy (optional)
acc_heat_manual <- mean(pred_heat == test_dt$HeatBin)

cat("\nConfusion Matrix – Heating (table only):\n")
print(table(Actual = test_dt$HeatBin, Predicted = pred_heat))

cat(sprintf("\nDecision Tree Heating Accuracy (manual): %.4f\n", acc_heat_manual))

# Full confusion matrix + statistics using caret
cm_heat <- confusionMatrix(pred_heat, test_dt$HeatBin)

cat("\nConfusion Matrix and Statistics – Heating (caret):\n")
print(cm_heat)   # this prints confusion matrix + Accuracy, Kappa, Sensitivity, etc.

cat(sprintf("\nDecision Tree Heating Accuracy (caret): %.4f\n",
            cm_heat$overall["Accuracy"]))


# ----------------------------------------------------------
# 3. Decision Tree — Cooling (1 for A/B, 0 for C/D)
# ----------------------------------------------------------

cat("\n=================== DECISION TREE – COOLING ===================\n")

form_cool <- as.formula(
  paste("CoolBin ~", paste(predictors, collapse = " + "))
)

tree_cool <- rpart(
  form_cool,
  data = train_dt,
  method = "class"
)

rpart.plot(
  tree_cool,
  main = "Decision Tree – Cooling Load (1 = A/B, 0 = C/D)"
)

pred_cool <- predict(tree_cool, newdata = test_dt, type = "class")

# Manual accuracy (optional)
acc_cool_manual <- mean(pred_cool == test_dt$CoolBin)

cat("\nConfusion Matrix – Cooling (table only):\n")
print(table(Actual = test_dt$CoolBin, Predicted = pred_cool))

cat(sprintf("\nDecision Tree Cooling Accuracy (manual): %.4f\n", acc_cool_manual))

# Full confusion matrix + statistics using caret
cm_cool <- confusionMatrix(pred_cool, test_dt$CoolBin)

cat("\nConfusion Matrix and Statistics – Cooling (caret):\n")
print(cm_cool)

cat(sprintf("\nDecision Tree Cooling Accuracy (caret): %.4f\n",
            cm_cool$overall["Accuracy"]))


############################################################
# h. Random Forest (A/B vs C/D) for Heating & Cooling
############################################################

library(randomForest)
library(caret)

# ----------------------------------------------------------
# 1. Assign 1 for A/B, 0 for C/D WITH LABELS
# ----------------------------------------------------------

df_raw$HeatBin <- factor(
  ifelse(df_raw$HeatCat %in% c("A","B"), 1, 0),
  levels = c(0,1),
  labels = c("C/D", "A/B")     # <-- LABELS ADDED
)

df_raw$CoolBin <- factor(
  ifelse(df_raw$CoolCat %in% c("A","B"), 1, 0),
  levels = c(0,1),
  labels = c("C/D", "A/B")     # <-- LABELS ADDED
)

# ----------------------------------------------------------
# 2. Train-test split
# ----------------------------------------------------------

set.seed(123)
idx <- sample(nrow(df_raw), 0.7 * nrow(df_raw))

train_rf <- df_raw[idx, ]
test_rf  <- df_raw[-idx, ]

predictors <- c("X1","X2","X3","X4","X5","X6","X7","X8")

# ----------------------------------------------------------
# 3. RANDOM FOREST — HEATING
# ----------------------------------------------------------

cat("\n=================== RANDOM FOREST – HEATING ===================\n")

rf_heat <- randomForest(
  x = train_rf[, predictors],
  y = train_rf$HeatBin,
  ntree = 500,
  importance = TRUE
)

print(rf_heat)

pred_heat <- predict(rf_heat, test_rf[, predictors])

cm_heat <- confusionMatrix(pred_heat, test_rf$HeatBin)

cat("\nConfusion Matrix and Statistics – Heating (RF):\n")
print(cm_heat)

cat(sprintf("\nRandom Forest Heating Accuracy: %.4f\n",
            cm_heat$overall["Accuracy"]))


# ----------------------------------------------------------
# 4. RANDOM FOREST — COOLING
# ----------------------------------------------------------

cat("\n=================== RANDOM FOREST – COOLING ===================\n")

rf_cool <- randomForest(
  x = train_rf[, predictors],
  y = train_rf$CoolBin,
  ntree = 500,
  importance = TRUE
)

print(rf_cool)

pred_cool <- predict(rf_cool, test_rf[, predictors])

cm_cool <- confusionMatrix(pred_cool, test_rf$CoolBin)

cat("\nConfusion Matrix and Statistics – Cooling (RF):\n")
print(cm_cool)

cat(sprintf("\nRandom Forest Cooling Accuracy: %.4f\n",
            cm_cool$overall["Accuracy"]))


############################################################
# i. XGBoost (Boosting) – 1 = A/B, 0 = C/D
############################################################

library(xgboost)
library(caret)

# ----------------------------------------------------------
# 1. Binary targets (A/B = 1, C/D = 0)
# ----------------------------------------------------------

df_raw$HeatBin <- ifelse(df_raw$HeatCat %in% c("A","B"), 1, 0)
df_raw$CoolBin <- ifelse(df_raw$CoolCat %in% c("A","B"), 1, 0)

df_raw$HeatBin <- factor(df_raw$HeatBin, levels = c(0,1), labels = c("C/D","A/B"))
df_raw$CoolBin <- factor(df_raw$CoolBin, levels = c(0,1), labels = c("C/D","A/B"))

# ----------------------------------------------------------
# 2. Train/Test split (iris style)
# ----------------------------------------------------------

set.seed(123)
idx <- sample(nrow(df_raw), 0.7 * nrow(df_raw))

train_xgb <- df_raw[idx, ]
test_xgb  <- df_raw[-idx, ]

predictors <- c("X1","X2","X3","X4","X5","X6","X7","X8")

x_train <- as.matrix(train_xgb[, predictors])
x_test  <- as.matrix(test_xgb[, predictors])

# Convert factor back to numeric for XGB
y_heat_train <- ifelse(train_xgb$HeatBin == "A/B", 1, 0)
y_cool_train <- ifelse(train_xgb$CoolBin == "A/B", 1, 0)

# ----------------------------------------------------------
# 3. XGBOOST – HEATING LOAD
# ----------------------------------------------------------

cat("\n=================== XGBOOST – HEATING (A/B vs C/D) ===================\n")

dtrain_heat <- xgb.DMatrix(data = x_train, label = y_heat_train)

xgb_heat <- xgboost(
  data = dtrain_heat,
  max.depth = 5,
  eta = 1,
  nthread = 2,
  nrounds = 1000,
  objective = "binary:logistic",
  verbose = 0
)

# Predictions
heat_prob <- predict(xgb_heat, x_test)
heat_pred <- ifelse(heat_prob >= 0.5, "A/B", "C/D")
heat_pred <- factor(heat_pred, levels = c("C/D","A/B"))

cm_heat <- confusionMatrix(heat_pred, test_xgb$HeatBin)

cat("\nConfusion Matrix and Statistics – Heating (XGBoost):\n")
print(cm_heat)

cat(sprintf("\nXGBoost Heating Accuracy: %.4f\n",
            cm_heat$overall["Accuracy"]))

# Variable Importance Plot
cat("\nVariable Importance – Heating (XGBoost):\n")
xgb.plot.importance(xgb.importance(model = xgb_heat), measure = "Gain")


# ----------------------------------------------------------
# 4. XGBOOST – COOLING LOAD
# ----------------------------------------------------------

cat("\n=================== XGBOOST – COOLING (A/B vs C/D) ===================\n")

dtrain_cool <- xgb.DMatrix(data = x_train, label = y_cool_train)

xgb_cool <- xgboost(
  data = dtrain_cool,
  max.depth = 5,
  eta = 1,
  nthread = 2,
  nrounds = 1000,
  objective = "binary:logistic",
  verbose = 0
)

cool_prob <- predict(xgb_cool, x_test)
cool_pred <- ifelse(cool_prob >= 0.5, "A/B", "C/D")
cool_pred <- factor(cool_pred, levels = c("C/D","A/B"))

cm_cool <- confusionMatrix(cool_pred, test_xgb$CoolBin)

cat("\nConfusion Matrix and Statistics – Cooling (XGBoost):\n")
print(cm_cool)

cat(sprintf("\nXGBoost Cooling Accuracy: %.4f\n",
            cm_cool$overall["Accuracy"]))

# Variable Importance Plot
cat("\nVariable Importance – Cooling (XGBoost):\n")
xgb.plot.importance(xgb.importance(model = xgb_cool), measure = "Gain")


# ---------------------------
# Variable Importance Plot
# ---------------------------
cat("\nVariable Importance – Heating (XGBoost):\n")
xgb.plot.importance(xgb.importance(model = xgb_heat), measure = "Gain")


# ---------------------------
# ERROR PLOT (Training Logloss)
# ---------------------------
cat("\nError Plot – Heating (XGBoost):\n")

eval_heat <- xgb_heat$evaluation_log

ggplot(eval_heat, aes(x = iter, y = train_logloss)) +
  geom_line(color = "red") +
  ggtitle("XGBoost Error Plot – Heating Load") +
  xlab("Iteration") + ylab("Training Logloss") +
  scale_x_continuous(limits = c(0, 50)) +
  theme_minimal()


# ----------------------------------------------------------


# ---------------------------
# ERROR PLOT (Training Logloss)
# ---------------------------
cat("\nError Plot – Cooling (XGBoost):\n")

eval_cool <- xgb_cool$evaluation_log

ggplot(eval_cool, aes(x = iter, y = train_logloss)) +
  geom_line(color = "blue") +
  ggtitle("XGBoost Error Plot – Cooling Load") +
  xlab("Iteration") + ylab("Training Logloss") +
  scale_x_continuous(limits = c(0, 50)) +
  theme_minimal()

