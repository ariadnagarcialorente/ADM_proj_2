library(readr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(partykit)

# Set seed and number of crossvalidation folds
set.seed(42)
cv_folds = 5


# Load dataset and rename labels to clinical meaning
df <- read_csv("data.csv")

df <- df %>%
  mutate(
    target = factor(ifelse(num > 0, "Disease", "Healthy"),
                    levels = c("Healthy", "Disease")),
    sex     = factor(sex,     levels = c(0,1),   labels = c("Female","Male")),
    cp      = factor(cp,      levels = 1:4,       labels = c("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic")),
    fbs     = factor(fbs,     levels = c(0,1),    labels = c("<=120 mg/dl",">120 mg/dl")),
    restecg = factor(restecg, levels = 0:2,       labels = c("Normal","ST-T abnormality","LV hypertrophy")),
    exang   = factor(exang,   levels = c(0,1),    labels = c("No","Yes")),
    slope   = factor(slope,   levels = 1:3,       labels = c("Upsloping","Flat","Downsloping")),
    thal    = factor(thal,    levels = c(3,6,7),  labels = c("Normal","Fixed defect","Reversible defect"))
  ) %>%
  select(-num)

summary(df)


# Train unpruned CART
control = rpart.control(
  cp        = 0,
  minsplit  = 2,
  minbucket = 1,
  xval      = cv_folds
)
fit <- rpart(target ~ ., data = df, method = "class", control = control)
nodes <- as.numeric(rownames(fit$frame))
cat("Tree Depth:", max(rpart:::tree.depth(nodes)), "\n")
cat("Total Leaves:", sum(fit$frame$var == "<leaf>"), "\n")
printcp(fit)
plotcp(fit)
prp(fit,
    faclen     = 1,
    varlen     = 1,
    box.palette = "GnRd"
)

# Train pruned CART
cp.select <- function(big.tree) {
  min.x <- which.min(big.tree$cptable[, 4]) #column 4 is xerror
  for(i in 1:nrow(big.tree$cptable)) {
    if(big.tree$cptable[i, 4] < big.tree$cptable[min.x, 4] + big.tree$cptable[min.x, 5]) return(big.tree$cptable[i, 1]) #column 5: xstd, column 1: cp 
  }
}

pruned_tree <- prune(fit, cp = cp.select(fit))

nodes_pruned <- as.numeric(rownames(pruned_tree$frame))
cat("Tree Depth:", max(rpart:::tree.depth(nodes_pruned)), "\n")
cat("Total Leaves:", sum(pruned_tree$frame$var == "<leaf>"), "\n")

printcp(pruned_tree)
plotcp(pruned_tree)

prp(pruned_tree,
    faclen     = 0,
    varlen     = 0,
    extra   = 104,  # shows % of observations + class probability
    box.palette = "GnRd"
)



#Train CTree
ctrl <- trainControl(method="cv", number=cv_folds)

tree_001 <- train(target ~ ., data=df, method="ctree", trControl=ctrl,
                  tuneGrid = data.frame(mincriterion = 0.01))
tree_050 <- train(target ~ ., data=df, method="ctree", trControl=ctrl,
                  tuneGrid = data.frame(mincriterion = 0.50))
tree_099 <- train(target ~ ., data=df, method="ctree", trControl=ctrl,
                  tuneGrid = data.frame(mincriterion = 0.99))

get_depth <- function(node, d = 0) {
  if (node$terminal) return(d)
  max(get_depth(node$left, d+1), get_depth(node$right, d+1))
}

get_leaves <- function(node) {
  if (node$terminal) return(1)
  get_leaves(node$left) + get_leaves(node$right)
}

# --- Summary table ---
models      <- list(tree_001, tree_050, tree_099)
criteria    <- c(0.01, 0.50, 0.99)

for (i in seq_along(models)) {
  tree     <- models[[i]]$finalModel@tree
  acc      <- max(models[[i]]$results$Accuracy)
  kappa    <- max(models[[i]]$results$Kappa)
  depth    <- get_depth(tree)
  leaves   <- get_leaves(tree)
  
  cat(sprintf(
    "mincriterion=%.2f | Depth: %d | Leaves: %d | CV Accuracy: %.2f%% | Kappa: %.3f\n",
    criteria[i], depth, leaves, acc * 100, kappa
  ))
}

# --- Plot all three ---
par(mfrow=c(1,3))
plot(tree_001$finalModel, main="mincriterion=0.01 (liberal)")
plot(tree_050$finalModel, main="mincriterion=0.50 (moderate)")
plot(tree_099$finalModel, main="mincriterion=0.99 (conservative)")
