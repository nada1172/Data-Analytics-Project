# =====================================================================
# IS 6114 - Data Analytics Project
# Part 1: Data Wrangling - UCI Adult (Census Income) dataset
#
# Input (from UCI):
#   data/raw/adult.data
#   data/raw/adult.test
#
# Outputs (all inside data/processed/):
#   adult_clean.csv / adult_clean.rds : cleaned, unscaled data for EDA
#   adult_train_clean.csv             : cleaned original training partition
#   adult_test_clean.csv              : cleaned original test partition
#   adult_dummies.csv                 : train-fitted encoding; source is metadata
#   adult_train_dummies.csv           : numeric ML train data, unscaled
#   adult_test_dummies.csv            : numeric ML test data, unscaled
#   validation_report.csv             : results of the 24 quality checks
#
# Run from project root: Rscript code/01_data_wrangling.R
# Install once if needed: install.packages(c("dplyr", "tidyr"))
# =====================================================================

# ---- 0. Setup --------------------------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})
set.seed(42)
raw_dir <- "data/raw"
out_dir <- "data/processed"

train_path <- file.path(raw_dir, "adult.data")
test_path <- file.path(raw_dir, "adult.test")
if (!file.exists(train_path) || !file.exists(test_path)) {
  stop("Missing input files. Put adult.data and adult.test in data/raw/ and run from the project root.")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Underscores make column names easier to reference than hyphens in R.
col_names <- c("age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "sex",
               "capital_gain", "capital_loss", "hours_per_week",
               "native_country", "income")

# ---- 1. Load and combine the original partitions --------------------
read_adult <- function(path, skip = 0L) {
  read.csv(path, header = FALSE, col.names = col_names, skip = skip,
           strip.white = TRUE, stringsAsFactors = FALSE)
}
train_raw <- read_adult(train_path)
test_raw <- read_adult(test_path, skip = 1L)
cat("Original train/test rows:", nrow(train_raw), "/", nrow(test_raw), "\n")

# Adult test labels have a trailing dot; trim whitespace as well.
test_raw$income <- sub("\\.$", "", trimws(test_raw$income))
train_raw$income <- trimws(train_raw$income)
stopifnot(all(c(train_raw$income, test_raw$income) %in% c("<=50K", ">50K")))

train_raw$source <- "train"
test_raw$source <- "test"
# Row-bind only. Keep the source so the original holdout can be restored.
adult <- bind_rows(train_raw, test_raw)
original_n <- nrow(adult)
cat("Rows after combining:", original_n, "\n")

# ---- 2. Duplicates and missing values --------------------------------
# '?' means missing. Applying trimws to character columns also handles
# any remaining whitespace that strip.white did not normalize.
adult <- adult %>%
  mutate(across(where(is.character), ~ na_if(trimws(.x), "?")))

# Match on original 15 fields, not source; training rows come first.
# Consequently, any train/test overlap is removed from TEST, not train.
# Exact matches cannot establish personal identity without a unique ID;
# document this duplicate-removal policy as an analytical choice.
original_cols <- setdiff(col_names, character(0))
dup_flag <- duplicated(adult[, original_cols, drop = FALSE])
removed_by_source <- table(factor(adult$source[dup_flag], levels = c("train", "test")))
cat("Exact duplicate rows removed:", sum(dup_flag),
    "(train:", removed_by_source[["train"]],
    ", test:", removed_by_source[["test"]], ")\n")
adult <- adult[!dup_flag, , drop = FALSE]
n_after_dedupe <- nrow(adult)
cat("Rows after removing exact duplicates:", n_after_dedupe, "\n")

missing_tbl <- data.frame(
  column = names(adult),
  missing = colSums(is.na(adult)),
  percent = round(100 * colMeans(is.na(adult)), 2),
  row.names = NULL
) %>% filter(missing > 0)
print(missing_tbl)
cat("Rows containing at least one missing value:",
    sum(!complete.cases(adult)),
    "(", round(100 * mean(!complete.cases(adult)), 2), "%)\n")
print(table(workclass_missing = is.na(adult$workclass),
            occupation_missing = is.na(adult$occupation)))

# No inference about WHY occupation/workclass is missing is made here.
adult <- adult %>%
  mutate(workclass = replace_na(workclass, "Unknown"),
         occupation = replace_na(occupation, "Unknown"))

# Learn the native-country mode from TRAIN ONLY. Then apply to both sets.
mode_country <- names(sort(table(adult$native_country[adult$source == "train"]),
                           decreasing = TRUE))[1]
if (length(mode_country) != 1L || is.na(mode_country)) {
  stop("Could not determine native_country mode from training records.")
}
cat("Training-set mode of native_country:", mode_country, "\n")
adult$native_country[is.na(adult$native_country)] <- mode_country
stopifnot(!anyNA(adult))

# ---- 3. Add country -> region lookup ---------------------------------
# The four broad regions are our analytical grouping, not a UCI field.
region_list <- list(
  "North America" = c("United-States", "Canada", "Outlying-US(Guam-USVI-etc)"),
  "Latin America" = c("Mexico", "Puerto-Rico", "Cuba", "Jamaica", "Honduras",
                      "Dominican-Republic", "Ecuador", "Haiti", "Columbia",
                      "Guatemala", "Nicaragua", "El-Salvador",
                      "Trinadad&Tobago", "Peru"),
  "Europe" = c("England", "Germany", "Greece", "Italy", "Poland",
               "Portugal", "Ireland", "France", "Hungary", "Scotland",
               "Yugoslavia", "Holand-Netherlands"),
  "Asia" = c("India", "Japan", "China", "Iran", "Philippines", "Vietnam",
             "Laos", "Taiwan", "Thailand", "Cambodia", "Hong", "South")
)
region_lookup <- data.frame(
  native_country = unlist(region_list, use.names = FALSE),
  region = rep(names(region_list), lengths(region_list)),
  stringsAsFactors = FALSE
)
if (anyDuplicated(region_lookup$native_country)) {
  stop("Region lookup has duplicate country keys; left_join could multiply rows.")
}
missing_regions <- setdiff(unique(adult$native_country), region_lookup$native_country)
if (length(missing_regions)) {
  stop("Region lookup is missing countries: ", paste(missing_regions, collapse = ", "))
}
rows_before_join <- nrow(adult)
adult <- left_join(adult, region_lookup, by = "native_country")
stopifnot(nrow(adult) == rows_before_join, !anyNA(adult$region))
print(table(adult$region))

# ---- 4. Convert types ------------------------------------------------
adult$income <- ifelse(adult$income == ">50K", 1L, 0L)
cat_cols <- c("workclass", "education", "marital_status", "occupation",
              "relationship", "race", "sex", "native_country", "region",
              "source")
adult[cat_cols] <- lapply(adult[cat_cols], factor)
cat("Share of >50K:", round(100 * mean(adult$income), 2), "%\n")
str(adult)

# ---- 5. Filtering and subsetting -------------------------------------
# Preserve 'fnlwgt' and human-readable 'education' in the EDA output.
# They will be excluded from the model feature set below instead.
investors <- adult %>% filter(capital_gain > 0)
cat("Rows with capital_gain > 0:", nrow(investors), "\n")

# ---- 6. Feature engineering ------------------------------------------
adult <- adult %>% mutate(
  capital_net = capital_gain - capital_loss,
  age_group = cut(age, breaks = c(16, 25, 35, 45, 60, Inf),
                  labels = c("17-25", "26-35", "36-45", "46-60", "61+"))
)
print(table(adult$age_group))

# ---- 7. Outliers and anomalies ---------------------------------------
print(summary(adult$capital_gain))
adult$gain_99999_flag <- as.integer(adult$capital_gain == 99999)
cat("Rows with capital_gain == 99999:", sum(adult$gain_99999_flag), "\n")
cat("hours_per_week range:", range(adult$hours_per_week), "\n")
# A value of 99999 is flagged, not assumed to prove top-coding or a true income.

# ---- 8. Reshaping and leakage-safe dummy encoding --------------------
# (a) Long -> wide table for EDA
pivot_edu_sex <- adult %>%
  group_by(education_num, sex) %>%
  summarise(pct_over_50k = round(100 * mean(income), 1), .groups = "drop") %>%
  pivot_wider(names_from = sex, values_from = pct_over_50k)
print(pivot_edu_sex)

# (b) ML predictors. Source is metadata, not a predictor. Age group
# repeats age; country is represented by region. Omit capital_net from
# modeling if the original capital_gain and capital_loss are retained,
# because capital_net = capital_gain - capital_loss exactly.
model_df <- adult %>%
  select(-fnlwgt, -education, -age_group, -native_country, -capital_net)
train_model <- model_df %>% filter(source == "train") %>% select(-source)
test_model  <- model_df %>% filter(source == "test") %>% select(-source)

train_features <- train_model %>% select(-income)
test_features  <- test_model %>% select(-income)
factor_features <- names(train_features)[vapply(train_features, is.factor, logical(1))]

# Set category levels using TRAIN ONLY. Explicitly stop for unseen test
# categories rather than silently giving them NA or changing columns.
for (cl in factor_features) {
  train_features[[cl]] <- droplevels(train_features[[cl]])
  train_levels <- levels(train_features[[cl]])
  if (length(train_levels) < 2L) {
    stop("Model factor has fewer than two levels in training: ", cl)
  }
  unseen <- setdiff(unique(as.character(test_features[[cl]])), train_levels)
  if (length(unseen)) {
    stop("Unseen category in test column '", cl, "': ",
         paste(unseen, collapse = ", "),
         ". Decide how to handle new levels using training-only rules.")
  }
  test_features[[cl]] <- factor(as.character(test_features[[cl]]),
                                 levels = train_levels)
}

# model.matrix() uses treatment coding with an intercept (k-1 columns
# per factor), which we then remove from both matrices.
X_train <- model.matrix(~ ., data = train_features)[, -1, drop = FALSE]
X_test  <- model.matrix(~ ., data = test_features)[, -1, drop = FALSE]
if (!identical(colnames(X_train), colnames(X_test))) {
  stop("Training and testing dummy matrices have different columns.")
}
# Export CSV-friendly consistent column names.
clean_dummy_names <- make.names(colnames(X_train), unique = TRUE)
colnames(X_train) <- clean_dummy_names
colnames(X_test) <- clean_dummy_names
cat("Model predictors before encoding:", ncol(train_features),
    "| after encoding:", ncol(X_train), "\n")

# ---- 9. Train-only standardization demonstration ----------------------
# Do NOT scale the saved outputs. The ML workflow must fit scaling
# on its training partition (and on training folds during CV).
num_cols <- c("age", "education_num", "capital_gain", "capital_loss",
              "hours_per_week", "capital_net")
standardize_train_test <- function(train, test, cols) {
  if (!all(cols %in% names(train)) || !all(cols %in% names(test))) {
    stop("Missing numeric columns in standardize_train_test().")
  }
  mu <- vapply(train[cols], mean, numeric(1))
  sigma <- vapply(train[cols], sd, numeric(1))
  if (any(!is.finite(sigma) | sigma <= 0)) {
    stop("Cannot scale a column with zero or invalid training standard deviation: ",
         paste(names(sigma)[!is.finite(sigma) | sigma <= 0], collapse = ", "))
  }
  for (cl in cols) {
    train[[cl]] <- (train[[cl]] - mu[[cl]]) / sigma[[cl]]
    test[[cl]] <- (test[[cl]] - mu[[cl]]) / sigma[[cl]]
  }
  list(train = train, test = test, mu = mu, sigma = sigma)
}

# Test this helper only on a stratified 80/20 split of the original TRAIN
# partition. The official adult.test partition is never used for fitting.
train_for_demo <- adult %>% filter(source == "train")
pos <- which(train_for_demo$income == 1L)
neg <- which(train_for_demo$income == 0L)
demo_train_idx <- c(sample(pos, round(0.8 * length(pos))),
                    sample(neg, round(0.8 * length(neg))))
demo <- standardize_train_test(train_for_demo[demo_train_idx, , drop = FALSE],
                               train_for_demo[-demo_train_idx, , drop = FALSE],
                               num_cols)
print(round(vapply(demo$train[num_cols], mean, numeric(1)), 3))
print(round(vapply(demo$train[num_cols], sd, numeric(1)), 3))

# ---- 10. Data validation (24 documented checks) -----------------------
validation <- data.frame(check = character(), result = character())
check <- function(name, condition) {
  validation <<- rbind(validation,
                       data.frame(check = name,
                                  result = if (isTRUE(condition)) "PASS" else "FAIL"))
}
fac_cols <- c("workclass", "education", "marital_status", "occupation",
              "relationship", "race", "sex", "native_country", "region",
              "source", "age_group")

# (a) Completeness (3)
check("Original rows = cleaned rows + removed duplicates",
      original_n == nrow(adult) + sum(dup_flag))
check("No NA in cleaned columns", !anyNA(adult))
check("No '?' in cleaned categorical labels",
      !any(vapply(adult, function(x) any(as.character(x) == "?"), logical(1))))

# (b) Structure / types (5)
check("fnlwgt and education retained in clean EDA data",
      all(c("fnlwgt", "education") %in% names(adult)))
check("New columns exist", all(c("region", "capital_net", "age_group",
                                  "gain_99999_flag") %in% names(adult)))
check("Categorical columns are factors",
      all(vapply(adult[fac_cols], is.factor, logical(1))))
check("Numerical columns are numeric",
      all(vapply(adult[num_cols], is.numeric, logical(1))))
check("Income target contains only 0 and 1",
      all(adult$income %in% c(0L, 1L)))

# (c) Expected ranges in this Adult source (5)
check("age between 17 and 90", all(adult$age >= 17 & adult$age <= 90))
check("education_num between 1 and 16",
      all(adult$education_num >= 1 & adult$education_num <= 16))
check("hours_per_week between 1 and 99",
      all(adult$hours_per_week >= 1 & adult$hours_per_week <= 99))
check("capital_gain between 0 and 99999",
      all(adult$capital_gain >= 0 & adult$capital_gain <= 99999))
check("capital_loss is not negative", all(adult$capital_loss >= 0))

# (d) Category consistency (5)
check("No whitespace around factor labels",
      all(vapply(adult[fac_cols],
                 function(x) all(levels(x) == trimws(levels(x))), logical(1))))
check("workclass has 9 levels (including Unknown)", nlevels(adult$workclass) == 9)
check("occupation has 15 levels (including Unknown)", nlevels(adult$occupation) == 15)
check("region has 4 levels", nlevels(adult$region) == 4)
check("sex has 2 levels", nlevels(adult$sex) == 2)

# (e) Engineered feature consistency (2)
check("capital_net = gain - loss",
      all(adult$capital_net == adult$capital_gain - adult$capital_loss))
check("gain_99999_flag matches value 99999",
      all(adult$gain_99999_flag == as.integer(adult$capital_gain == 99999)))

# (f) Report balance; only REQUIRE both classes in both partitions (1)
share_by_source <- tapply(adult$income, adult$source, mean) * 100
cat("Share of >50K in each source (%):\n")
print(round(share_by_source, 2))
if (abs(diff(share_by_source)) >= 2) {
  warning("Train/test positive-class shares differ by at least 2 percentage points.")
}
check("Train and test each contain both classes",
      all(vapply(split(adult$income, adult$source),
                 function(x) length(unique(x)) == 2L, logical(1))))

# (g) Model output checks (3)
check("Dummy row counts match original partitions",
      nrow(X_train) == sum(adult$source == "train") &&
        nrow(X_test) == sum(adult$source == "test"))
check("Train/test dummy columns match and values are finite",
      identical(colnames(X_train), colnames(X_test)) &&
        all(is.finite(X_train)) && all(is.finite(X_test)) &&
        all(apply(X_train, 2, var) > 0))
check("Demo training means=0 and standard deviations=1",
      all(abs(vapply(demo$train[num_cols], mean, numeric(1))) < 1e-8) &&
        all(abs(vapply(demo$train[num_cols], sd, numeric(1)) - 1) < 1e-8))

print(validation, right = FALSE)
cat("Checks passed:", sum(validation$result == "PASS"), "of", nrow(validation), "\n")
stopifnot(nrow(validation) == 24L)
if (!all(validation$result == "PASS")) {
  stop("At least one data validation check failed; outputs have not been saved.")
}

# ---- 11. Save outputs ------------------------------------------------
# Save the cleaned EDA table, original partitions and train-fitted
# numeric matrices. 'source' in the combined dummy CSV is metadata only;
# NEVER feed it into a predictive model as a feature.
write.csv(adult, file.path(out_dir, "adult_clean.csv"), row.names = FALSE)
saveRDS(adult, file.path(out_dir, "adult_clean.rds"))
write.csv(adult[adult$source == "train", , drop = FALSE],
          file.path(out_dir, "adult_train_clean.csv"), row.names = FALSE)
write.csv(adult[adult$source == "test", , drop = FALSE],
          file.path(out_dir, "adult_test_clean.csv"), row.names = FALSE)

train_dummies <- data.frame(income = train_model$income, X_train,
                            check.names = FALSE)
test_dummies <- data.frame(income = test_model$income, X_test,
                           check.names = FALSE)
write.csv(train_dummies, file.path(out_dir, "adult_train_dummies.csv"),
          row.names = FALSE)
write.csv(test_dummies, file.path(out_dir, "adult_test_dummies.csv"),
          row.names = FALSE)
combined_dummies <- bind_rows(
  data.frame(source = "train", train_dummies, check.names = FALSE),
  data.frame(source = "test", test_dummies, check.names = FALSE)
)
write.csv(combined_dummies, file.path(out_dir, "adult_dummies.csv"),
          row.names = FALSE)
write.csv(validation, file.path(out_dir, "validation_report.csv"),
          row.names = FALSE)

# Round-trip: the RDS object must be identical; CSV must preserve size
# and selected numeric/label values, not just the number of columns.
stopifnot(identical(readRDS(file.path(out_dir, "adult_clean.rds")), adult))
csv_back <- read.csv(file.path(out_dir, "adult_clean.csv"),
                     stringsAsFactors = FALSE)
stopifnot(nrow(csv_back) == nrow(adult), ncol(csv_back) == ncol(adult),
          identical(as.integer(csv_back$income), as.integer(adult$income)),
          isTRUE(all.equal(as.numeric(csv_back$capital_net),
                           as.numeric(adult$capital_net))))
cat("\nFinal cleaned data:", nrow(adult), "rows x", ncol(adult), "columns\n")
cat("ML train/test rows:", nrow(X_train), "/", nrow(X_test), "\n")
cat("Files saved in", out_dir, "\n")
