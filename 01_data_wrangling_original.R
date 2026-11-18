# =====================================================================
# IS 6114 - Data Analytics Project
# Part 1: Data Wrangling - UCI Adult (Census Income) dataset
#
# Input : data/raw/adult.data   (download from UCI, dataset id 2)
#         data/raw/adult.test
# Output: data/processed/adult_clean.csv   (clean, NOT scaled - for EDA)
#         data/processed/adult_clean.rds   (same data, keeps factor types)
#         data/processed/adult_dummies.csv (one-hot encoded, NOT scaled - for ML)
#
# Run from the project root:  Rscript code/01_data_wrangling.R
# =====================================================================

# ---- 0. Setup --------------------------------------------------------
# install.packages(c("dplyr", "tidyr"))   # run once if not installed
library(dplyr)
library(tidyr)

set.seed(42)   # same results every time (reproducibility)

raw_dir <- "data/raw"
out_dir <- "data/processed"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# The raw files have no header. R does not like "-" in names,
# so we use "_" instead of the original names (e.g. education-num).
col_names <- c("age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "sex",
               "capital_gain", "capital_loss", "hours_per_week",
               "native_country", "income")


# ---- 1. Load and combine the two files -------------------------------
read_adult <- function(path, skip = 0) {
  read.csv(path,
           header           = FALSE,
           col.names        = col_names,
           skip             = skip,
           strip.white      = TRUE,    # removes the extra space before values
           stringsAsFactors = FALSE)
}

train_raw <- read_adult(file.path(raw_dir, "adult.data"))
test_raw  <- read_adult(file.path(raw_dir, "adult.test"), skip = 1)  # 1st line is not data

cat("Train rows:", nrow(train_raw), " | Test rows:", nrow(test_raw), "\n")
# expected: 32,561 and 16,281

# Test labels end with a dot (">50K.") -> remove it
test_raw$income <- sub("\\.$", "", test_raw$income)
stopifnot(all(c(train_raw$income, test_raw$income) %in% c("<=50K", ">50K")))

# Keep track of where each row came from, then stack the two files
train_raw$source <- "train"
test_raw$source  <- "test"
adult <- bind_rows(train_raw, test_raw)

cat("Rows after combining:", nrow(adult), "\n")   # expected: 48,842


# ---- 2. Duplicates and missing values --------------------------------
# Missing values are written as "?" -> convert to NA
adult <- adult %>%
  mutate(across(where(is.character), ~ na_if(.x, "?")))

# Duplicates: check all columns EXCEPT 'source', otherwise the same
# person in train and test would not be detected as a duplicate.
# (Done before dropping any column, see the report section 3.2.)
dup_flag <- duplicated(adult[, setdiff(names(adult), "source")])
cat("Duplicate rows removed:", sum(dup_flag), "\n")
adult <- adult[!dup_flag, ]
n_after_dedupe <- nrow(adult)   # used again in the validation step
cat("Rows after removing duplicates:", n_after_dedupe, "\n")

# Missing values table (Table 2 in the report)
missing_tbl <- data.frame(
  column  = names(adult),
  missing = colSums(is.na(adult)),
  percent = round(100 * colMeans(is.na(adult)), 2),
  row.names = NULL
) %>% filter(missing > 0)
print(missing_tbl)

cat("Rows with at least one missing value:",
    sum(!complete.cases(adult)), "(",
    round(100 * mean(!complete.cases(adult)), 2), "% )\n")

# Do workclass and occupation go missing together? (supports our decision)
print(table(workclass_missing  = is.na(adult$workclass),
            occupation_missing = is.na(adult$occupation)))

# Decision:
#   workclass, occupation -> new category "Unknown" (missing is not random)
#   native_country        -> mode (most common country)
adult <- adult %>%
  mutate(workclass  = replace_na(workclass,  "Unknown"),
         occupation = replace_na(occupation, "Unknown"))

mode_country <- names(sort(table(adult$native_country), decreasing = TRUE))[1]
cat("Mode of native_country:", mode_country, "\n")
adult$native_country[is.na(adult$native_country)] <- mode_country

stopifnot(sum(is.na(adult)) == 0)


# ---- 3. Merging with a lookup table (country -> region) --------------
# Done AFTER filling native_country, so no row gets an empty region.
region_list <- list(
  "North America" = c("United-States", "Canada", "Outlying-US(Guam-USVI-etc)"),
  "Latin America" = c("Mexico", "Puerto-Rico", "Cuba", "Jamaica", "Honduras",
                      "Dominican-Republic", "Ecuador", "Haiti", "Columbia",
                      "Guatemala", "Nicaragua", "El-Salvador",
                      "Trinadad&Tobago", "Peru"),
  "Europe"        = c("England", "Germany", "Greece", "Italy", "Poland",
                      "Portugal", "Ireland", "France", "Hungary", "Scotland",
                      "Yugoslavia", "Holand-Netherlands"),
  "Asia"          = c("India", "Japan", "China", "Iran", "Philippines",
                      "Vietnam", "Laos", "Taiwan", "Thailand", "Cambodia",
                      "Hong", "South")   # "South" is unclear in the source; we assume South Korea
)

region_lookup <- data.frame(
  native_country = unlist(region_list, use.names = FALSE),
  region         = rep(names(region_list), lengths(region_list))
)

adult <- left_join(adult, region_lookup, by = "native_country")

# Every country must have a region (stops the script if one is missing)
stopifnot(!any(is.na(adult$region)))
print(table(adult$region))


# ---- 4. Data type conversion -----------------------------------------
adult$income <- ifelse(adult$income == ">50K", 1L, 0L)   # 1 = >50K, 0 = <=50K

cat_cols <- c("workclass", "education", "marital_status", "occupation",
              "relationship", "race", "sex", "native_country", "region",
              "source")
adult[cat_cols] <- lapply(adult[cat_cols], factor)

cat("Share of >50K:", round(100 * mean(adult$income), 2), "%\n")
str(adult)


# ---- 5. Filtering and subsetting -------------------------------------
# fnlwgt   = census sampling weight, not a feature of the person
# education = same information as education_num (text vs number)
adult <- adult %>% select(-fnlwgt, -education)

# Example subset: people with investment income
investors <- adult %>% filter(capital_gain > 0)
cat("People with capital_gain > 0:", nrow(investors), "\n")


# ---- 6. Feature engineering ------------------------------------------
adult <- adult %>%
  mutate(
    capital_net = capital_gain - capital_loss,
    age_group   = cut(age,
                      breaks = c(16, 25, 35, 45, 60, Inf),
                      labels = c("17-25", "26-35", "36-45", "46-60", "61+"))
  )
print(table(adult$age_group))


# ---- 7. Outliers and anomalies ---------------------------------------
print(summary(adult$capital_gain))
adult$gain_99999_flag <- as.integer(adult$capital_gain == 99999)
cat("Rows with capital_gain = 99999:", sum(adult$gain_99999_flag), "\n")
cat("hours_per_week range:", range(adult$hours_per_week), "\n")


# ---- 8. Reshaping ----------------------------------------------------
# (a) Long -> wide: % earning >50K by education level and sex
pivot_edu_sex <- adult %>%
  group_by(education_num, sex) %>%
  summarise(pct_over_50k = round(100 * mean(income), 1), .groups = "drop") %>%
  pivot_wider(names_from = sex, values_from = pct_over_50k)
print(pivot_edu_sex)

# (b) One-hot encoding for modeling (dummy variables, k-1 per factor)
#     source    -> not a feature
#     age_group -> repeats age
#     native_country -> replaced by region (41 levels is too many)
model_df  <- adult %>% select(-source, -age_group, -native_country)
X_dummies <- model.matrix(income ~ ., data = model_df)[, -1]   # drop intercept
cat("Columns before encoding:", ncol(model_df) - 1,
    " | after encoding:", ncol(X_dummies), "\n")


# ---- 9. Standardization ----------------------------------------------
# z = (x - mean) / sd
# IMPORTANT: mean and sd must come from the TRAINING part only,
# otherwise the test data leaks into the model (data leakage).
# So we do NOT scale the saved file. The ML step uses this function
# after the train/test split.
num_cols <- c("age", "education_num", "capital_gain", "capital_loss",
              "hours_per_week", "capital_net")

standardize_train_test <- function(train, test, cols) {
  mu    <- sapply(train[cols], mean)
  sigma <- sapply(train[cols], sd)
  for (cl in cols) {
    train[[cl]] <- (train[[cl]] - mu[[cl]]) / sigma[[cl]]
    test[[cl]]  <- (test[[cl]]  - mu[[cl]]) / sigma[[cl]]
  }
  list(train = train, test = test, mu = mu, sigma = sigma)
}

# Demonstration only (the real split is done in the ML part):
# stratified 80/20 split, then check mean = 0 and sd = 1 on train
pos <- which(adult$income == 1)
neg <- which(adult$income == 0)
train_idx <- c(sample(pos, round(0.8 * length(pos))),
               sample(neg, round(0.8 * length(neg))))
demo <- standardize_train_test(adult[train_idx, ], adult[-train_idx, ], num_cols)
print(round(sapply(demo$train[num_cols], mean), 3))   # all ~0
print(round(sapply(demo$train[num_cols], sd), 3))     # all ~1


# ---- 10. Data validation ---------------------------------------------
# Each check is recorded as PASS or FAIL (Table 3 in the report).
# If any check fails, the script stops BEFORE saving, so bad data
# is never passed to the EDA and ML parts.
validation <- data.frame(check = character(), result = character())

check <- function(name, condition) {
  validation <<- rbind(validation,
                       data.frame(check  = name,
                                  result = ifelse(isTRUE(condition), "PASS", "FAIL")))
}

fac_cols <- c("workclass", "marital_status", "occupation", "relationship",
              "race", "sex", "native_country", "region", "source", "age_group")

# (a) Completeness
check("Row count equals rows after removing duplicates",
      nrow(adult) == n_after_dedupe)
check("No missing values (NA) in any column",
      sum(is.na(adult)) == 0)
check("No '?' values left",
      !any(sapply(adult, function(x) any(as.character(x) == "?"))))

# (b) Structure and types
check("fnlwgt and education were removed",
      !any(c("fnlwgt", "education") %in% names(adult)))
check("New columns exist (region, capital_net, age_group, gain_99999_flag)",
      all(c("region", "capital_net", "age_group", "gain_99999_flag") %in% names(adult)))
check("Categorical columns are factors",
      all(sapply(adult[fac_cols], is.factor)))
check("Numerical columns are numeric",
      all(sapply(adult[num_cols], is.numeric)))
check("Target income contains only 0 and 1",
      all(adult$income %in% c(0, 1)))

# (c) Value ranges
check("age between 17 and 90",            all(adult$age >= 17 & adult$age <= 90))
check("education_num between 1 and 16",   all(adult$education_num >= 1 & adult$education_num <= 16))
check("hours_per_week between 1 and 99",  all(adult$hours_per_week >= 1 & adult$hours_per_week <= 99))
check("capital_gain between 0 and 99999", all(adult$capital_gain >= 0 & adult$capital_gain <= 99999))
check("capital_loss is not negative",     all(adult$capital_loss >= 0))

# (d) Consistency of categories (no extra spaces, expected number of levels)
check("No extra spaces in category labels",
      all(sapply(adult[fac_cols], function(x) all(levels(x) == trimws(levels(x))))))
check("workclass has 9 levels (8 + Unknown)",   nlevels(adult$workclass) == 9)
check("occupation has 15 levels (14 + Unknown)", nlevels(adult$occupation) == 15)
check("region has 4 levels",                     nlevels(adult$region) == 4)
check("sex has 2 levels",                        nlevels(adult$sex) == 2)

# (e) Logic of the new features
check("capital_net = capital_gain - capital_loss",
      all(adult$capital_net == adult$capital_gain - adult$capital_loss))
check("Flag matches rows with capital_gain = 99999",
      all(adult$gain_99999_flag == as.integer(adult$capital_gain == 99999)))

# (f) Class balance is similar in the train and test files (< 2 points)
share_by_source <- tapply(adult$income, adult$source, mean) * 100
print(round(share_by_source, 2))
check("Share of >50K is similar in train and test (difference < 2 points)",
      abs(diff(share_by_source)) < 2)

# (g) Data ready for modeling
check("Dummy matrix has the same number of rows as the clean data",
      nrow(X_dummies) == nrow(adult))
check("No constant (zero-variance) dummy columns",
      all(apply(X_dummies, 2, var) > 0))
check("Standardized train variables have mean 0 and sd 1",
      all(abs(sapply(demo$train[num_cols], mean)) < 1e-8) &&
      all(abs(sapply(demo$train[num_cols], sd) - 1) < 1e-8))

print(validation, right = FALSE)
cat("Checks passed:", sum(validation$result == "PASS"), "of", nrow(validation), "\n")
stopifnot("Some validation checks failed - see the table above" =
            all(validation$result == "PASS"))


# ---- 11. Save outputs ------------------------------------------------
write.csv(adult, file.path(out_dir, "adult_clean.csv"), row.names = FALSE)
saveRDS(adult, file.path(out_dir, "adult_clean.rds"))   # keeps factor types
write.csv(data.frame(income = model_df$income, X_dummies),
          file.path(out_dir, "adult_dummies.csv"), row.names = FALSE)
write.csv(validation, file.path(out_dir, "validation_report.csv"), row.names = FALSE)

# Round-trip check: read the saved files back and compare
stopifnot("RDS file is not identical to the data in memory" =
            identical(readRDS(file.path(out_dir, "adult_clean.rds")), adult))
csv_back <- read.csv(file.path(out_dir, "adult_clean.csv"))
stopifnot("CSV file has a different size" =
            nrow(csv_back) == nrow(adult) && ncol(csv_back) == ncol(adult))

cat("\nFinal clean data:", nrow(adult), "rows x", ncol(adult), "columns\n")
cat("Files saved in", out_dir, "\n")
