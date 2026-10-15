# ==================================
# Video Game Sales: descriptive & inferential statistics
# ==================================
#
# Cleans the raw Kaggle dataset, engineers the grouping features used throughout
# the project, and runs the descriptive / inferential analysis.
#
# Run from the repository root:
#   Rscript statistics/vgsales_stats.R
#
# Outputs:
#   data/processed/vgsales_clean.csv   cleaned dataset (input of the ML notebook)
#   figures/statistics/*.png           all figures
#   results/tables/*.csv               descriptive tables and test results
#
# Sales are expressed in millions of units sold.

library(tidyverse)
library(moments)
library(patchwork)
library(scales)
library(boot)
library(rstatix)
library(corrplot)
library(broom)

# Allow running the script from inside statistics/ as well (e.g. RStudio "Source")
if (!file.exists("data/raw/vgsales.csv") && file.exists("../data/raw/vgsales.csv")) {
  setwd("..")
}

RAW_PATH = "data/raw/vgsales.csv"
CLEAN_PATH = "data/processed/vgsales_clean.csv"
FIG_DIR = "figures/statistics"
TABLE_DIR = "results/tables"

dir.create(dirname(CLEAN_PATH), recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)

fig_path = function(name) file.path(FIG_DIR, name)
save_table = function(df, name) write_csv(df, file.path(TABLE_DIR, name))

set.seed(42)

# The raw file encodes missing values as the string "N/A"
vgsales = read_csv(RAW_PATH, na = c("", "NA", "N/A"), show_col_types = FALSE)

# ==================================
# Function for descriptive statistics
# ==================================

desc_stats = function(data, vars) {
  map_dfr(vars, function(v) {
    x = data[[v]]
    x = x[!is.na(x)]
    n = length(x)

    mn = mean(x)
    med = median(x)
    sd_x = sd(x)
    q1 = quantile(x, 0.25)
    q3 = quantile(x, 0.75)
    iqr_x = q3 - q1

    g1 = skewness(x)
    g2 = kurtosis(x) - 3   # moments::kurtosis is Pearson's; subtract 3 for excess kurtosis
    pearson = 3 * (mn - med) / sd_x

    tibble(
      Variable = v,
      N = n,
      Mean = round(mn, 3),
      Median = round(med, 3),
      SD = round(sd_x, 3),
      Min = round(min(x), 3),
      Max = round(max(x), 3),
      Q1 = round(q1, 3),
      Q3 = round(q3, 3),
      IQR = round(iqr_x, 3),
      CV = round(sd_x / mn * 100, 2),
      Fisher = round(g1, 3),
      Kurtosis = round(g2, 3),
      Pearson_Skew = round(pearson, 3)
    )
  })
}

# ==================================
# Missing values
# ==================================

# There are 271 missing values in "Year" (~1.6% of the data) and 58 in "Publisher"
# (<1%). Even though they are few, we handle them before making any comparison.

# For "Year" we impute the median release year of the same platform: games for a
# given platform are released within a similar time window, so the platform median
# is a reasonable estimate of the release date.

# For "Publisher", instead of dropping the 58 rows, we label them as "Unknown"
# (a category that already exists in the raw data).

missing_values = vgsales |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing") |>
  mutate(Pct = round(Missing / nrow(vgsales) * 100, 2))

print(missing_values)
save_table(missing_values, "missing_values.csv")

# YEAR
vgsales = vgsales |>
  group_by(Platform) |>
  mutate(Year = if_else(
    is.na(Year),
    median(Year, na.rm = TRUE),
    Year
  )) |>
  ungroup() |>
  filter(!is.na(Year))

# PUBLISHER
vgsales = vgsales |>
  mutate(Publisher = replace_na(Publisher, "Unknown"))

stopifnot(!anyNA(vgsales))

# Categorical variables as factors
vgsales = vgsales |>
  mutate(
    Platform = as.factor(Platform),
    Genre = as.factor(Genre),
    Publisher = as.factor(Publisher)
  )

# ==================================
# Groups
# ==================================

# Years into decades

vgsales = vgsales |>
  mutate(
    Decade = case_when(
      Year < 1990 ~ "1980s",
      Year < 2000 ~ "1990s",
      Year < 2010 ~ "2000s",
      TRUE ~ "2010s"
    ),
    Decade = factor(Decade, levels = c("1980s", "1990s", "2000s", "2010s"))
  )

print(table(vgsales$Decade))

# Top 7 genres by number of games; the remaining ones are grouped as "Other"

top_genres = vgsales |>
  count(Genre, sort = TRUE) |>
  slice_head(n = 7) |>
  pull(Genre) |>
  as.character()

vgsales = vgsales |>
  mutate(
    Genre_Group = if_else(
      as.character(Genre) %in% top_genres,
      as.character(Genre),
      "Other"
    ),
    Genre_Group = factor(Genre_Group, levels = c(top_genres, "Other"))
  )

print(table(vgsales$Genre_Group))

# Platform -> console generation

gen_map = c(
  "2600"="Gen3-4", "NES"="Gen3-4", "SNES"="Gen3-4", "GB"="Gen3-4",
  "GEN"="Gen3-4", "SCD"="Gen3-4", "NG"="Gen3-4",
  "PS"="Gen5", "N64"="Gen5", "SAT"="Gen5", "3DO"="Gen5",
  "PS2"="Gen6", "XB"="Gen6", "GC"="Gen6", "DC"="Gen6", "GBA"="Gen6",
  "PS3"="Gen7", "X360"="Gen7", "Wii"="Gen7", "DS"="Gen7", "PSP"="Gen7",
  "PS4"="Gen8", "XOne"="Gen8", "WiiU"="Gen8", "3DS"="Gen8", "PSV"="Gen8",
  "PC"="PC/Other", "WS"="PC/Other", "TG16"="PC/Other", "PCFX"="PC/Other", "GG"="PC/Other"
)

unmapped = setdiff(levels(vgsales$Platform), names(gen_map))
if (length(unmapped) > 0) stop("Platforms missing from gen_map: ", paste(unmapped, collapse = ", "))

vgsales = vgsales |>
  mutate(
    Platform_Gen = factor(unname(gen_map[as.character(Platform)]),
                          levels = c("Gen3-4", "Gen5", "Gen6", "Gen7", "Gen8", "PC/Other"))
  )

print(table(vgsales$Platform_Gen))

# Publishers by number of games: top 10 = Major, 11-50 = Mid, rest = Small

publisher_count = vgsales |> count(Publisher, sort = TRUE)
major_publishers = publisher_count |> slice_head(n = 10) |> pull(Publisher)
mid_publishers = publisher_count |> slice(11:50) |> pull(Publisher)

vgsales = vgsales |>
  mutate(
    Publisher_Tier = case_when(
      Publisher %in% major_publishers ~ "Major",
      Publisher %in% mid_publishers ~ "Mid",
      TRUE ~ "Small"
    ),
    Publisher_Tier = factor(Publisher_Tier, levels = c("Major", "Mid", "Small"))
  )

print(table(vgsales$Publisher_Tier))

# Sales tiers (Global_Sales is in millions of units):
#   Niche        <= 0.17M  (0.17 is the median, so half of the games are Niche)
#   Moderate     0.17M - 1M
#   Hit          1M - 5M
#   Blockbuster  > 5M

vgsales = vgsales |>
  mutate(
    Sales_Tier = cut(
      Global_Sales,
      breaks = c(0, 0.17, 1, 5, Inf),
      labels = c("Niche", "Moderate", "Hit", "Blockbuster"),
      right = TRUE
    )
  )

print(table(vgsales$Sales_Tier))

# Log scale for sales (the distribution is extremely right-skewed)

vgsales = vgsales |>
  mutate(
    Log_NA_Sales = log1p(NA_Sales),
    Log_EU_Sales = log1p(EU_Sales),
    Log_JP_Sales = log1p(JP_Sales),
    Log_Other_Sales = log1p(Other_Sales),
    Log_Global_Sales = log1p(Global_Sales)
  )

# Share of each region in global sales

vgsales = vgsales |>
  mutate(
    NA_Share = round(NA_Sales / Global_Sales, 3),
    EU_Share = round(EU_Sales / Global_Sales, 3),
    JP_Share = round(JP_Sales / Global_Sales, 3),
    Other_Share = round(Other_Sales / Global_Sales, 3)
  )

stopifnot(!anyNA(vgsales))

# ==================================
# Clean csv
# ==================================

write_csv(vgsales, CLEAN_PATH)

# ==================================
# Descriptive statistics
# ==================================

raw_vars = c("Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales")
desc_raw = desc_stats(vgsales, raw_vars)
print(desc_raw)
save_table(desc_raw, "descriptive_raw_sales.csv")

log_vars = c("Log_Global_Sales", "Log_NA_Sales", "Log_EU_Sales", "Log_JP_Sales", "Log_Other_Sales")
desc_log = desc_stats(vgsales, log_vars)
print(desc_log)
save_table(desc_log, "descriptive_log_sales.csv")

# ==================================
# Stratified group tables
# ==================================

strat_table = function(data, group_var) {
  data |>
    group_by(.data[[group_var]]) |>
    summarise(
      n = n(),
      pct = round(n() / nrow(data) * 100, 1),
      mean_global = round(mean(Global_Sales), 3),
      median_global = round(median(Global_Sales), 3),
      sd_global = round(sd(Global_Sales), 3),
      skewness = round(skewness(Global_Sales), 3),
      min_global = round(min(Global_Sales), 3),
      max_global = round(max(Global_Sales), 3),
      median_NA = round(median(NA_Sales), 3),
      median_EU = round(median(EU_Sales), 3),
      median_JP = round(median(JP_Sales), 3),
      .groups = "drop"
    )
}

strat_genre = strat_table(vgsales, "Genre_Group") |> arrange(desc(median_global))
strat_publisher = strat_table(vgsales, "Publisher_Tier")
strat_decade = strat_table(vgsales, "Decade")
strat_platform = strat_table(vgsales, "Platform_Gen")

print(strat_genre)
print(strat_publisher)
print(strat_decade)
print(strat_platform)

save_table(strat_genre, "strata_genre.csv")
save_table(strat_publisher, "strata_publisher_tier.csv")
save_table(strat_decade, "strata_decade.csv")
save_table(strat_platform, "strata_platform_gen.csv")

# ==================================
# Visualization
# ==================================

# Theme
shared_theme = function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10, color = "grey45"),
      axis.title = element_text(size = 10, color = "grey30"),
      axis.text = element_text(size = 9,  color = "grey40"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92"),
      plot.background = element_rect(fill = "white", color = NA),
      legend.position = "none"
    )
}

# Palettes
pal_genre = c("#3266AD","#0F6E56","#993C1D","#7B4EA0",
              "#C4820A","#1F8FAB","#B03060","#666666")
pal_decade = c("#C4820A","#0F6E56","#3266AD","#993C1D")
pal_platform = c("#3266AD","#0F6E56","#993C1D",
                 "#7B4EA0","#C4820A","#666666")
pal_tier = c("#3266AD","#0F6E56","#993C1D")
pal_region = c("#3266AD","#0F6E56","#993C1D")

# Axis limits to keep extreme outliers from flattening the plots
cap995 = function(x, p = 0.995) quantile(x, p, na.rm = TRUE)
cap99 = function(x, p = 0.99) quantile(x, p, na.rm = TRUE)
cap95 = function(x, p = 0.95) quantile(x, p, na.rm = TRUE)
cap90 = function(x, p = 0.90) quantile(x, p, na.rm = TRUE)

# ==================================
# P1: Distribution histograms
# ==================================

p1a = ggplot(vgsales, aes(x = Global_Sales)) +
  geom_histogram(bins = 60, fill = "#B03060",
                 color = "white", linewidth = 0.15) +
  coord_cartesian(xlim = c(0, cap995(vgsales$Global_Sales))) +
  scale_y_continuous(labels = label_comma()) +
  labs(title = "Global Sales",
       subtitle = "Capped at .995 percentile",
       x = "Sales (millions of units)", y = "Count") +
  shared_theme()

p1b = ggplot(vgsales, aes(x = Log_Global_Sales)) +
  geom_histogram(bins = 50, fill = "#993C1D",
                 color = "white", linewidth = 0.15) +
  scale_y_continuous(labels = label_comma()) +
  labs(title = "Logarithmic Global Sales",
       subtitle = "After transformation, uncapped",
       x = "log1p(Sales)", y = "Count") +
  shared_theme()

ggsave(fig_path("p1_distributions.png"), plot = p1a | p1b,
       width = 12, height = 5, dpi = 150)

# ==================================
# P2: Density by decade
# ==================================

p2 = ggplot(vgsales, aes(x = Log_Global_Sales,
                         color = Decade, fill = Decade)) +
  geom_density(alpha = 0.15, linewidth = 0.8) +
  coord_cartesian(xlim = c(0, cap99(vgsales$Log_Global_Sales))) +
  scale_color_manual(values = pal_decade) +
  scale_fill_manual(values  = pal_decade) +
  labs(title = "Sales distribution by decade",
       subtitle = "Density plot, capped at 99th percentile",
       x = "log1p(Global Sales)", y = "Density",
       color = "Decade", fill = "Decade") +
  shared_theme() +
  theme(legend.position = "right")

ggsave(fig_path("p2_density_decade.png"), plot = p2, width = 9, height = 5, dpi = 150)

# ==================================
# P3 & P4: Genre
# ==================================

# violin
p3 = vgsales |>
  mutate(Genre_Group = fct_reorder(Genre_Group, Log_Global_Sales, median)) |>
  ggplot(aes(x = Genre_Group, y = Log_Global_Sales,
             fill = Genre_Group)) +
  geom_violin(alpha = 0.55, trim = TRUE, color = NA) +
  geom_boxplot(width = 0.12, outlier.shape = NA,
               fill = "white", color = "grey30", linewidth = 0.5) +
  coord_flip(ylim = c(0, cap95(vgsales$Log_Global_Sales))) +
  scale_fill_manual(values = pal_genre) +
  labs(title    = "Global sales by genre",
       subtitle = "Capped at 95th percentile",
       x = NULL, y = "log1p(Global Sales)") +
  shared_theme()

ggsave(fig_path("p3_violin_genre.png"), plot = p3, width = 10, height = 6, dpi = 150)

# density by genre (same idea as the decade density), only for the 5 genres most relevant to the study
p4 = vgsales |>
  filter(Genre_Group %in% c("Action","Sports","Role-Playing","Shooter","Racing")) |>
  ggplot(aes(x = Log_Global_Sales,
             color = Genre_Group, fill = Genre_Group)) +
  geom_density(alpha = 0.12, linewidth = 0.8) +
  coord_cartesian(xlim = c(0, cap99(vgsales$Log_Global_Sales))) +
  scale_color_manual(values = pal_genre[1:5]) +
  scale_fill_manual(values  = pal_genre[1:5]) +
  labs(title    = "Sales density, top 5 genres",
       subtitle = "Shows genre density, capped at 99th percentile",
       x = "log1p(Global Sales)", y = "Density",
       color = NULL, fill = NULL) +
  shared_theme() +
  theme(legend.position = "right")

ggsave(fig_path("p4_density_genre.png"), plot = p4, width = 9, height = 5, dpi = 150)

# ==================================
# P5 & P6: Publisher
# ==================================

# violin
p5 = ggplot(vgsales, aes(x = Publisher_Tier, y = Log_Global_Sales,
                         fill = Publisher_Tier)) +
  geom_violin(alpha = 0.55, trim = TRUE, color = NA) +
  geom_boxplot(width = 0.12, outlier.shape = NA,
               fill = "white", color = "grey30", linewidth = 0.5) +
  coord_cartesian(ylim = c(0, cap99(vgsales$Log_Global_Sales))) +
  scale_fill_manual(values = pal_tier) +
  labs(title    = "Global sales by publisher tier",
       subtitle = "Capped at 99th percentile",
       x = NULL, y = "log1p(Global Sales)") +
  shared_theme()

ggsave(fig_path("p5_violin_publisher.png"), plot = p5, width = 8, height = 5, dpi = 150)

# density
p6 = ggplot(vgsales, aes(x = Log_Global_Sales,
                         color = Publisher_Tier, fill = Publisher_Tier)) +
  geom_density(alpha = 0.12, linewidth = 0.8) +
  coord_cartesian(xlim = c(0, cap95(vgsales$Log_Global_Sales))) +
  scale_color_manual(values = pal_tier) +
  scale_fill_manual(values  = pal_tier) +
  labs(title    = "Sales density by publisher tier",
       subtitle = "Capped at 95th percentile",
       x = "log1p(Global Sales)", y = "Density",
       color = NULL, fill = NULL) +
  shared_theme() +
  theme(legend.position = "right")

ggsave(fig_path("p6_density_publisher.png"), plot = p6, width = 9, height = 5, dpi = 150)

# ==================================
# P7 & P8: Time
# ==================================

# violin by decade
p7 = ggplot(vgsales, aes(x = Decade, y = Log_Global_Sales,
                         fill = Decade)) +
  geom_violin(alpha = 0.55, trim = TRUE, color = NA) +
  geom_boxplot(width = 0.12, outlier.shape = NA,
               fill = "white", color = "grey30", linewidth = 0.5) +
  coord_cartesian(ylim = c(0, cap99(vgsales$Log_Global_Sales))) +
  scale_fill_manual(values = pal_decade) +
  labs(title    = "Global sales by decade",
       subtitle = "Capped at 99th percentile",
       x = NULL, y = "log1p(Global Sales)") +
  shared_theme()

ggsave(fig_path("p7_violin_decade.png"), plot = p7, width = 9, height = 5, dpi = 150)

# histograms with facet_wrap
p8 = ggplot(vgsales, aes(x = Log_Global_Sales, fill = Decade)) +
  geom_histogram(bins = 40, color = "white", linewidth = 0.1) +
  coord_cartesian(xlim = c(0, cap995(vgsales$Log_Global_Sales))) +
  facet_wrap(~ Decade, ncol = 2, scales = "free_y") +
  scale_fill_manual(values = pal_decade) +
  scale_y_continuous(labels = label_comma()) +
  labs(title    = "Sales distribution by decade",
       subtitle = "Capped at .995 percentile",
       x = "log1p(Global Sales)", y = "Count") +
  shared_theme()

ggsave(fig_path("p8_hist_facet_decade.png"), plot = p8, width = 10, height = 7, dpi = 150)

# ==================================
# P9: Platform_Gen
# ==================================

# violin by platform generation
p9 = ggplot(vgsales, aes(x = Platform_Gen, y = Log_Global_Sales,
                         fill = Platform_Gen)) +
  geom_violin(alpha = 0.55, trim = TRUE, color = NA) +
  geom_boxplot(width = 0.12, outlier.shape = NA,
               fill = "white", color = "grey30", linewidth = 0.5) +
  coord_cartesian(ylim = c(0, cap95(vgsales$Log_Global_Sales))) +
  scale_fill_manual(values = pal_platform) +
  labs(title    = "Global sales by platform gen",
       subtitle = "Capped at 95th percentile",
       x = NULL, y = "log1p(Global Sales)") +
  shared_theme()

ggsave(fig_path("p9_violin_platform.png"), plot = p9, width = 10, height = 5, dpi = 150)

# ==================================
# P10: Regional
# ==================================

# Density by region
p10 = vgsales |>
  select(NA_Sales, EU_Sales, JP_Sales) |>
  pivot_longer(everything(),
               names_to  = "Region",
               values_to = "Sales") |>
  mutate(
    Region = recode(Region,
                    "NA_Sales" = "North America",
                    "EU_Sales" = "Europe",
                    "JP_Sales" = "Japan"),
    Log_Sales = log1p(Sales)
  ) |>
  ggplot(aes(x = Log_Sales, color = Region, fill = Region)) +
  geom_density(alpha = 0.12, linewidth = 0.8) +
  coord_cartesian(xlim = c(0, cap90(log1p(vgsales$NA_Sales)))) +
  scale_color_manual(values = pal_region) +
  scale_fill_manual(values = pal_region) +
  labs(title = "Regional sales distributions",
       subtitle = "Capped at 90th percentile",
       x = "log1p(Sales)", y = "Density",
       color = NULL, fill = NULL) +
  shared_theme() +
  theme(legend.position = "right")

ggsave(fig_path("p10_density_regional.png"), plot = p10, width = 9, height = 5, dpi = 150)

# ==================================
# Confidence intervals
# ==================================

# Percentile bootstrap CIs for the median (non-parametric, no normality assumption).

ci_vars = c("Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales")
ci_levels = c(0.85, 0.95, 0.99)

ci_median = map_dfr(ci_vars, function(v) {
  x = vgsales[[v]]
  bootstrap_fn = function(data, i) median(data[i])
  b = boot(x, bootstrap_fn, R = 1000)
  map_dfr(ci_levels, function(lv) {
    ci = boot.ci(b, conf = lv, type = "perc")
    tibble(
      Variable = v,
      Level = paste0(lv * 100, "%"),
      Median = round(median(x), 3),
      Lower = round(ci$percent[4], 4),
      Upper = round(ci$percent[5], 4)
    )
  })
})

print(ci_median)
save_table(ci_median, "bootstrap_ci_median.csv")

# ==================================
# Hypothesis tests
# ==================================

# Shapiro-Wilk only accepts up to 5000 observations, so we test a random sample.

vars_shapiro = c("Global_Sales", "NA_Sales", "EU_Sales",
                 "JP_Sales", "Other_Sales", "Log_Global_Sales")

vgsales_sample = vgsales |> slice_sample(n = 5000)

normality_table = map_dfr(vars_shapiro, function(v) {
  x = vgsales_sample[[v]]
  test = shapiro.test(x)
  tibble(
    variable = v,
    w = round(unname(test$statistic), 4),
    p_value = format.pval(test$p.value, digits = 3),
    normal = if_else(test$p.value > 0.05, "Yes", "No")
  )
})

print(normality_table)
save_table(normality_table, "shapiro_normality.csv")

# ==================================
# Non-parametric statistics
# ==================================

group_vars = c("Publisher_Tier", "Decade", "Platform_Gen", "Genre_Group")

# -- Kruskal-Wallis --

kruskal_results = map_dfr(group_vars, function(g) {
  test = kruskal.test(vgsales$Global_Sales, vgsales[[g]])
  tibble(
    group = g,
    H = round(unname(test$statistic), 2),
    df = unname(test$parameter),
    p_value = format.pval(test$p.value, digits = 3)
  )
})

print(kruskal_results)
save_table(kruskal_results, "kruskal_wallis.csv")

# -- Dunn post-hoc (Bonferroni) --

dunn_results = map_dfr(group_vars, function(g) {
  vgsales |>
    dunn_test(as.formula(paste("Global_Sales ~", g)),
              p.adjust.method = "bonferroni") |>
    select(group1, group2, statistic, p, p.adj, p.adj.signif) |>
    mutate(group = g, .before = 1)
})

print(dunn_results, n = Inf)
save_table(dunn_results, "dunn_posthoc.csv")

# Auxiliary summary to interpret the tests

group_summary = function(data, group_var) {
  data |>
    group_by(across(all_of(group_var))) |>
    summarise(
      n = n(),
      median = round(median(Global_Sales), 3),
      IQR = round(IQR(Global_Sales), 3),
      .groups = "drop"
    ) |>
    rename(Group = 1) |>
    mutate(Group = as.character(Group), Variable = group_var, .before = 1)
}

group_summaries = map_dfr(group_vars, ~ group_summary(vgsales, .x))
print(group_summaries, n = Inf)
save_table(group_summaries, "group_medians.csv")

# -- Mann-Whitney / Wilcoxon --
# H1: Role-Playing games have a larger share of their sales in Japan than other genres.

rpg_JPshare = vgsales |> filter(Genre_Group == "Role-Playing") |> pull(JP_Share)
non_rpg_JPshare = vgsales |> filter(Genre_Group != "Role-Playing") |> pull(JP_Share)

wilcox_rpg = wilcox.test(rpg_JPshare, non_rpg_JPshare, alternative = "greater")
print(wilcox_rpg)

save_table(
  tibble(
    test = "JP_Share: Role-Playing > other genres",
    W = unname(wilcox_rpg$statistic),
    p_value = format.pval(wilcox_rpg$p.value, digits = 3),
    median_rpg = median(rpg_JPshare),
    median_other = median(non_rpg_JPshare)
  ),
  "wilcoxon_rpg_jp_share.csv"
)

# ==================================
# Correlation
# ==================================

# quantitative variables
num_vars = vgsales |>
  select(Global_Sales, NA_Sales, EU_Sales, JP_Sales,
         Other_Sales, NA_Share, EU_Share, JP_Share)

cor_matrix = cor(num_vars, method = "spearman")
pval_matrix = cor_pmat(num_vars, method = "spearman") |>
  column_to_rownames("rowname") |>
  as.matrix()

save_table(as_tibble(cor_matrix, rownames = "variable"), "spearman_correlation.csv")

png(fig_path("p11_spearman_corr_matrix.png"), width = 9, height = 6, units = "in", res = 150)
corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  order = "hclust",
  tl.col = "grey30",
  tl.srt = 45,
  addCoef.col = "white",
  number.cex  = 0.75,
  p.mat = pval_matrix,
  sig.level = 0.01,
  insig = "pch",
  col = colorRampPalette(c("#7B3294","white","#008837"))(200),
  title = "Spearman correlation matrix",
  mar = c(0, 0, 2, 0)
)
invisible(dev.off())

# ==================================
# Regression
# ==================================

vgsales = vgsales |>
  mutate(
    Publisher_Tier = relevel(Publisher_Tier, ref = "Small"),
    Genre_Group = relevel(Genre_Group, ref = "Action"),
    Platform_Gen = relevel(Platform_Gen, ref = "Gen3-4"),
    Decade = relevel(Decade, ref = "1980s")
  )

tidy_model = function(model) {
  tidy(model, conf.int = TRUE, conf.level = 0.95) |>
    mutate(across(where(is.numeric), ~ round(., 4)))
}

glance_model = function(model) {
  glance(model) |>
    select(r.squared, adj.r.squared, sigma, statistic, p.value, df) |>
    mutate(across(where(is.numeric), ~ round(., 4)))
}

# -- Model 1: log global sales --
model1 = lm(Log_Global_Sales ~ Publisher_Tier + Genre_Group + Platform_Gen + Decade,
            data = vgsales)
print(summary(model1))

# -- Model 2: share of sales in Japan --
model2 = lm(JP_Share ~ Genre_Group + Platform_Gen,
            data = vgsales)
print(summary(model2))

save_table(tidy_model(model1), "regression_model1_coefficients.csv")
save_table(tidy_model(model2), "regression_model2_coefficients.csv")
save_table(
  bind_rows(glance_model(model1) |> mutate(model = "model1", .before = 1),
            glance_model(model2) |> mutate(model = "model2", .before = 1)),
  "regression_fit.csv"
)

# ==================================
# Regression plots
# ==================================

# -- Diagnostics --
png(fig_path("diagnostic_model1.png"), width = 9, height = 7, units = "in", res = 150)
par(mfrow = c(2, 2))
plot(model1)
mtext("Model 1 diagnostics", outer = TRUE, line = -1.5, font = 2)
invisible(dev.off())

png(fig_path("diagnostic_model2.png"), width = 9, height = 7, units = "in", res = 150)
par(mfrow = c(2, 2))
plot(model2)
mtext("Model 2 diagnostics", outer = TRUE, line = -1.5, font = 2)
invisible(dev.off())

# -- Model equations (intercept + 3 largest significant effects) --

model_equation = function(model, outcome_label) {
  coefs = coef(model)
  r2 = round(summary(model)$r.squared, 3)
  sig_coefs = tidy(model) |>
    filter(term != "(Intercept)", p.value < 0.05) |>
    slice_max(abs(estimate), n = 3) |>
    mutate(
      term = str_remove(term,
                        "Publisher_Tier|Genre_Group|Platform_Gen|Decade"),
      part = paste0(ifelse(estimate >= 0, "+ ", "- "),
                    round(abs(estimate), 3),
                    " (", term, ")")
    ) |>
    pull(part)

  paste0(outcome_label, " = ",
         round(coefs["(Intercept)"], 3), " ",
         paste(sig_coefs, collapse = " "),
         "  |  R² = ", r2)
}

# -- Model 1 --
eq1 = model_equation(model1, "log(Global Sales)")

reg1 = vgsales |>
  mutate(fitted = fitted(model1)) |>
  ggplot(aes(x = fitted, y = Log_Global_Sales)) +
  geom_point(alpha = 0.12, size = 0.7, color = "#3266AD") +
  geom_abline(slope = 1, intercept = 0,
              color = "#993C1D", linewidth = 0.9) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              color = "#0F6E56", linewidth = 0.8,
              linetype = "dashed") +
  coord_cartesian(
    xlim = c(0, quantile(fitted(model1), 0.99)),
    ylim = c(0, quantile(vgsales$Log_Global_Sales, 0.99))
  ) +
  labs(
    title = "Model 1",
    subtitle = eq1,
    x = "Fitted values",
    y = "Observed log(Global Sales)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.subtitle    = element_text(size = 8, color = "grey40")
  )

ggsave(fig_path("reg_model1_fitted.png"), plot = reg1, width = 9, height = 6, dpi = 150)

# -- Model 2 --
eq2 = model_equation(model2, "JP Share")

reg2 = vgsales |>
  mutate(fitted = fitted(model2)) |>
  ggplot(aes(x = fitted, y = JP_Share)) +
  geom_point(alpha = 0.12, size = 0.7, color = "#7B4EA0") +
  geom_abline(slope = 1, intercept = 0,
              color = "#993C1D", linewidth = 0.9) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              color = "#0F6E56", linewidth = 0.8,
              linetype = "dashed") +
  coord_cartesian(
    xlim = c(0, quantile(fitted(model2),  0.99)),
    ylim = c(0, quantile(vgsales$JP_Share, 0.99))
  ) +
  scale_y_continuous(labels = label_percent()) +
  scale_x_continuous(labels = label_percent()) +
  labs(
    title = "Model 2",
    subtitle = eq2,
    x = "Fitted values",
    y = "Observed JP Share"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.subtitle    = element_text(size = 8, color = "grey40")
  )

ggsave(fig_path("reg_model2_fitted.png"), plot = reg2, width = 9, height = 6, dpi = 150)
