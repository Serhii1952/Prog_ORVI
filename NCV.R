####################################
#Nested Cross-Validation со стратификацией#
####################################
# Установка необходимых библиотек (если не установлены)
install.packages(c("readxl", "caret", "pROC", "dplyr"))
library(readxl)
library(caret)
library(pROC)
library(dplyr)
#####Чтение данных из памяти***
df <- read.table("clipboard", header = TRUE, sep = "\t")
# Убедимся, что Y — это фактор (нужно для стратификации и классификации)
df$Y <- as.factor(df$Y)
KStol <- 6 # Пример: отбираем 6 лучших признаков

# 2. Настройка внешней кросс-валидации (Outer Loop)



answ<-floor(runif(1, min=1, max=1000))
set.seed(answ) ## Выбор случайного числа для расчетов
n_folds_outer <- 5
outer_folds <- createFolds(df$Y, k = n_folds_outer, list = TRUE)

outer_results <- list()

# --- ВНЕШНИЙ ЦИКЛ (Оценка модели) ---
for(i in 1:n_folds_outer) {
  
  # Разбиение на Test и Train для внешнего кольца
  test_idx <- outer_folds[[i]]
  train_data <- df[-test_idx, ]
  test_data <- df[test_idx, ]
  # --- ВНУТРЕННИЙ ЦИКЛ (Отбор признаков / Tuning) ---
  # В данном примере мы просто отберем KStol признаков по корреляции/значимости
  # на обучающей выборке текущего фолда
  
  # Простой фильтр признаков на основе p-value логистической регрессии
  p_values <- sapply(names(train_data)[names(train_data) != "Y"], function(x) {
    model_simple <- glm(as.formula(paste("Y ~", x)), data = train_data, family = binomial)
    summary(model_simple)$coefficients[2, 4] # Берем Pr(>|z|)
  })

  selected_features <- names(sort(p_values))[1:KStol]
  formula_final <- as.formula(paste("Y ~", paste(selected_features, collapse = " + ")))
  
  # 3. Обучение финальной модели на выбранных признаках
  final_model <- glm(formula_final, data = train_data, family = binomial)
  
  # 4. Предсказание на отложенном (внешнем) фолде
  probs <- predict(final_model, newdata = test_data, type = "response")
  # 5. Расчет ROC и AUC
  roc_obj <- roc(test_data$Y, probs, quiet = TRUE)
  auc_val <- auc(roc_obj)
  
  outer_results[[i]] <- list(auc = auc_val, roc = roc_obj, features = selected_features)
  
  cat(sprintf("Фолд %d: AUC = %.3f, Признаки: %s\n", i, auc_val, paste(selected_features, collapse=", ")))
}

# Итоговая средняя метрика по Nested CV
mean_auc <- mean(sapply(outer_results, function(x) x$auc))
cat(sprintf("\nСредний AUC по Nested CV: %.4f\n", mean_auc))

# Визуализация последней ROC-кривой
plot(outer_results[[1]]$roc, main = "ROC Curve (Fold 1)")

# Вывод признаков для каждого фолда
for(i in 1:length(outer_results)) {
  cat(sprintf("Фолд %d: %s\n", i, paste(outer_results[[i]]$features, collapse=", ")))
}

# Посчитаем частоту появления признаков
all_selected <- unlist(lapply(outer_results, function(x) x$features))
table(all_selected) %>% sort(decreasing = TRUE)

# Конец цикла


# Подготовка пустого графика
plot(NULL, xlim=c(1,0), ylim=c(0,1), 
     xlab="Специфичность (1 - FPR)", ylab="Чувствительность (TPR)", 
     main="Nested CV: ROC-кривые для всех фолдов")
abline(a=1, b=-1, lty=2, col="gray") # Диагональ случайного гадания

# Цвета для разных фолдов
colors <- rainbow(n_folds_outer)
auc_values <- c()

# Добавляем цикл отрисовки внутрь вашего процесса (или после него)
for(i in 1:n_folds_outer) {
  current_roc <- outer_results[[i]]$roc
  auc_values[i] <- outer_results[[i]]$auc
  
  # Отрисовка кривой конкретного фолда
  plot(current_roc, add=TRUE, col=colors[i], lwd=2)
}

# Добавление легенды с AUC для каждого фолда
legend("bottomright", 
       legend = paste0("Фолд ", 1:5, " (AUC = ", round(auc_values, 3), ")"), 
       col = colors, lwd = 2, cex = 0.8)

# Добавление среднего значения AUC на график
text(0.2, 0.1, labels = paste("Средний AUC =", round(mean(auc_values), 3)), 
     font = 2, col = "black")
# дополнит
af<-table(all_selected) %>% sort(decreasing = TRUE)
mean_auc
af
