
#install.packages('arules')
library(arules)

dataset = read.transactions('groceries.csv', sep = ',', rm.duplicates = TRUE)

summary(dataset)

itemFrequencyPlot(dataset, topN = 20)

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.002, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])

rules = apriori(data = dataset, parameter = list(support = 0.003, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])

# Training Eclat on the dataset
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])
