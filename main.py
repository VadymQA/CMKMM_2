import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize

pd.set_option('display.width', 12000)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 900)

data = pd.read_csv("files/Energy_and_Water_Data.csv", index_col="Order")
data_test = pd.read_csv("files/Energy_and_Water_Data.csv", index_col="Order")
print(data.shape)



data = data.drop(['Parent Property Id', 'Parent Property Name'], axis=1)
data = data.replace('Not Available', np.nan)

# data_test = data_test.replace('Not Available', np.nan)
# data_test['ENERGY STAR Score'] = data_test["ENERGY STAR Score"].astype(float)
# #debaggin
# data_test = data_test.replace('Not Available', np.nan)


print(data.isnull().mean() * 100)
count = 0
for col in data.columns:
    x = data[col].isnull().mean()*100
    if x > 50:
        print(col + " was droped because contains "+ str(x) + " percent wrong data")
        data=data.drop([col], axis= 1)
        count= count+1
print(count)
count=0

valid_columns=[]
for col in data.columns:
    if ("kWh" in col) or ("CO2e" in col) or ("kBtu" in col) or ("Score" in col) or ("therms" in col) or ("ft²" in col) or ("gal" in col):
        valid_columns.append(col)
        data[col] = data[col].astype(float)


data_new = data
for col in data_new.columns:
    if (col == 'Borough') or (col == 'Largest Property Use Type'):
        print('1')
    else:
        data_new = data_new.drop(columns=[col], axis = 1)

# print(data_new)
one_hot_encoded_data = pd.get_dummies(data_new, columns=['Borough', 'Largest Property Use Type'])
# print(one_hot_encoded_data)

corr_one_hot = one_hot_encoded_data = one_hot_encoded_data.corr()
# print(corr_one_hot.sort_values)



#1.lower outer fence: Q1 - 3*IQ
#2.upper outer fence: Q3 + 3*IQ
# for col in data.columns: find fences and filter
print(data.shape)
count = 0

for_drop = []
for col in valid_columns:
    # if (data.dtypes[col] == 'float64') or (data.dtypes[col] == 'float32'):
        Q1=data[col].quantile(0.25)
        Q3=data[col].quantile(0.75)
        IQ=Q3-Q1
        print(str(Q1) + "-Q1, " + str(Q3) + "-Q3, " + str(IQ) + "-IQ, for "+ col )
        bot_fence=Q1-3*IQ
        top_fence=3*IQ+Q3
        for x in data.index:
            # print(data.loc[x, col])
            y = data.loc[x, col]
            if (y < bot_fence) or (y > top_fence):
                data = data.drop(x)
                for_drop.append(x)
                count = count + 1



print(data.shape)
print(str(count) + " count")
data_for_drop = pd.DataFrame(for_drop)
data_for_drop_filtered= data_for_drop.drop_duplicates()
print("drop")
print(data_for_drop_filtered)
data = data.drop(data_for_drop_filtered)
# print(len(data_for_drop))


# print(data["ENERGY STAR Score"].value_counts(" "))
# print(data["Borough"].value_counts( ))
# print(data["Primary Property Type - Self Selected"].value_counts( ))



# data['ENERGY STAR Score'] = data['ENERGY STAR Score'].astype(float)
#осздать лист типов buildings
data = data.dropna(subset=['ENERGY STAR Score'])
types = data['Primary Property Type - Self Selected'].value_counts()
print(types)
types = list(types[types.values > 50].index)
print(types)
# Plot each building
for buildings in types:
    test_data = data[data['Largest Property Use Type'] == buildings]
    sns.kdeplot(test_data['ENERGY STAR Score'], fill=True, common_norm=False, alpha=0.6, linewidth=0)
# label the plot
plt.xlabel('Energy Star Score')
plt.ylabel('Density')
plt.title('Density Plot of Energy Star Scores by Building Type')
plt.show()


types2 = list(data['Borough'].value_counts().index)
print("list")
print(types2)
# Plot each building
for borough in types2:
    test_data = data[data['Borough'] == borough]
    sns.kdeplot(test_data['ENERGY STAR Score'], fill=True, common_norm=False, alpha=0.6, linewidth=0)
# label the plot
plt.xlabel('Energy Star Score')
plt.ylabel('Density')
plt.title('Density Plot of Energy Star Scores by Borough Type')
plt.show()


correlations_data = data.corr()['ENERGY STAR Score'].sort_values()
print(correlations_data)

print("__________________________")
count=0
for col in valid_columns:
    for x in data.index:
        if data.loc[x, col] > 100000:
            data.loc[x, col] = np.log10(data.loc[x, col])
            count = count + 1

# print(count)

print("__________________________")

one_hot_encoded_data = pd.get_dummies(data, columns = ['Borough', 'Largest Property Use Type'])

# Create correlation matrix
corr_matrix = one_hot_encoded_data.corr().abs()
print(corr_matrix)

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
print(upper_tri)

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.6)]

data_final = one_hot_encoded_data.drop(to_drop, axis=1)
print(data_final)












# # Create a list of buildings with more than 100 measurements
#
# types = data.dropna(subset=['ENERGY STAR Score'])
# types = types['Primary Property Type - Self Selected'].value_counts()
# types = list(types[types.values > 100].index)
#
# # Plot of distribution of scores for building categories
# # figsize(12, 10)
#
# # Plot each building
# for b_type in types:
#     # Select the building type
#     subset = data[data['Primary Property Type - Self Selected'] == b_type]
#
#     # Density plot of Energy Star scores
#     sns.kdeplot(subset['ENERGY STAR Score'], label=b_type, alpha=0.8)
#
# # label the plot
# plt.xlabel('Energy Star Score', size=20)
# plt.ylabel('Density', size=20)
# plt.title('Density Plot of Energy Star Scores by Building Type', size=28)





# # Make default density plot
# test = data["ENERGY STAR Score"].value_counts(" ")
# test1 = data["Borough"].value_counts(" ")
# sns.kdeplot(test, )
# sns.kdeplot(test1)
# plt.show()

# sns.histplot(data['Borough'], kde=True)
# sns.histplot(data['Primary Property Type - Self Selected'], kde=True)
# plt.show()
