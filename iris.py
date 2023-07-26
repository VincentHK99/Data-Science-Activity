# import relevant packages
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

#loading in through seaborn (double checked that data is the same)
iris = sns.load_dataset('iris')

# check for missing values in the dataset.
#if data is missing may have to impute or 
# drop variables from dataset
iris.isna().sum()

# below is a code from a package I am currently working on. 
# the package utilizes networkx to give the user a quick and 
# easy way to visualise relationships between variables in a dataset.
# It should help the user easily identify which variables are suitable
# for use in a machine learning model

# species variable must be converted into dummy varibles in order to be read 
# into a machine learning model
iris_dummies = pd.get_dummies(iris)

corr_dict = dict(iris_dummies.corr())

variable_names  = iris_dummies.columns.to_list()
for name in variable_names:
  for key, value in corr_dict[name].items():
    if abs(value) > 0.5 and value < 1:
      print(name,'--',round(value,2),'--',key)

G=nx.Graph()
for variable in variable_names:
  G.add_node(variable)

for name in variable_names:
  for key, value in corr_dict[name].items():
    if abs(value) > 0.5 and value < 1:
      if abs(value) < 0.75:
        strength_of_corr = 'moderate'
      else:
        strength_of_corr = 'strong'
      G.add_edge(name,key,correlation=round(value,2),strength_of_corr=strength_of_corr)



edge_labels = {(u,v):(d['correlation'],d['strength_of_corr']) for u,v,d, in G.edges(data=True)}
node_pos = nx.planar_layout(G)
nx.draw(G,pos=node_pos,with_labels=True, node_color="red",node_size=3000,
        font_color="white",font_size=6, font_weight="bold",
        width=5,edge_color='lightgrey')

nx.draw_networkx_edge_labels(G,pos=node_pos,edge_labels=edge_labels,label_pos = 0.5)

plt.show()

# Correlations of interest (strong correlations)
#  - petal length <--> petal width
#  - specices setosa <--> petal width
#  - petal length <--> sepal length
#  - sepal length <--> petal width

# there are also other moderate correlations not listed above
# for further detials check output of nextworx function

# Create a function to plot strong correlations listed above
def scatter_iris(x_var,y_var):
  plt.title(f'{str(x_var)} vs {str(y_var)} (corr = {round(np.corrcoef(iris[x_var], iris[y_var])[0][1],2)})')
  plt.scatter(iris[x_var],iris[y_var])
  plt.xlabel(str(x_var))
  plt.ylabel(str(y_var))
  plt.plot()


# strong relation between the length and width of a petals
scatter_iris('petal_length','petal_width')

#strong relation between length of petal and sepal
scatter_iris('petal_length','sepal_length')

#strong relation between length of sepal and width of petal
scatter_iris('sepal_length','petal_width')



### Bar chart petal width and sepal length accross species
mean_petal_width = []
mean_sepal_length = []
mean_sepal_width = []
species_list = list(set(iris['species']))

for i in species_list:
  mean_petal_width.append(iris['petal_width'][iris['species'] == i].mean())
  mean_sepal_length.append(iris['sepal_length'][iris['species'] == i].mean())
  mean_sepal_width.append(iris['sepal_width'][iris['species'] == i].mean())


# See that Setosa species has lower mean petal width.
# this is also reflected in the strong negative correlation 
# between the setosa species dummy and petal width variables
plt.bar(species_list, mean_petal_width)
plt.title('Mean petal width by species')
plt.xlabel('Species')
plt.ylabel('Petal Width')
plt.show()



# setosa species also has slightly lower mean sepal lenth
# this is reflected in the moderate negative correlation 
# between the setosa dummy and sepal length variables
plt.bar(species_list, mean_sepal_length)
plt.title('Mean sepal length by species')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.show()


# setosa species has higher mean sepal width than other species
# this is reflected in the moderate positive correlation 
# between the setosa dummy and sepal width variables
plt.bar(species_list, mean_sepal_width)
plt.title('Mean sepal width by species')
plt.xlabel('Species')
plt.ylabel('Sepal Width')
plt.show()

########################################################################
######### NOTES ON THE DATASET/ POSSIBLE MODELLING ISSUES ##############
########################################################################

# setosa species has some strong correlations in the dataset
# the species tends to be characterised by a significantly smaller petal width,
# a slightly small sepal length and a larger sepal width

# virginica tends to have larger petal width and sepal length
# as characterised by correlations and mean values

# versicolour seems to be less correlated with variables in the dataset
# thus making the species harder to characterise
# this might make it harder for the machine learning model to predict values
# for this category of species

# for a model predicting species of a an IRIS
# TARGET VARIABLE: Species
# APPROPRIATE FEATURES: Sepal width, sepal length, petal width, petal length

# Note: when predicting categorical variables, each variable must be assigned a number.
# in this case since there would be 3 categories each category of species would be assigned
# a number from 0 - setosa , 1 - versicolour, 2 - virginica

