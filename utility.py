import pandas as pd
import sklearn.feature_selection
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.5) # set seaborn default for plots

def get_categories_info(dataframe):
    for col_name in dataframe.columns:
        if dataframe[col_name].dtypes == 'object':
            unique_cat = len(dataframe[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
            
def dummy_df(dataframe, todummy_list):
    for category in todummy_list:
        dummies = pd.get_dummies(dataframe[category], prefix=category, dummy_na=False)
        
        dataframe = dataframe.drop(category, 1)
        
        dataframe = pd.concat([dataframe, dummies], axis=1)
        
    return dataframe

def get_features_kbest(X_train, y_train, num_k):
    select = sklearn.feature_selection.SelectKBest(k=num_k)
    selected_features = select.fit(X_train, y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X_train.columns[i] for i in indices_selected]

    return colnames_selected

# bar chart using seaborn
def bar_chart(dataframe, feature):
    survived = dataframe[dataframe['Survived']==1][feature].value_counts()
    dead = dataframe[dataframe['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.title("Survived and dead passengers by '{colname}' feature".format(colname=feature))
    
    
def __extract_titles(names):
    titles = list()
    for name in names:
        titles.append(__get_title(name))
    
    return titles

def __get_title(name): return name.split(',')[1].split('.')[0].strip()

def feature_engineer_title(dataframe):
    titles = __extract_titles(dataframe['Name'])
    dataframe['Title'] = pd.Series(titles)
    dataframe = dataframe.drop('Name', axis=1)
    
    # We're going to keep only 'Mr', 'Miss' and 'Mrs', all others will fall into a new category named 'Other'
    dataframe['Title'] = ['Other' if title not in ['Mr', 'Mrs', 'Miss'] else title for title in dataframe['Title']]
    
    # Title mapping
    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Other': 3}
    dataframe['Title'] = dataframe['Title'].map(title_mapping)
    
    return dataframe

def __age_median_by_title(dataframe):
    medians = {}
    for title in dataframe['Title'].unique():
        medians[title] = pd.Series.median(dataframe[dataframe['Title']==title]['Age'])
    
    return medians

def feature_engineer_age(dataframe):
    medians_by_title = __age_median_by_title(dataframe)
    
    for index_row, row in dataframe.iterrows():
        if pd.isna(row['Age']):
            dataframe.loc[index_row, 'Age'] = medians_by_title[row['Title']]
    
    return dataframe
    
def lattice_plot(dataframe, feature, interval):
    facet = sns.FacetGrid(dataframe, hue='Survived', aspect=4)
    facet.map(sns.kdeplot, feature, shade=True)
    facet.set(xlim=(interval[0], interval[1]))
    facet.add_legend()
    plt.title("Survived and dead passengers by '{colname}' feature".format(colname=feature))
    
    plt.show()

def binning_age(dataframe):

    dataframe.loc[dataframe['Age'] <= 16, 'Age'] = 0 # child
    dataframe.loc[(dataframe['Age'] > 16) & (dataframe['Age'] <= 26), 'Age'] = 1 # young
    dataframe.loc[(dataframe['Age'] > 26) & (dataframe['Age'] <= 36), 'Age'] = 2 # adult
    dataframe.loc[(dataframe['Age'] > 36) & (dataframe['Age'] <= 62), 'Age'] = 3 # mid-age
    dataframe.loc[dataframe['Age'] > 62, 'Age'] = 4 # senior
    
    return dataframe

def __fare_median_by_title(dataframe):
    medians = {}
    for title in dataframe['Title'].unique():
        medians[title] = pd.Series.median(dataframe[dataframe['Title']==title]['Fare'])
    
    return medians


def feature_engineer_fare(dataframe):
    medians_by_title = __fare_median_by_title(dataframe)
    
    for index_row, row in dataframe.iterrows():
        if pd.isna(row['Fare']):
            dataframe.loc[index_row, 'Fare'] = medians_by_title[row['Title']]
    
    return dataframe

def binning_fare(dataframe):

    dataframe.loc[dataframe['Fare'] <= 17, 'Fare'] = 0
    dataframe.loc[(dataframe['Fare'] > 17) & (dataframe['Fare'] <= 30), 'Fare'] = 1 
    dataframe.loc[(dataframe['Fare'] > 30) & (dataframe['Fare'] <= 100), 'Fare'] = 2 
    dataframe.loc[dataframe['Fare'] > 100, 'Fare'] = 3 
    
    return dataframe

def __cabin_median_by_title(dataframe):
    medians = {}
    for title in dataframe['Title'].unique():
        medians[title] = pd.Series.median(dataframe[dataframe['Title']==title]['Cabin'])
    
    return medians

def feature_engineer_cabin(dataframe):
    medians_by_title = __fare_median_by_title(dataframe)
    
    for index_row, row in dataframe.iterrows():
        if pd.isna(row['Cabin']):
            dataframe.loc[index_row, 'Cabin'] = medians_by_title[row['Title']]
    
    return dataframe
