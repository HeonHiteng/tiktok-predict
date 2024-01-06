from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

### git clone https://github.com/VBradCulbertson/Forecasting-TikTok-Trend-Engagement-Using-ML
# Read data into dataframes.
suggvids = pd.read_csv('Forecasting-TikTok-Trend-Engagement-Using-ML/sug_users_vids_all.csv')
suggvids1 =  pd.read_csv('Forecasting-TikTok-Trend-Engagement-Using-ML/sug_users_vids1.csv')
suggvids2 = pd.read_csv('Forecasting-TikTok-Trend-Engagement-Using-ML/sug_users_vids2.csv')
suggvids3 = pd.read_csv('Forecasting-TikTok-Trend-Engagement-Using-ML/sug_users_vids3.csv')
suggvids4 = pd.read_csv('Forecasting-TikTok-Trend-Engagement-Using-ML/sug_users_vids4.csv')
suggvids5 = pd.read_csv('Forecasting-TikTok-Trend-Engagement-Using-ML/top_users_vids.csv')
data = pd.concat([suggvids,suggvids1,suggvids2,suggvids3,suggvids4,suggvids5])

# Drop dupicate records.
data.drop_duplicates(keep = 'first', inplace = True)

# Prettify column names.
data.rename(columns={'id': 'ID',
                    'create_time': 'Create Time',
                    'user_name': 'User',
                    'hashtags': 'Hashtags',
                    'song': 'Song Title',
                    'video_length': 'Length',
                    'n_likes': 'Likes',
                    'n_shares': 'Shares',
                    'n_comments': 'Comments',
                    'n_plays': 'Views',
                    'n_followers': 'Followers',
                    'n_total_likes': 'Total Likes',
                    'n_total_vids': 'Total Videos'}, inplace = True)

# Convert unix datetime to standard datetime format.
data['Create Time'] = pd.to_datetime(data['Create Time'], unit = 's')

# Calculate total engagement and engagement rate.
data['Engagement'] = data['Likes'] + data['Shares'] + data['Comments'] + data['Views']
# Previous exploration resulted in infinite values of Engagment Rate. I've elected to resolve 
# this by setting Engagement Rate to 0 for any User with 0 followers.
# The industry standard in social media is used below for the calculation of Engagement Rate's numerator.
data['Engagement Rate'] = np.where(data['Followers'] == 0, 0, ((data['Likes'] + data['Shares'] + data['Comments']) / data['Followers']))

n_uses = data['Song Title'].value_counts()

# Filter for songs used more than 25 times.
hot_songs = data[data['Song Title'].isin(n_uses[n_uses >= 20].index)]
# Add column for number of uses for each song in filtered df.
hot_songs['Song Uses'] = hot_songs['Song Title'].map(hot_songs['Song Title'].value_counts())

# Create df for window calculation, grouped by song/sound.
windows = hot_songs.groupby('Song Title')['Create Time'].agg(Min = 'min',
                              Max = 'max').reset_index()
# Calculate the window of activity for each song/sound.
delta = windows['Max'] - windows['Min']
windows['Window'] = delta.dt.days
# Add the fields to the original df by merging.
hot_songs = pd.merge(hot_songs, windows)
# Calculate position in window.
hot_songs['Window Pos'] = ((hot_songs['Create Time'] - hot_songs['Min']) / (hot_songs['Max'] - hot_songs['Min'])) * 100
# Calculate number of days since the Song/Sound originally appeared in a video.
hot_songs['Days Since Debut'] = hot_songs.groupby('Song Title')['Create Time'].transform(lambda x: (x-x.min()).dt.days)
# Filtering out some of the outliers with greatest influence over distribution.
hot_songs = hot_songs[hot_songs['Engagement'] <= 5000000]

# Observed instances of "Original Sound" used as Song Title (varied languages).
orig_sounds = ['original sound', 'sonido original', 'som original', 'Originalton',
               'Original Sound', 'orijinal ses', 'son original', 'оригинальный звук']

# Subset the df to remove "original sound" song titles.
no_orig = hot_songs[~hot_songs['Song Title'].isin(orig_sounds)]
no_orig['Song Uses'] = no_orig['Song Title'].apply(lambda x: (no_orig['Song Title'] == x).sum())

X_neither = no_orig.drop(columns = ['User', 'ID', 'Song Uses', 'Song Title', 'Hashtags',
                                    'Create Time', 'Min', 'Views', 'Max', 'Likes',
                                    'Comments', 'Shares', 'Window', 'Window Pos', 'Engagement Rate',
                                    'Engagement'])
y_neither = no_orig['Engagement']

# Reserve test data.
X_remainder, X_test, y_remainder, y_test = train_test_split(X_neither, y_neither, test_size = .25, random_state = 42)

# Set aside validation data.
X_train, X_validation_raw, y_train, y_validation = train_test_split(X_neither, y_neither, test_size = .25, random_state = 42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation_raw)

my_random_forest = RandomForestRegressor()
my_random_forest.fit(X_train, y_train)

decision_tree_scores = []
for sub_tree in my_random_forest.estimators_:
    decision_tree_scores.append(sub_tree.score(X_train, y_train))
    
print("Performance on Train data:")
print(f"Average Decision Tree: {np.mean(decision_tree_scores)}")
print(f"Random Forest: {my_random_forest.score(X_train, y_train)}")

decision_tree_scores = []
for sub_tree in my_random_forest.estimators_:
    decision_tree_scores.append(sub_tree.score(X_validation, y_validation))

print("Performance on Validation data:")
print(f"Average Decision Tree: {np.mean(decision_tree_scores)}")
print(f"Random Forest: {my_random_forest.score(X_validation, y_validation)}")

with open('model_random_forest.pkl','wb') as f:
    pickle.dump(my_random_forest,f)

with open('model_scaler.pkl','wb') as f:
    pickle.dump(scaler,f)