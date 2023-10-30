#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast
ast.literal_eval


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies.head(1)


# In[6]:


credits.head(1)


# In[7]:


movies.merge(credits,on='title').shape


# In[8]:


movies.shape


# In[9]:


credits.shape


# In[10]:


#genres
#id
#keywords
#title
#overview
#cast
#crew


# In[11]:


movies['original_language'].value_counts()


# In[12]:


movies.info()


# In[13]:


movies = movies[['title','overview','genres','keywords']]


# In[14]:


movies.head()


# In[15]:


movies.isnull().sum()


# In[16]:


movies.dropna(inplace=True)


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.iloc[0].genres


# In[19]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
       L.append(i['name']) 
    return L
    


# In[20]:


movies['genres']=movies['genres'].apply(convert)


# In[21]:


movies.head()


# In[22]:


movies['keywords']= movies['keywords'].apply(convert)


# In[23]:


movies.head()


# In[24]:


movies['overview'][0]


# In[25]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[26]:


movies.head()


# In[27]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])


# In[28]:


movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])


# In[29]:


movies.head()


# In[30]:


movies['tags']=movies['genres']+movies['keywords']


# In[31]:


movies.head()


# In[32]:


new_df=movies[['title','tags']]


# In[33]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[34]:


new_df.head()


# In[35]:


import nltk


# In[36]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[37]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)
    


# In[38]:


new_df['tags']=new_df['tags'].apply(stem)


# In[39]:


pip install nltk


# In[40]:


new_df['tags'][0]


# In[41]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[42]:


new_df.head()


# In[43]:


new_df['tags'][0]


# In[44]:


new_df['tags'][1]


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[46]:


cv.fit_transform(new_df['tags']).toarray()


# In[47]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[48]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[49]:


vectors


# In[50]:


vectors[0]


# In[51]:


cv.get_feature_names()


# In[52]:


stem('Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d')


# In[53]:


from sklearn.metrics.pairwise import cosine_similarity


# In[54]:


cosine_similarity(vectors)


# In[55]:


cosine_similarity(vectors).shape


# In[56]:


similarity=cosine_similarity(vectors)


# In[57]:


similarity


# In[58]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[59]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        print()


# In[60]:


recommend('Tangled')


# In[61]:


new_df.iloc[1216].title


# In[62]:


import pickle


# In[63]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[64]:


new_df['title'].values 


# In[65]:


new_df.to_dict()


# In[66]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[67]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




