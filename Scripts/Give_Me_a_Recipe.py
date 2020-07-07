#!/usr/bin/env python
# coding: utf-8

# # Give Me A Recipe

# In[1]:



from Bounding_Box_Detection import *
import pandas as pd
import numpy as np
import streamlit as st
import glob

st.title("Give Me A Recipe!")
st.header('        ')



# In[2]:


recipe_df = pd.read_csv('all_recipes_actual_ingredient_vector.csv')
import ast
recipe_df['Ingredients'] = recipe_df['Ingredients'].apply(lambda x: ast.literal_eval(x))
recipe_df['Remove_Indices_1'] = recipe_df['Remove_Indices_1'].apply(lambda x: ast.literal_eval(x))


# In[3]:


def recipe_database_query(available_ing, df, criteria):
    Ingredient_columns = ['avocado', 'banana', 'basil', 'beans', 'beef', 'bell pepper', 'broccoli', 'butter', 'carrot', 
                          'celery', 'chicken breast', 'cilantro', 'cucumber', 'egg', 'eggplant', 'flour', 'garlic', 
                          'garlic powder', 'ginger', 'green beans', 'ground beef', 'honey', 'lemon', 'lettuce', 'lime',
                          'mayonnaise', 'milk', 'mozzarella', 'mushroom', 'oil', 'olive oil', 'onion', 'oregano', 
                          'paprika', 'pasta', 'pepper', 'potato', 'red onion', 'salmon', 'salt', 'scallion', 'shrimp',
                          'soy sauce', 'sugar', 'tomato', 'tomato paste', 'water', 'yogurt', 'zucchini']
    remaining_ingredients = []
    remaining_ingredients_count = []
    used_ingredients_ratio = []
    used_ingredients_all = []
	
    for index,row in df.iterrows():
        count_used_ing = 0
        a_indices = []
        used_ingredients = []
        for i in available_ing:
            if row[i] != -1:
                count_used_ing += 1
                used_ingredients.append(i)
                a_indices.append(row[i])
        used_ingredients_ratio.append(count_used_ing / len(available_ing))
        remaining_ingredients_count.append(row['Number_of_Actual_Ingredients']- count_used_ing)
        remove_indices = row['Remove_Indices_1'] + a_indices
        remaining_ingredients.append([i for j, i in enumerate(row['Ingredients']) if j not in remove_indices]) 
        used_ingredients_all.append(used_ingredients)
     
      
    df['Needed_Ingredients'] = remaining_ingredients
    df['Number_of_needed_Ingredients'] = remaining_ingredients_count
    df['Ratio_of_Used_Ingredients'] = used_ingredients_ratio
    df['Used_Ingredients'] = used_ingredients_all
    
    if criteria == 'Minimum Waste':
        df1 = df.sort_values(by = ['Ratio_of_Used_Ingredients','Number_of_needed_Ingredients' ],ascending = [False,True])
    if criteria == 'Minimum Shopping':
        df1 = df[df['Ratio_of_Used_Ingredients'] > 0]
        df1 = df1.sort_values(by = ['Number_of_needed_Ingredients','Ratio_of_Used_Ingredients'],ascending = [True,False])
    

    
    return df1


# In[4]:


def showresults (df1,criteria ):

    st.header('**Suggested Recipes With '+ criteria+':**')

    for i in range(0,5):
        st.subheader (df1.iloc[i]['Recipe_Name'] )
        list_all = df1.iloc[i]['Ingredients']
        remove_indices = df1.iloc[i]['Remove_Indices_1']
        other_ingredients = [i for j, i in enumerate(list_all) if j in remove_indices]
        st.write ('**This recipe uses: **', ', '.join(df1.iloc[i]['Used_Ingredients']) )
        st.write ('**Other Ingredients: **', ', '.join(other_ingredients))
        st.write ('**You still Need: **', ', '.join(df1.iloc[i]['Needed_Ingredients']) )
        st.write (df1.iloc[i]['Directions'])
        recipe_name=df1.iloc[i]['Recipe_Name']
        path='RecipeImages/'+recipe_name+'/*.jpg'
        recipe_image_folder=glob.glob(path)	
        try:
            st.image(recipe_image_folder[0],width=320, caption=recipe_name)
        except:
            st.write('No Picture Is Available')

# In[5]:
st.sidebar.markdown('**Upload an Image of Your Ingredients**')
img_file_buffer = st.sidebar.file_uploader('', type=["png", "jpg", "jpeg"])

st.sidebar.header('        ')

st.sidebar.markdown('**What kind of recipe do you want?**')
add_selectbox = st.sidebar.selectbox(
    "",
    ("Minimum Waste", "Minimum Shopping")
)



if img_file_buffer is not None:
	imageLocation = st.empty()
	imageLocation.image(img_file_buffer, width = 360)
	file_name, ingredient = ingredient_detection(img_file_buffer)
	imageLocation.image(file_name, width=360)
	recipe_database = recipe_database_query( ingredient, recipe_df, add_selectbox)
	showresults(recipe_database, add_selectbox)

	    
	 

