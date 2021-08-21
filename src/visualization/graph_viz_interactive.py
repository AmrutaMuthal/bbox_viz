import streamlit as st
import pandas as pd
import numpy as np
import os, cv2
import sys
import scipy.io
import random
from pathlib import Path  
import json
import PIL
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import SessionState
import collections
import networkx as nx
from plotly import express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    # Render the readme as markdown using st.markdown.
    session_state = SessionState._get_state()
    if session_state.page_number is None:
        session_state.page_number = 0
    if session_state.object_type is None:
        session_state.object_type = 0
    if session_state.label_col is None:
        session_state.label_col = [0.0,0.0,0.0]
    if session_state.part_disp is None:
        session_state.part_disp = False
    if session_state.bbox_disp is None:
        session_state.bbox_disp = False
    if session_state.graph is None:
        session_state.graph = False
    if session_state.set is None:
        session_state.set = 0
    
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    f = open(os.path.join(path,'instructions.md'), "r")
    image_list = pd.read_csv(os.path.join(path,'src','visualization','dataset.csv'))
    readme_text = st.markdown(f.read())
    
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        f = open(path+'src/visualization/'+"graph_viz.py", "r")
        st.code(f.read())
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app(session_state,image_list)
        
def rgb_to_hex(rgb):
            return '%02x%02x%02x' % rgb
                
def run_the_app(session_state,image_list):
    
    data_mode = st.sidebar.selectbox("Select data view", ["Raw data","Bounding box data"],0)
    labels = os.listdir(anno_source)
    session_state.object_type = st.sidebar.selectbox("Search for which objects?", labels, 0)
    session_state.set = st.sidebar.selectbox("Which dataset?",["train","test","val"], 0)
    image_list = image_list[(image_list['label']==session_state.object_type) &
                            (image_list['set']==session_state.set)]['image'].to_list()

    session_state.gt_overlay = st.sidebar.checkbox("Image overlay")
    
    n_labels = len(labels)
    diff = int(360/(n_labels+2))
    colours = np.arange(0,360,diff)
    colours = np.uint8([[[col,255,255] for col in colours[1:-1]]])
    colours = cv2.cvtColor(colours,cv2.COLOR_HSV2RGB)[0]
    label_col = colours[np.where(np.array(labels)==session_state.object_type)][0]
           
    session_state.bbox_disp = False
    session_state.graph = False
    font                   = cv2.FONT_HERSHEY_PLAIN
    fontScale              = 0.8
    fontColor              = (255,255,255)
    lineType               = 1
    
    colors = [(1, 0, 0),
          (0.737, 0.561, 0.561),
          (0.255, 0.412, 0.882),
          (0.545, 0.271, 0.0745),
          (0.98, 0.502, 0.447),
          (0.98, 0.643, 0.376),
          (0.18, 0.545, 0.341),
          (0.502, 0, 0.502),
          (0.627, 0.322, 0.176),
          (0.753, 0.753, 0.753),
          (0.529, 0.808, 0.922),
          (0.416, 0.353, 0.804),
          (0.439, 0.502, 0.565),
          (0.784, 0.302, 0.565),
          (0.867, 0.627, 0.867),
          (0, 1, 0.498),
          (0.275, 0.51, 0.706),
          (0.824, 0.706, 0.549),
          (0, 0.502, 0.502),
          (0.847, 0.749, 0.847),
          (1, 0.388, 0.278),
          (0.251, 0.878, 0.816),
          (0.933, 0.51, 0.933),
          (0.961, 0.871, 0.702),
          (0,     0,     0)]
    colors = (np.asarray(colors)*255)
       
    prev_button, _ ,next_button = st.beta_columns([1, 8, 1])
    last_page = len(image_list)-1
    session_state.page_number = min(session_state.page_number,last_page)
    if next_button.button("Next"):

        if session_state.page_number + 1 > last_page:
            session_state.page_number = 0
        else:
            session_state.page_number += 1

    if prev_button.button("Previous"):

        if session_state.page_number - 1 < 0:
            session_state.page_number = last_page
        else:
            session_state.page_number -= 1
    
    selected_idx = st.slider("Select image index",0,len(image_list),session_state.page_number)
    session_state.page_number = selected_idx
    
    if data_mode == "Bounding box data":
        show_bbox_data(session_state,image_list[selected_idx],label_col,colors)
    else:
        show_raw_data(session_state,image_list[selected_idx],label_col,colors)
        
    session_state.sync()

def show_raw_data(session_state,selected_image,label_col,colors):
    
    session_state.part_disp = st.sidebar.checkbox("Display parts",value = session_state.part_disp)
   
    f = open(os.path.join(path,'src','visualization','part_label.json'),)
    with open(os.path.join(path,'src','visualization','tree.json'),) as fp:
        tree = json.load(fp)
    part_labels = json.load(f)
    part_labels = part_labels[session_state.object_type]
    part_labels[""] = 24
    
    
    with open(os.path.join(anno_source,session_state.object_type,'bbox',selected_image+'.json')) as fp:
        annotation_dict = json.load(fp)
    image_path = annotation_dict['image']
    
    im_x_min,im_y_min,im_x_max,im_y_max =annotation_dict['bbox']
    raw_image = cv2.imread(os.path.join(img_source,image_path+'.jpg'))
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    
    fig, axes = plt.subplots(1,3, figsize = (5,5))
    part_image = raw_image.copy()

    part_dict = annotation_dict['parts']
    axes[0].imshow(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1], interpolation='none')
    axes[0].axis('off')
    
    if session_state.gt_overlay:
        alpha =0.2
        axes[1].imshow(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1], interpolation='none')
        axes[2].imshow(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1], interpolation='none')
        
    else:
        alpha =1.0
        axes[1].imshow(np.zeros((im_x_max-im_x_min,im_y_max-im_y_min,3)))
        axes[2].imshow(np.zeros((im_x_max-im_x_min,im_y_max-im_y_min,3)))
    
    axes[1].axis('off')
    axes[2].axis('off')
    
    if part_dict and session_state.part_disp:
        
        overlay = np.uint(np.zeros(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1].shape,'int'))
        for part in part_dict.keys():
            mask = np.repeat(np.expand_dims(np.matrix(part_dict[part]['mask']),axis=-1),3,axis=2)
            mask = np.uint(mask*np.expand_dims(np.matrix(colors[part_labels[part]-1]).astype(int),axis=0))
            overlay+= mask[im_x_min:im_x_max+1,im_y_min:im_y_max+1]


        axes[1].imshow(overlay,interpolation='none',alpha=alpha)
        axes[1].axis('off')      
        
        bb_image = raw_image.copy()
     
        for part in part_dict.keys():

            x_min,y_min,x_max,y_max =part_dict[part]['bbox']
            rect = patches.Rectangle((y_min-im_y_min,x_min-im_x_min),y_max-y_min,x_max-x_min,
                                     linewidth=1,edgecolor=colors[part_labels[part]-1]/255,facecolor='none')

            axes[2].add_patch(rect)
        
        l = min(6,len(part_labels))
        b = int(np.ceil(len(part_labels)/6))
        label_matrix = np.zeros((b,l)).astype('str')
        labels = list(part_labels.keys())
        for i in range(b):
            for j in range(l):
                try:
                    label_matrix[i][j] = labels[i*6+j]
                    
                except:
                    label_matrix[i][j] = ""

        
        def rgb_to_hex(rgb):
            return '%02x%02x%02x' % rgb
        
        label_df = pd.DataFrame(label_matrix)
        label_df = label_df.style.applymap(lambda x: ('background-color : #'
                                                     +rgb_to_hex(tuple(list(colors[part_labels[x]-1].astype(int))))))
         
    
    st.pyplot(fig)
    if session_state.part_disp:
        st.dataframe(label_df)
        st.subheader("Part Legend")  
    session_state.sync()

def show_bbox_data(session_state,selected_image,label_col,colors):
    
    session_state.bbox_disp = st.sidebar.checkbox("Display bounding box view", value = session_state.bbox_disp)
    session_state.graph = st.sidebar.checkbox("Display adjacency graph",value = session_state.graph)
   
    f = open(os.path.join(path,'src','visualization','part_label.json'),)
    with open(os.path.join(path,'src','visualization','tree.json'),) as fp:
        tree = json.load(fp)
    part_labels = json.load(f)
    part_labels = part_labels[session_state.object_type]
    labels = list(part_labels.keys())
    part_labels[""] = 24
    
    
    with open(os.path.join(anno_source,session_state.object_type,'bbox',selected_image+'.json')) as fp:
        annotation_dict = json.load(fp)
    image_path = annotation_dict['image']
    
    im_x_min,im_y_min,im_x_max,im_y_max =annotation_dict['bbox']
    raw_image = cv2.imread(os.path.join(img_source,image_path+'.jpg'))
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    graph = make_subplots(rows=1, cols=2)
    part_image = raw_image.copy()
    
    
    graph.add_trace(px.imshow(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1]).data[0],row=1,col=1)
    
                    
    if session_state.gt_overlay:
        alpha =0.2
        graph.add_trace(px.imshow(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1]).data[0],row=1,col=2)
    
    else:
        graph.add_trace(px.imshow(np.zeros((im_x_max-im_x_min,im_y_max-im_y_min,3))).data[0],row=1,col=2)
        
        
    
    part_dict = annotation_dict['parts']
    
    if part_dict and session_state.bbox_disp:
        bb_image = raw_image.copy()
        part_centers = np.zeros((24,3))

        for part in part_dict.keys():

            x_min,y_min,x_max,y_max =part_dict[part]['bbox']
            part_centers[part_labels[part]-1]= [1,(x_min+x_max)/2,(y_min+y_max)/2]
            
            part_col = colors[part_labels[part]-1]
            clr_str = 'rgb('+str(int(part_col[0]))+','+str(int(part_col[1]))+','+str(int(part_col[1]))+')'
            graph.add_shape(type="rect",
                xref="x", yref="y",
                y0=x_min-im_x_min, x0=y_min-im_y_min, y1=x_max-im_x_min, x1=y_max-im_y_min,
                line=dict(color=clr_str,width=1),
                row=1,col=2
            )
            
        labels = list(part_labels.keys())
        
        parts_present = [labels[i] for i in np.where(part_centers[:,0]==1)[0]]
        part_centers = part_centers[np.where(part_centers[:,0]==1)]
        node_x = [x-im_x_min for _,x,y in part_centers]
        node_y = [y-im_y_min for _,x,y in part_centers]

        graph.add_trace(go.Scatter(
            x=node_y, y=node_x,
            mode='markers',
            hoverinfo='text',
            text = parts_present,
            marker=dict(
                showscale=False,
                reversescale=True,
                # colorscale='YlreGnBu',
                color=['red']*len(node_x),
                size=10,
                opacity = 0.0,
                colorbar=dict(
                    thickness=15,
                   
                ),
                line_width=2)),
            row=1, col=2)
        
        l = min(6,len(part_labels))
        b = int(np.ceil(len(part_labels)/6))
        label_matrix = np.zeros((b,l)).astype('str')
        labels = list(part_labels.keys())
        for i in range(b):
            for j in range(l):
                try:
                    label_matrix[i][j] = labels[i*6+j]
                    
                except:
                    label_matrix[i][j] = ""
        
        label_df = pd.DataFrame(label_matrix)
        label_df = label_df.style.applymap(lambda x: ('background-color : #'
                                                     +rgb_to_hex(tuple(list(colors[part_labels[x]-1].astype(int))))))
        
            
    if part_dict and session_state.graph:
        
        part_centers = np.zeros((24,3))
        for part in part_dict.keys():
            x_min,y_min,x_max,y_max =part_dict[part]['bbox']
            part_centers[part_labels[part]-1]= [1,(x_min+x_max)/2,(y_min+y_max)/2]
            
        adj_mat = np.matrix(annotation_dict['adj'])
        G = nx.from_numpy_matrix(adj_mat)
        parts_present = [labels[i] for i in np.where(part_centers[:,0]==1)[0]]
        part_centers = part_centers[np.where(part_centers[:,0]==1)]

        edge_x = []
        edge_y = []
        for i in range(len(adj_mat)):
            for j in range(i,len(adj_mat)):
                
                if adj_mat[i,j]==1:
                    _, x0, y0 = part_centers[i]
                    
                    _, x1, y1 = part_centers[j]
                    edge_x.append(x0-im_x_min)
                    edge_x.append(x1-im_x_min)
                    edge_x.append(None)
                    edge_y.append(y0-im_y_min)
                    edge_y.append(y1-im_y_min)
                    edge_y.append(None)

        graph.add_trace(go.Scatter(
            x=edge_y, y=edge_x,
            line=dict(width=1, color='blue'),
            hoverinfo='none',
            mode='lines'),
             row=1, col=2)
        
        node_x = [x-im_x_min for _,x,y in part_centers]
        node_y = [y-im_y_min for _,x,y in part_centers]
        
        graph.add_trace(go.Scatter(
            x=node_y, y=node_x,
            mode='markers',
            hoverinfo='text',
            text = parts_present,
            marker=dict(
                showscale=False,
                reversescale=True,
                color=['red']*len(node_x),
                size=10,
                colorbar=dict(
                    thickness=15,
                ),
                line_width=2)),
            row=1, col=2)
        
        
        
        l = min(6,len(part_labels))
        b = int(np.ceil(len(part_labels)/6))
        label_matrix = np.zeros((b,l)).astype('str')
        
        for i in range(b):
            for j in range(l):
                try:
                    label_matrix[i][j] = labels[i*6+j]
                    
                except:
                    label_matrix[i][j] = ""

        
        label_df = pd.DataFrame(label_matrix)
        label_df = label_df.style.applymap(lambda x: ('background-color : #'
                                                     +rgb_to_hex(tuple(list(colors[part_labels[x]-1].astype(int))))))
    
    
    graph.update_layout(
        showlegend = False,
        width = 500,
        height =  500,
        plot_bgcolor='rgb(0,0,0)'
        )
    graph.update_xaxes(visible=False)
    graph.update_yaxes(
        autorange="reversed",
        visible=False,
        scaleanchor = "x",
        scaleratio = 1,
     )
    st.plotly_chart(graph)
    if session_state.bbox_disp:
        st.dataframe(label_df)
        st.subheader("Part Legend")  
    session_state.sync()    

    

if __name__ == "__main__":
    
    curr_path = sys.argv[1]
    path = Path(curr_path)
    anno_source = os.path.join(path,'PASCAL-VOC','xybb-objects-new')
    img_source = os.path.join(path,'PASCAL-VOC','scene')
    path = Path(os.getcwd())
    main()