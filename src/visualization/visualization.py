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

def main():
    # Render the readme as markdown using st.markdown.
    session_state = SessionState._get_state()
    if session_state.page_number is None:
        session_state.page_number = 0
    
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    f = open(os.path.join(path,'instructions.md'), "r")
    readme_text = st.markdown(f.read())
    
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        f = open(source+'src/visualization/'+"visualization.py", "r")
        st.code(f.read())
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app(session_state)
        
def run_the_app(session_state):
    
    labels = os.listdir(anno_source)
    object_type = st.sidebar.selectbox("Search for which objects?", labels, 1)
    image_list = os.listdir(os.path.join(anno_source,object_type,'bbox'))
    image_list = [name.split('.')[0] for name in image_list]

    image_list = np.append(image_list,['Random'])
    selected_image = st.sidebar.selectbox("Select image", image_list, 1)
    
    n_labels = len(labels)
    diff = int(360/(n_labels+2))
    colours = np.arange(0,360,diff)
    colours = np.uint8([[[col,255,255] for col in colours[1:-1]]])
    colours = cv2.cvtColor(colours,cv2.COLOR_HSV2RGB)[0]
    label_col = colours[np.where(np.array(labels)==object_type)][0]
    
    part_disp = st.sidebar.checkbox("Display parts")
    bbox_disp = st.sidebar.checkbox("Display bounding box view")
    obbox_disp = st.sidebar.checkbox("Display oriented bounding box view")
        
    prev_button, _ ,next_button = st.beta_columns([1, 10, 1])
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
    
    selected_image = st.slider('Select image index',0,len(image_list),session_state.page_number)
    session_state.page_number = selected_image
        
    display_multi(object_type,image_list,selected_image,label_col,part_disp,bbox_disp,obbox_disp)
    
    session_state.sync()

def display_multi(object_type,image_list,selected_image,label_col,part_disp,bbox_disp,obbox_disp):
    
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
          (0.961, 0.871, 0.702)]
    colors = (np.asarray(colors)*255)

    n = sum([part_disp,bbox_disp,obbox_disp])
    
    disp_image = []
    disp_list=[image_list[selected_image]]
    st.write(str(disp_list[0]))
    f = open(os.path.join(path,'src','visualization','part_label.json'),)
    part_labels = json.load(f)
    part_labels = part_labels[object_type]
    
    image = disp_list[0]
    with open(os.path.join(anno_source,object_type,'bbox',image+'.json')) as fp:
        annotation_dict = json.load(fp)
        
    num = len(annotation_dict.keys())
    object_idx = st.radio('Select the object you want to focus', list(range(num)),0)
    idx = list(annotation_dict.keys())[object_idx]
    im_x_min,im_y_min,im_x_max,im_y_max =annotation_dict[idx]['bbox']
    raw_image = cv2.imread(os.path.join(img_source,image+'.jpg'))
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    ob_image = raw_image.copy()
    if n >0:
        fig, axes = plt.subplots(1,n, figsize = (12,4))
        part_image = raw_image.copy()
        obb_image = raw_image.copy()
        
        t=-1

        part_dict = annotation_dict[idx]['parts']
        if part_dict and part_disp:
            t+=1
            if n>1:
                axes[t].imshow(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1], interpolation='none')
            else:
                axes.imshow(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1], interpolation='none')
            
            overlay = np.uint(np.zeros(part_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1].shape,'int'))
            for part in part_dict.keys():
                mask = np.repeat(np.expand_dims(np.matrix(part_dict[part]['mask']),axis=-1),3,axis=2)
                mask = np.uint(mask*np.expand_dims(np.matrix(colors[part_labels[part]-1]).astype('int'),axis=0))
                overlay+= mask[im_x_min:im_x_max+1,im_y_min:im_y_max+1]
                

            if n>1:
                axes[t].imshow(overlay,interpolation='none',alpha=0.2)
                axes[t].axis('off')  
                
            else:
                axes.imshow(overlay,interpolation='none',alpha=0.2)
                axes.axis('off')  


        if part_dict and obbox_disp:
            t+=1
            for part in part_dict.keys():
                contours,_ = cv2.findContours(np.uint8(np.matrix(part_dict[part]['mask'])),
                           cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

                for c in contours:
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    obb_image = cv2.drawContours(obb_image, [box],0,colors[part_labels[part]-1])
            if n>1:
                axes[t].imshow(obb_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1])
                axes[t].axis('off')  
            else:
                axes.imshow(obb_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1])
                axes.axis('off')  
            
        if part_dict and bbox_disp:
            bb_image = raw_image.copy()
            t +=1
            if n > 1:
                axes[t].imshow(raw_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1])
                axes[t].axis('off')
            else:
                axes.imshow(raw_image[im_x_min:im_x_max+1,im_y_min:im_y_max+1])
                axes.axis('off')  

            for part in part_dict.keys():

                x_min,y_min,x_max,y_max =part_dict[part]['bbox']
                rect = patches.Rectangle((y_min-im_y_min,x_min-im_x_min),y_max-y_min,x_max-x_min,
                                         linewidth=1,edgecolor=colors[part_labels[part]-1]/255,facecolor='none')

                if n>1:
                    axes[t].add_patch(rect)
                else:
                    axes.add_patch(rect)
                    
                    
        st.pyplot(fig)
    
    fig2, axes2 = plt.subplots(1,1, figsize = (4,4))
    contours, _ = cv2.findContours(np.uint8(np.matrix(annotation_dict[idx]['mask'])),
                               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ob_image = cv2.drawContours(ob_image, contours, -1,label_col.tolist() , 2)
    
    if n>0:
        axes2.imshow(ob_image)
        axes2.axis('off')  
    else:
        plt.imshow(ob_image)
        plt.axis('off') 
    
    st.pyplot(fig2)
     
    st.subheader('Ground Truth')
    

if __name__ == "__main__":
    
    curr_path = os.getcwd()
    path = Path(curr_path)
    anno_source = os.path.join(path,'src','PASCAL-VOC','xybb-objects')
    img_source = os.path.join(path,'src','PASCAL-VOC','scene')
    main()
