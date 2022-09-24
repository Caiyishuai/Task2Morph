# -*- coding: utf-8 -*-
'''
description:
Changing task features
'''
import xml.etree.ElementTree as et

def modify_flip_box_pos(file_name,new_str_pos):
    # <joint name="box" type="revolute" axis = "0 -1 0 " pos="10 4 -10" quat="1 0 0 0" damping="1e4"/> 
    file_name = '../../../assets/'+file_name+".xml"
    doc = et.parse(file_name)
    root = doc.getroot()
    box_joint = root[4][0][0]
    # box_pos =  box_joint.attrib['pos']
    # print(box_pos)
    box_joint.set('pos', new_str_pos)
    doc.write(file_name, 'UTF-8')



def modify_flip_box_size(file_name,a,b,c):
    # body name="box" type="cuboid" size="8 3 3" pos="4 0 1.5" quat="1 0 0 0" density="1.0" 
    # size = "a b c" pos="a/2,0,c/2"
    file_name = '../../../assets/'+file_name+".xml"
    doc = et.parse(file_name)
    root = doc.getroot()
    box_body = root[4][0][1]
    box_body.set("size", str(a)+" "+str(b)+" "+str(c))
    box_body.set("pos", str(a/2)+" 0 "+str(c/2))
    
    #<endeffector joint="box" pos="8 0 2" radius="0.2"/>
    # pos = "a,0,c*(2/3)"
    variable_box = root[-1][-1]
    variable_box.set('pos', str(a)+" 0 "+str(c*(2/3)))
    doc.write(file_name, 'UTF-8')
    

