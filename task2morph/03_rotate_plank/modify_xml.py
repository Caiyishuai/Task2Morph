# -*- coding: utf-8 -*-
'''
description:
Changing task features
'''

import xml.etree.ElementTree as et

def modify_flip_box_pos(file_name,new_str_pos):
    file_name = '../../assets/'+file_name+".xml"
    doc = et.parse(file_name)
    root = doc.getroot()
    box_joint = root[4][0][0]
    box_pos =  box_joint.attrib['pos']
    print(box_pos)
    box_joint.set('pos', new_str_pos)
    doc.write(file_name, 'UTF-8')


