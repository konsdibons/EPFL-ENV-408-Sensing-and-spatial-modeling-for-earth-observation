# %%
import numpy as np
import xml.etree.ElementTree as ET
import json

def parse_xml(xml_path):
    markers_parse = ET.parse(xml_path)
    markers_block = markers_parse.getroot()

    mark_ids = {'m_id': [], 'm_label': []}
    mark_2D = {'m_label': [], 'u': [], 'v': []}

    for markers in markers_block[0].findall('markers'):    
        for m in markers:
            mark_ids['m_id'].append(int(m.attrib.get("id")))
            mark_ids['m_label'].append(m.attrib.get("label"))
            
    for frames in markers_block[0].findall('frames'):  
        for m_img in frames[0][0]:
            this_marker = int(m_img.attrib.get("marker_id"))
            this_label  = mark_ids['m_label'][mark_ids['m_id'].index(this_marker)]

            for location in m_img:
                if json.loads(location.attrib.get("pinned").lower()):
                    this_camera = int(location.attrib.get("camera_id"))

                    mark_2D['m_label'].append(this_label)
                    mark_2D['u'].append(float(location.attrib.get("x")))
                    mark_2D['v'].append(float(location.attrib.get("y")))
    uv = np.array([mark_2D['u'], mark_2D['v']]).T
    return mark_2D['m_label'], np.array([mark_2D['u'], mark_2D['v']]).T
# %%
