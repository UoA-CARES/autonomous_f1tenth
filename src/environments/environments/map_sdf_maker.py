import xml.etree.ElementTree as ET

#######################################################################################################
### CHANGE STUFF HERE TO MATCH ACTUAL STL FILES
TRACK_NAMES = ['test_track_01', 'test_track_02']
WIDTHS = [150, 200, 250, 300, 350]
IS_CREATING_INDIVIDUAL_TRACKS = True
OUTPUT_FILE_NAME = "multi_track_test_01"
########################################################################################################
# TRACK_NAMES = ['track_01', 'track_02', 'track_03', 'track_04', 'track_04', 'track_05', 'track_06']

BASE_SDF = '''<?xml version='1.0'?>
<sdf version="1.6">
  <world name="empty">

    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>10.0</real_time_factor>
    </physics>

    <plugin
      filename="gz-sim-physics-system"
      name="gz::sim::systems::Physics">
    </plugin>
    <plugin
        filename="gz-sim-sensors-system"
        name="gz::sim::systems::Sensors">
        <render_engine>ogre2</render_engine>
    </plugin>
    <plugin
      filename="gz-sim-user-commands-system"
      name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin
      filename="gz-sim-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster">
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    
    
  </world>
</sdf>'''

def create_track_element(track_name:str, offset_x:int):
  # Define the new model element with all its children and sub-elements
  model = ET.Element('model', attrib={'name': track_name})

  pose = ET.SubElement(model, 'pose')
  pose.text = f'{str(offset_x)} 0 0 0 0 0'

  static = ET.SubElement(model, 'static')
  static.text = 'true'

  # Link element
  link = ET.SubElement(model, 'link', attrib={'name': 'track'})

  # Visual element
  visual = ET.SubElement(link, 'visual', attrib={'name': 'track_vis'})
  geometry = ET.SubElement(visual, 'geometry')

  # Mesh under visual
  mesh_visual = ET.SubElement(geometry, 'mesh')
  uri_visual = ET.SubElement(mesh_visual, 'uri')
  uri_visual.text = f'model://src/environments/meshes/{track_name}.stl'
  scale_visual = ET.SubElement(mesh_visual, 'scale')
  scale_visual.text = '0.001 0.001 0.001'

  # Material element
  material = ET.SubElement(visual, 'material')
  ambient = ET.SubElement(material, 'ambient')
  ambient.text = '0.8 0.8 0.8 1'
  diffuse = ET.SubElement(material, 'diffuse')
  diffuse.text = '0.8 0.8 0.8 1'
  specular = ET.SubElement(material, 'specular')
  specular.text = '0.8 0.8 0.8 1'

  # Collision element
  collision = ET.SubElement(link, 'collision', attrib={'name': 'track_col'})
  geometry_collision = ET.SubElement(collision, 'geometry')

  # Mesh under collision
  mesh_collision = ET.SubElement(geometry_collision, 'mesh')
  uri_collision = ET.SubElement(mesh_collision, 'uri')
  uri_collision.text = f'model://src/environments/meshes/{track_name}.stl'
  scale_collision = ET.SubElement(mesh_collision, 'scale')
  scale_collision.text = '0.001 0.001 0.001'

  return model

######################################################################################
if __name__ == "__main__":
  
  
  if IS_CREATING_INDIVIDUAL_TRACKS:

    for track_name in TRACK_NAMES:
      for width in WIDTHS:
        root = ET.fromstring(BASE_SDF)

        current_track_name = f"{track_name}_{str(width)}"
        element = create_track_element(current_track_name, offset_x=0)
        root.find('world').append(element)

        tree = ET.ElementTree(root)
        tree.write(f"{current_track_name}.xml")
    
  
  else:
    i = 0
    root = ET.fromstring(BASE_SDF)

    for track_name in TRACK_NAMES:
      for width in WIDTHS:
        current_track_name = f"{track_name}_{str(width)}"
        element = create_track_element(current_track_name, offset_x=i*30) #each track offset by 30m
        root.find('world').append(element)
        i += 1

    
    tree = ET.ElementTree(root)
    tree.write(f"{OUTPUT_FILE_NAME}.sdf")

