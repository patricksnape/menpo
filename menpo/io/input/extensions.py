# A list of extensions that different importers support.
from .landmark import (LM3Importer, LANImporter, LM2Importer, BNDImporter,
                       LJSONImporter)
from .landmark_mesh import MeshPTSImporter
from .mesh import AssimpImporter, WRLImporter, MJSONImporter
from .image import PILImporter, ABSImporter
from .landmark_image import ImageASFImporter, ImagePTSImporter


mesh_types = {'.dae': AssimpImporter,
              '.3ds': AssimpImporter,
              '.ase': AssimpImporter,
              '.obj': AssimpImporter,
              '.ifc': AssimpImporter,
              '.xgl': AssimpImporter,
              '.zgl': AssimpImporter,
              '.ply': AssimpImporter,
              '.dxf': AssimpImporter,
              '.lwo': AssimpImporter,
              '.lws': AssimpImporter,
              '.lxo': AssimpImporter,
              '.stl': AssimpImporter,
              '.x': AssimpImporter,
              '.ac': AssimpImporter,
              '.md5': AssimpImporter,
              '.smd': AssimpImporter,
              '.vta': AssimpImporter,
              '.m3': AssimpImporter,
              '.3d': AssimpImporter,
              '.b3d': AssimpImporter,
              '.q3d': AssimpImporter,
              '.q3s': AssimpImporter,
              '.nff': AssimpImporter,
              '.off': AssimpImporter,
              '.raw': AssimpImporter,
              '.ter': AssimpImporter,
              '.mdl': AssimpImporter,
              '.hmp': AssimpImporter,
              '.ndo': AssimpImporter,
              '.ms3d': AssimpImporter,
              '.cob': AssimpImporter,
              '.scn': AssimpImporter,
              '.bvh': AssimpImporter,
              '.csm': AssimpImporter,
              '.xml': AssimpImporter,
              '.irrmesh': AssimpImporter,
              '.irr': AssimpImporter,
              '.md2': AssimpImporter,
              '.md3': AssimpImporter,
              '.pk3': AssimpImporter,
              '.mdc': AssimpImporter,
              # '.blend': AssimpImporter,
              '.wrl': WRLImporter,
              '.mjson': MJSONImporter}

image_types = {'.bmp': PILImporter,
               '.dib': PILImporter,
               '.dcx': PILImporter,
               '.eps': PILImporter,
               '.ps': PILImporter,
               '.gif': PILImporter,
               '.im': PILImporter,
               '.jpg': PILImporter,
               '.jpe': PILImporter,
               '.jpeg': PILImporter,
               '.pcd': PILImporter,
               '.pcx': PILImporter,
               '.png': PILImporter,
               '.pbm': PILImporter,
               '.pgm': PILImporter,
               '.ppm': PILImporter,
               '.psd': PILImporter,
               '.tif': PILImporter,
               '.tiff': PILImporter,
               '.xbm': PILImporter,
               # '.pdf': PILImporter,
               '.xpm': PILImporter,
               '.abs': ABSImporter}

all_image_types = {}
all_image_types.update(image_types)

all_mesh_and_image_types = {}
all_mesh_and_image_types.update(all_image_types)
all_mesh_and_image_types.update(mesh_types)

image_landmark_types = {'.asf': ImageASFImporter,
                        '.lm2': LM2Importer,
                        '.pts': ImagePTSImporter,
                        '.ptsx': ImagePTSImporter,
                        '.ljson': LJSONImporter}

mesh_landmark_types = {'.pts3': MeshPTSImporter,
                       '.lm3': LM3Importer,
                       '.lan': LANImporter,
                       '.bnd': BNDImporter,
                       '.ljson': LJSONImporter}

all_landmark_types = {}
all_landmark_types.update(image_landmark_types)
all_landmark_types.update(mesh_landmark_types)
