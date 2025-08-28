'version: 1.0.0'
# import napari
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from pathlib import Path
from quickpiv import generate_pseudotracks
import ast 

version = 2.0

class TrackObj():

    def __init__(self,input_path,manualtracks_filename,pseudotracks_filename=None,registration_name = None):

        self.input_dir = input_path
        self.manual_filename = str(manualtracks_filename.stem)
        self.manual_tracks = None
        self.starting_points = None
        self.pseudo_tracks_name = pseudotracks_filename
        self.pseudo_tracks = None
        self.registration_name = None
        self.rotation = []
        self.translation = []
        self.pixel_dims = (0.406,0.406,2)
        self.track_longform = False #when false tracks are in their orignal form as network graph, when true tracks have been split so that each division is a new track from start to finish
        self.interSize = (40,40,8)
        self.tstep = 1
        self.track_registered = False
        self.registered_tracks = None
        self.nn_radius = None
        self.track_alignment = None
        self.selected_pseudotracks = None

    def get_manual_tracks(self):

        
        if "Link" or "Spot" in self.manual_filename:
            self.manual_filename = ("-").join(self.manual_filename.split("-")[:-1])
        
        manual_tracks_links = pd.read_csv(self.input_dir.joinpath(self.manual_filename+"-Link.csv"),header=[0,1,2])
        manual_tracks_links.columns = [" ".join(col[:2]) for col in manual_tracks_links.columns]

        manual_tracks_spot = pd.read_csv(self.input_dir.joinpath(self.manual_filename+"-Spot.csv"),header=[0,1,2])
        cols = []
        for col in manual_tracks_spot.columns:
            if "Unnamed" in col[1]:
                cols.append(col[0])
            else:
                cols.append(" ".join(col[:2]))
        manual_tracks_spot.columns = cols
        
        manual_tracks_spot.set_index("Label",inplace=True,drop=False)
        manual_tracks_spot['parent_spot'] = -1
        for link_index in manual_tracks_links.index:
            manual_tracks_spot.loc[int(manual_tracks_links.loc[link_index,"Link target IDs Target spot id"]),"parent_spot"] = int(manual_tracks_links.loc[link_index,"Link target IDs Source spot id"])

        manual_tracks_spot.columns = ['label','id','detection_quality','n_outgoing','n_links','n_incoming','time','x','y','z','radius','parent_spot']

        manual_tracks_spot.loc[:,'z'] = manual_tracks_spot.z / (self.pixel_dims[2]/self.pixel_dims[0]) # the way mastodon and napari render the H5 file, they scale the z pixels by the scale factor of anisotropy so we have to undo that
        
        manual_tracks_spot.loc[:,['x','y','z']] = manual_tracks_spot.loc[:,['x','y','z']] * self.pixel_dims

        self.manual_tracks = manual_tracks_spot
        return 
    
    # def get_starting_points(self):

    #     assert (self.manual_tracks is not None), "THIS OBJECT DOES NOT CONTAIN MANUAL TRACKS. To generate manual tracks use get_manual_tracks() method"

    #     tracks = self.manual_tracks

    #     sp = tracks[tracks.time == 0]
    #     sp = sp.set_index("track_id",drop=False)
    #     sp = sp.sort_index()

    #     sp_scaled = np.asarray(sp.loc[:,['x','y','z']])/self.interSize

    #     self.starting_points =  np.round(sp_scaled).astype(np.uint8)

    #     return 

    def split_divisions(self):

        tracks = self.manual_tracks
        tracks.set_index("label",inplace=True,drop=False)
        track_ends = tracks[tracks.n_outgoing == 0]

        spot_indicies = []
        # starting at the end of every track, follow the track upwards through the parent spots until you reach the start of the track, noting down the label of each spot
        for end in track_ends.index:
            spot_index = []
            lab = end
            ps = None
            while ps != -1:
                spot_index.append(lab)
                ps = tracks.loc[lab,"parent_spot"]
                lab = ps
            spot_indicies.append(spot_index)
        
        all_tracks = [tracks.loc[trk] for trk in spot_indicies]
        for trk,i in zip(all_tracks,range(len(all_tracks))):
            trk['track_id'] = i
        
        all_tracks = pd.concat(all_tracks,axis=0)

        self.manual_tracks = all_tracks.sort_index()
        self.track_longform = True

        return 

    def get_pseudo_tracks(self):

        if self.pseudo_tracks_name == None:
            pseudo_tracks = pd.read_csv(self.input_dir.joinpath("pseudotracks.csv"))
        elif self.pseudo_tracks_name != None and type(self.pseudo_tracks_name) == str:
            pseudo_tracks = pd.read_csv(self.input_dir.joinpath(self.pseudo_tracks_name))

        self.pseudo_tracks = pseudo_tracks
        return

    def get_registration(self):

        if self.registration_name == None:
            reg = pd.read_csv(self.input_dir.joinpath("registration.csv"))
        elif self.registration_name != None and type(self.registration_name) == str:
            reg = pd.read_csv(self.input_dir.joinpath(self.registration_name))

        for i in range(len(reg)):
            rotation_tuple = ast.literal_eval(",".join(reg.loc[0,'rotation'].split(' ')))
            rotation_matrix = np.array(rotation_tuple).reshape(3,3)
            
            translation_array = np.array(ast.literal_eval(",".join(reg.loc[0,'translation'].split(' '))))
            
            self.rotation.append(rotation_matrix)
            self.translation.append(translation_array)
    
    def register_manual_tracks(self):
        assert(len(self.rotation) > 0 and len(self.translation) > 0), "Can't find rotation or translation values, try running get_registraton() to load them in from file."
        # assert(self.manual_tracks != None),'Cannot find manual tracks, load them in using get_manual_tracks()'
        
        track_df = self.manual_tracks
        track_df.set_index('time',inplace=True,drop=False)
        self.manual_tracks.set_index('time',inplace=True,drop=False)
        
        for t in np.unique(track_df.index):
            if t == 0:
                continue
            
            transformed_tracks = np.dot(track_df.loc[t,['x','y','z']],self.rotation[t-1]) + self.translation[t-1]
            local_movement_tracks = track_df.loc[t,['x','y','z']] - (transformed_tracks - track_df.loc[t,['x','y','z']])
            track_df.loc[t,['x','y','z']] = local_movement_tracks
            
        
        track_df.reset_index(drop=True, inplace=True)
        self.registered_tracks = track_df
        self.track_registered = True
        
    def norm_dotprod(self,vector1, vector2):
        norm_vector1 = vector1/np.sqrt(np.sum(vector1**2))
        norm_vector2 = vector2/np.sqrt(np.sum(vector2**2))

        return np.dot(norm_vector1,norm_vector2)

    def vector(self,pt1,pt2):
        return pt2 - pt1

    def evaluate_tracks(self,tstep=None,nn_radius=10):
        assert(self.track_longform == True), "Tracks are still in their form as a network graph, run split_divisions to create individual tracks"
        assert(self.track_registered == True), "Manual Tracks have not been registered, run get_registration and register_manual_tracks so they can be correctly aligned with the pseudotracks"
        
        if tstep != None:
            self.tstep = tstep

        self.nn_radius = nn_radius


        manual_tracks = self.registered_tracks
        manual_tracks.reset_index(inplace=True,drop=True)
        pseudo_tracks = self.pseudo_tracks

        all_dot_prod = []
        selected_pseudotracks = []
        for trackid in np.unique(manual_tracks.track_id):
            track = manual_tracks[manual_tracks.track_id == trackid]
            track.set_index("time",inplace=True,drop=False)
            #track.sort_index(inplace=True)
            
            pseudo_tracks.set_index("time",inplace=True,drop=False)

            trk_start = track.index.min()
            #need to account for the fact that there are gaps in the manual tracks
            pseudo_tree = KDTree(pseudo_tracks.loc[trk_start,['x','y','z']])
            nn_index = pseudo_tree.query_ball_point(np.asarray(track.loc[trk_start,['x','y','z']]),r=nn_radius)
            nn_id = np.asarray(pseudo_tracks.loc[trk_start,'track_id'].iloc[nn_index])
            print(nn_id)
            for t1,t0 in zip(np.sort(track.index)[self.tstep:],np.sort(track.index)[:-self.tstep]):
                print(trackid,'-->',t0)
                for ps_id in nn_id:
                    pstrack = pseudo_tracks[pseudo_tracks.track_id == ps_id]
                    #print(track.loc[t,['x','y','z']])
                    manual_vector = self.vector(track.loc[t0,['x','y','z']],track.loc[t1,['x','y','z']])
                    pseudo_vector = self.vector(pstrack.loc[t0,['x','y','z']],pstrack.loc[t1,['x','y','z']])
                    dotprod = self.norm_dotprod(manual_vector,pseudo_vector)
                    all_dot_prod.append([dotprod,trackid,t0,ps_id])
                    
                    selected_pseudotracks.append(pstrack)
        
        self.track_alignment = pd.DataFrame(all_dot_prod,columns=['vector_alignment','track_id','time','ps_nn_id'])
        self.selected_pseudotracks = pd.concat(selected_pseudotracks,axis=0)
        return pd.DataFrame(all_dot_prod,columns=['vector_alignment','track_id','time','ps_nn_id'])

    def save_data(self,data_types_list):
        for dt in data_types_list:
            if dt == 'registered_tracks':
                try:
                    self.registered_tracks.reset_index(inplace=True, drop =False)
                except ValueError:
                    self.registered_tracks.reset_index(inplace=True,drop=True)
                
                self.registered_tracks.to_csv(self.input_dir.joinpath(self.manual_filename+'_registered.csv'),index=False)
                print('Registered manual tracks saved to file')

            elif dt == 'track_alignment':
                try:
                    self.track_alignment.reset_index(inplace=True, drop =False)
                except ValueError:
                    self.track_alignment.reset_index(inplace=True,drop=True)
                
                self.track_alignment.to_csv(self.input_dir.joinpath(self.manual_filename+'_alignment.csv'),index=False)
                print('Manual and psuedotracks alignment table saved to file')

            elif dt == 'selected_pseudotracks':
                try:
                    self.selected_pseudotracks.reset_index(inplace=True, drop =False)
                except ValueError:
                    self.selected_pseudotracks.reset_index(inplace=True,drop=True)
                
                self.selected_pseudotracks.to_csv(self.input_dir.joinpath(self.manual_filename+'_selectedpseudotracks.csv'),index=False)
                print('Neighboring pseudotracks saved to file')
            
            else:
                print('\nName of dataset doesnt match any TrackObj attributes which can be saved.\n Try again with a list of any of the following:\n\n"registered_tracks"\n"track_alignment"\n"selected_pseudotracks"')


track = TrackObj(Path("C:\\Users\\User\\OneDrive - University of Cambridge\\scripts\\vector_field_analysis\\expzacy0032\\piv_output"),Path("expzacy0032_Position1_Settings1_202507_clean_MastodonTable-Link.csv"))

track.get_manual_tracks()
track.split_divisions()
# track.manual_tracks.to_csv(track.input_dir.joinpath(track.manual_filename+'_unregtracks.csv'),index=False)
track.get_pseudo_tracks()
track.get_registration()
track.register_manual_tracks()


# track.save_data(['registered_tracks'])
# print(track.registered_tracks.loc[:,['x','y','z']])
# print(track.pseudo_tracks.loc[:,['x','y','z']])
align = track.evaluate_tracks(tstep=5,nn_radius=20)
track.save_data(['registered_tracks','track_alignment','selected_pseudotracks'])

