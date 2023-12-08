import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_al=pd.read_csv(r"data\raw\Participant1_Alert.csv")
df_dr=pd.read_csv(r"data\raw\Participant1_Drowsy.csv")

# label the drowsy for the driver
df_al['drowsy']=0
df_dr['drowsy']=1

# data integration
df=pd.concat([df_al,df_dr])

dp_list=['id','user_name','is_smoking','is_using_phone','is_wearing_seatbelt',
         'has_glasses','eyes_on_road','attentiveness','drowsiness',
         'aoi','aoi_x','aoi_y','aoi_z','Perclos','mouth_openness',
         'left_eye_openness_in_pixels','right_eye_openness_in_pixels',
         'eye_mode','fixation_length'] 

df.drop(dp_list,axis=1,inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

cal_list=['head_pose_yaw_cal', 'head_pose_pitch_cal', 'head_pose_roll_cal',
          'R_eye_x_cal', 'R_eye_y_cal', 'R_eye_z_cal', 
          'R_eye_gaze_yaw_cal', 'R_eye_gaze_pitch_cal',
          'L_eye_x_cal', 'L_eye_y_cal', 'L_eye_z_cal',
          'L_eye_gaze_yaw_cal', 'L_eye_gaze_pitch_cal', 
          'gaze_yaw_cal', 'gaze_pitch_cal',
          'gaze_x_cal', 'gaze_y_cal', 'gaze_z_cal',
          'head_coord_x_cal', 'head_coord_y_cal', 'head_coord_z_cal']

# drop the duplicated cols
df.drop(cal_list,axis=1,inplace=True)

# calculate the interval of a frametart=[]
start=[]
for i in range(len(df)):
    if df.frame_number.iloc[i]==1:
        start.append(i)

print(start)
end=start[1:]
end.append(len(df))
print(end)

xyz=['head_pose_yaw', 'head_pose_pitch', 'head_pose_roll',
      'R_eye_x', 'R_eye_y', 'R_eye_z', 
       'R_eye_gaze_yaw', 'R_eye_gaze_pitch',
       'L_eye_x', 'L_eye_y', 'L_eye_z', 
       'L_eye_gaze_yaw', 'L_eye_gaze_pitch',
       'gaze_yaw', 'gaze_pitch', 
       'gaze_x', 'gaze_y', 'gaze_z', 
       'head_coord_x', 'head_coord_y', 'head_coord_z']
pos=['pupil_dilation_ratio', 'right_eye_open_perc', 'left_eye_open_perc','drowsy']
binary=['camera_status', 'is_face_valid','is_face_human']


df_xyz=pd.DataFrame()
for i in range(len(start)):
        temp=df.iloc[start[i]:end[i]]
        a=pd.DataFrame((temp[xyz].max()-temp[xyz].min()))
        df_xyz=pd.concat([df_xyz,a.T])
df_xyz.shape

df_pos=pd.DataFrame()
for i in range(len(start)):
        temp=df.iloc[start[i]:end[i]]
        a=pd.DataFrame(temp[pos].mean())
        df_pos=pd.concat([df_pos,a.T])
df_pos.shape

df_dy=pd.concat([df_xyz,df_pos],axis=1)
df_dy.head()

frame_status=[]
camera_status=[]
is_face_valid=[]
is_face_human=[]
frame_number=[]
duration=[]
for i in range(len(start)):
        temp=df.iloc[start[i]:end[i]]
        frame_status.append(len(temp.loc[temp['frame_status']=='PROCESSED',:])/len(temp))
        camera_status.append(len(temp.loc[temp['camera_status']!='',:])/len(temp))
        is_face_valid.append(len(temp.loc[temp['is_face_valid']==True,:])/len(temp))
        is_face_human.append(len(temp.loc[temp['is_face_human']=='YES',:])/len(temp))
        frame_number.append(temp['frame_number'].count())
        duration.append(temp['timestamp'].max()-temp['timestamp'].min())
        
df_dy['frame_status']=frame_status
df_dy['camera_status']=camera_status
df_dy['is_face_valid']=is_face_valid
df_dy['is_face_humans']=is_face_human
df_dy['frame_number']=frame_number
df_dy['duration']=duration
df_dy['duration']=df_dy['duration'].dt.total_seconds()

df_dy.reset_index(inplace=True)
df_dy.drop('index',axis=1,inplace=True)
df_dy.head()

df_final=df_dy.loc[(df_dy['is_face_valid']>0.5)&(df_dy['is_face_humans']>0.5),:]

df_final.drop(['pupil_dilation_ratio'],axis=1,inplace=True)
df_final.drop(['frame_status','camera_status','is_face_valid','is_face_humans','frame_number','duration'],axis=1,inplace=True)
df_final.drop(['R_eye_gaze_yaw','R_eye_gaze_pitch','L_eye_gaze_yaw','L_eye_gaze_pitch',
               'R_eye_x','R_eye_y','R_eye_z','L_eye_x','L_eye_y','L_eye_z'],axis=1,inplace=True)

df_final.isnull().sum().sort_values(ascending=False)

df_final.dropna(how='any',inplace=True)