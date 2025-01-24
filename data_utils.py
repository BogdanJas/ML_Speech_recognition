import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_ravdess_tess(ravdess_path, tess_path):
    data = []
    
    if os.path.exists(ravdess_path):
        male_count = 0
        female_count = 0
        
        for actor in os.listdir(ravdess_path):
            if actor.startswith('Actor_'):
                actor_num = int(actor.split('_')[1])
                gender = 'female' if actor_num % 2 == 0 else 'male'
                if gender == 'male':
                    male_count += 1
                else:
                    female_count += 1
                    
                actor_path = os.path.join(ravdess_path, actor)
                for file_name in os.listdir(actor_path):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(actor_path, file_name)
                        parts = file_name.split('-')
                        emotion_code = parts[2]
                        intensity = parts[3]
                        
                        emotion_map = {
                            '01': 'neutral', '02': 'calm', '03': 'happy',
                            '04': 'sad', '05': 'angry', '06': 'fear',
                            '07': 'disgust', '08': 'surprised'
                        }
                        emotion = emotion_map.get(emotion_code, 'unknown')
                        
                        data.append({
                            'path': file_path,
                            'emotion': emotion,
                            'source': 'RAVDESS',
                            'gender': gender,
                            'intensity': 'strong' if intensity == '02' else 'normal'
                        })
    
    if os.path.exists(tess_path):
        for file_name in os.listdir(tess_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(tess_path, file_name)
                parts = file_name.split('_')
                if len(parts) >= 3:
                    word = parts[1]
                    emotion = parts[2].split('.')[0].lower()
                    if emotion == 'ps':
                        emotion = 'surprised'
                    
                    data.append({
                        'path': file_path,
                        'emotion': emotion,
                        'source': 'TESS',
                        'gender': 'female',
                        'word': word
                    })
    
    return pd.DataFrame(data)

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, duration=3, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)))
        
        return mfccs
    except Exception as e:
        return None

def prepare_data(df):
    features = []
    valid_indices = []
    
    for idx, row in enumerate(df.itertuples()):
        feature = extract_features(row.path)
        if feature is not None:
            features.append(feature)
            valid_indices.append(idx)
    
    df_valid = df.iloc[valid_indices]
    X = np.array(features)
    
    le = LabelEncoder()
    y = le.fit_transform(df_valid['emotion'])
    y = to_categorical(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, le