import glob
import librosa
import numpy as np

# Urban Sound Dataset 특징 추출 모듈


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(filenames):
    rows = len(filenames)
    features, labels, groups = np.zeros((rows, 193)), np.zeros((rows, 1)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        try:
            sound_class = int(fn.split('/')[2].split('\\')[1].split('-')[1])
            if sound_class == 0 or sound_class == 9 or sound_class == 2 or sound_class == 4:
                continue
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            y_col = int(fn.split('/')[2].split('\\')[1].split('-')[1])
            group = int(fn.split('/')[2].split('\\')[1].split('-')[0])
            print(fn.split('/')[2].split('\\')[1] + " 특징 추출 완료 (%d 번째)" % (i + 1))
        except:
            print("오류 발생")
            print(fn)
        else:
            features[i] = ext_features
            labels[i] = y_col
            groups[i] = group
            i += 1
    return features, labels, groups


audio_files = []
for i in range(1, 11):
    audio_files.extend(glob.glob('UrbanSound8K/audio/fold%d/*.wav' % i))

print(len(audio_files))
for i in range(9):
    files = audio_files[i * 1000: (i + 1) * 1000]
    X, y, groups = parse_audio_files(files)
    # for r in y:
    #     if np.sum(r) > 1.5:
    #         print('error occured')
    #         break
    np.savez('urban_sound_%d' % i, X=X, y=y, groups=groups)
    print('urban_sound_%d 저장' % i)