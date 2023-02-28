import os
import librosa
import python_speech_features
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier

JSON_PATH = "kittens.json"
audio_path = "kittens"


def create_mfccs(path, jsonpath):
    cats = {
        "mappings": [],
        "labels": [],
        "signals": [],
        "fs": [],
        "MFCCs": []
    }

    for (dirpath, dirnames, filenames) in os.walk(path):

        for f in filenames:
            filepath = os.path.join(dirpath, f)
            data, fs = librosa.load(filepath)
            data = data / np.max(data)
            data_m = data[0:44100]
            mfcc = python_speech_features.base.mfcc(signal=data_m,
                                                    samplerate=fs,
                                                    winlen=0.025,
                                                    winstep=0.01,
                                                    numcep=13,
                                                    nfilt=26,
                                                    nfft=1024,
                                                    lowfreq=0,
                                                    highfreq=None,
                                                    preemph=0.97,
                                                    ceplifter=22,
                                                    appendEnergy=True,
                                                    winfunc=np.hanning)

            delta = python_speech_features.base.delta(mfcc, 1024)
            delta2 = python_speech_features.base.delta(delta, 1024)
            features = np.concatenate((mfcc.T, delta.T, delta2.T))
            mapping = filepath.split("\\")[-1]

            label = None
            if 'eu' in filepath:
                label = 1
            elif 'mc' in filepath:
                label = 0
            elif 'pers' in filepath:
                label = 2

            cats["mappings"].append(mapping[:2])
            cats["labels"].append(label)
            cats["signals"].append(data.tolist())
            cats["fs"].append(fs)
            cats["MFCCs"].append(features.tolist())

    with open(jsonpath, "w") as fp:
        json.dump(cats, fp, indent=3)
    return cats


def show_specgrams(cats):
    for s in range(len(cats["signals"])):
        x = np.array(cats['signals'][s])
        plt.figure()
        plt.specgram(x=x / np.max(x), Fs=cats["fs"][s], NFFT=1024, mode='magnitude', noverlap=260, scale='dB',
                     cmap='gist_rainbow')
        plt.ylim(0, 600)
        plt.xlabel("Time [s]", fontsize=10)
        plt.ylabel("Frequency [Hz]", fontsize=10)
        plt.colorbar(format="%+2f", label='Power/frequency (dB/Hz)')
        if cats['mappings'][s] == 'eu':
            plt.title("European Shorthair", fontsize=11)
            plt.savefig("eu_" + str(s + 1) + "_.png")
        if cats['mappings'][s] == 'mc':
            plt.title("Maine Coon", fontsize=11)
            plt.savefig("mc_" + str(s - 35) + "_.png")
        if cats['mappings'][s] == 'pe':
            plt.title("Persian cat", fontsize=11)
            plt.savefig("pers" + str(s - 77) + "_.png")
        plt.show()


def classify(cats):

    x = cats['MFCCs']
    y = cats['labels']
    x = np.array(x)
    nsamples, nx, ny = x.shape
    x = x.reshape((nsamples, nx * ny))

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    # Binarize the output
    y_roc = label_binarize(y, classes=[1, 0, 2])
    n_classes = y_roc.shape[1]
    n_labels = ['European shorthair', 'Maine Coon', 'Persian']
    res_roc_auc_micro = []

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=18, test_size=0.2, shuffle='True')
    classifier = KNeighborsClassifier(n_neighbors=12, p=2, metric='euclidean', weights='distance')
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    disp = metrics.plot_confusion_matrix(classifier, x_test, y_test, cmap="Blues")
    disp.ax_.set_title("Confusion matrix")
    plt.savefig("cm_.png")
    plt.show()
    metrics.accuracy_score(y_test, pred)
    print(metrics.classification_report(y_test, pred))

    trainx, testx, trainy, testy = train_test_split(x, y_roc, random_state=18, test_size=0.2, shuffle='True')
    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=12, p=2, metric='euclidean', weights='distance'))
    clf.fit(trainx, trainy)
    y_score = clf.predict_proba(testx)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(testy[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(testy.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    res_roc_auc_micro.append(roc_auc["micro"])

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), lw=3)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label='{0} (area = {1:0.2f})'''.format(n_labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.savefig("roc.png")
    plt.show()


def main():
    data = create_mfccs(audio_path, JSON_PATH)
    show_specgrams(data)
    classify(data)


if __name__ == "__main__":
    main()
