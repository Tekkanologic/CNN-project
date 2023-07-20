import tensorflow as tf
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import os
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns

# Define data source and model
from audio_data_EVL import PrepData
from models import build_model_CNN3

# CPU only mode
### os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_mfccs = 13  # THIS IS PERMANENT DO NOT CHANGE

'GPU setup'
tf.compat.v1.logging.set_verbosity('FATAL')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


'initialize wandb logging and declare hyperparameters'
run = wandb.init(project="audio-cnn",
                 config={
                    "epochs": 15,
                    "learning_rate": 0.00001,
                    "batch_size": 64,
                    "dropout": 0.3,
                    "dense_layer_nodes": 1177,
                    "num_filters": 17
                 })
config = wandb.config

# Get datasets
data = PrepData(valid_size=0.30, test_size=0.20, batch_size=config.batch_size)
train, test, valid, maxlen = data.prep_data()

# create network
tf.keras.backend.clear_session()
input_shape_CNN = (config.batch_size, maxlen, num_mfccs, 1)
input_shape_RNN = (config.batch_size, maxlen, num_mfccs)
print(input_shape_RNN)

# Parameters
dropout = config.dropout
dense_layers = config.dense_layer_nodes
num_filters = config.num_filters

# build model
model = build_model_CNN3()
model.build(input_shape_CNN)

# compile model
optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) # loss='categorical_crossentropy'
model.summary()

# Train network
train_history = model.fit(train, validation_data=valid, batch_size=config.batch_size, callbacks=[WandbCallback()],
                          epochs=config.epochs)

'''
# Plot history
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

# test network on unseen data
test_loss, test_acc = model.evaluate(test, verbose=1, callbacks=[WandbCallback()])
test_error_rate = round((1 - test_acc) * 100, 2)
print('\n Test accuracy:', test_acc)
print('Test error rate: ', test_error_rate)

# -------------------------------------------------------------------------------------------------------------------- #

# Make predictions
data = test.unbatch()
data = list(data.as_numpy_iterator())
data = np.array(data, dtype=object)
y = np.array(data[:, 1]).astype(int)
pred = model.predict(test)
preds = pred.round()
predr = pred.ravel()

# Plot confusion matrix
confusion_mtx = tf.math.confusion_matrix(y, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt="d", annot_kws={"size": 13})
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.savefig('conf_mat.png')
plt.close()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, preds.round()).ravel()
tn, fp, fn, tp = cm.ravel()
print("tn: ", tn)
print("fp: ", fp)
print("fn: ", fn)
print("tp: ", tp)

# Performance metrics
recall = tp / (tp + fn)
precision = tp / (tp + fp)
accuracy = (tp + tn) / (tn + fp + fn + tp)
f1 = 2 * (precision * recall) / (precision + recall)
specificity = tn / (tn + fp)
print("Precision: ", round((100 * precision), 2), "%")
print("Recall: ", round((100 * recall), 2), "%")
print("F1 score: ", round(f1, 2))
print("Accuracy: ", round((100 * accuracy), 2), "%")
print("specificity: ", round((100 * specificity), 2), "%")

fpr, tpr, threshold = roc_curve(y, preds, drop_intermediate=True)
fnr = 1 - tpr
###print("threshold: ", threshold)
###print("false positive rate: ", fpr)
###print("true positive rate: ", tpr)
###print("false negative rate: ", fnr)

# eer_threshold = threshold(np.nanargmin(np.absolute((fnr - fpr))))
# The EER is defined as FPR = 1 - PTR = FNR
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print("EER: ", EER)
roc_auc = roc_auc_score(y, preds)
print("ROC AUC:", roc_auc)

# plot ROC curve figures
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {:.3f})'.format(roc_auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('ROC.png')
plt.close()

# -------------------------------------------------------------------------------------------------------------------- #

wandb.log({
    "test loss": test_loss,
    "test acc": test_acc,
    "test error rate": test_error_rate,
    "total params": model.count_params(),
    "Precision:": precision,
    "Recall": recall,
    "specificity": specificity,
    "Accuracy": accuracy,
    "F1 score": f1,
    "EER": EER,
    "AUC": roc_auc,
    "cm image": [wandb.Image('conf_mat.png', caption="confusion matrix")],
    "roc image": [wandb.Image('ROC.png', caption="ROC curve")]
    ### "roc": wandb.plot.roc_curve(y, preds, labels=["bonafide", "spoof"]),
    ### "conf_mat": wandb.plot.confusion_matrix(y_true=y, preds=preds, class_names=["bonafide", "spoof"])
})

# Save model to wandb
model.save(os.path.join(wandb.run.dir, "model.h5"))
model.save('model/audio_cnn.h5')