import pandas as pd
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns


def normalize(data):
    normalized = []
    for i in data:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

def preprocessing(data_input):
    tokenizer.fit_on_texts(data_input)
    return tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(normalize(data_input)), padding = 'post', maxlen=256)

max_vocab = 10000
tokenizer = Tokenizer(num_words=max_vocab) #Global variable

try:
    loaded_model = tf.keras.models.load_model("FakeNews.model") # If trained model is available
except: #Run training in an except block if trained model is unavailable
    print("Trained model not found, starting training.\n")
    print(tf.__version__)
    print(tf.config.list_physical_devices()) # Check to see if GPU-accelerated learning is possible
    plt.style.use('ggplot') # matplotlib

    #Load the datasets
    fake_df = pd.read_csv('Fake.csv')
    real_df = pd.read_csv('True.csv')

    # - OPTIONAL - Check dataset for null values
    print(fake_df.isnull().sum())
    print(real_df.isnull().sum())

    # - OPTIONAL - Check for unique values
    print(fake_df.subject.unique())
    print(real_df.subject.unique())

    # - OPTIONAL - Remove Date and subject from the datasets
    fake_df.drop(['date', 'subject'], axis=1, inplace=True)
    real_df.drop(['date', 'subject'], axis=1, inplace=True)

    # - OPTIONAL - FIGURE 1: Plots 2 bars with fake news and real news, to see the distribution of the articles.
    plt.figure(figsize=(10, 5))
    plt.bar('Fake News', len(fake_df), color='orange')
    plt.bar('Real News', len(real_df), color='green')
    plt.title('Distribution of Fake News and Real News', size=15)
    plt.xlabel('News Type', size=15)
    plt.ylabel('# of News Articles', size=15)

    # - REQUIRED - Fake news is classified as 0, Real as 1
    fake_df['class'] = 0
    real_df['class'] = 1



    # - REQUIRED - Combine Title with text for easier processing
    news_df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)
    print(news_df)
    news_df['text'] = news_df['title'] + news_df['text']
    news_df.drop('title', axis=1, inplace=True)

    # - REQUIRED - Split data into training and testing

    features = news_df['text']
    targets = news_df['class']
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=18)

    tokenizer.fit_on_texts(X_train)
    # tokenize it into vectors then normalize data using the normalization function
    # and finally add padding to make sure each article is of the same length

    X_train = preprocessing(X_train)
    X_test = preprocessing(X_test)



    #Build the neural network

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_vocab, 128), #Choses  the max_vocab variable as the max length of each article
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    print(model.summary())

    #Optional early stop, when validation loss no longer improves

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    #Train the model, epochs = "for loop through the training dataset" batch size = how many training batches
    history = model.fit(X_train, y_train, epochs=15,validation_split=0.1, batch_size=64, shuffle=True, callbacks=[early_stop])

    #Visualize learning

    history_dict = history.history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = history.epoch

    # FIGURE 3: Shows accuracy improvement for each epoch
    plt.figure(figsize=(12,9))
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy', size=20)
    plt.xlabel('Epochs', size=20)
    plt.ylabel('Accuracy', size=20)
    plt.legend(prop={'size': 20})
    plt.ylim((0.5,1))
    print(plt.show())


    #Evaluate the testing set
    model.evaluate(X_test, y_test)
    pred = model.predict(X_test)
    binary_predictions = []

    for i in pred:
        if i >= 0.5:
            binary_predictions.append(1)
        else:
            binary_predictions.append(0)
    print('Accuracy on testing set:', accuracy_score(binary_predictions, y_test))
    print('Precision on testing set:', precision_score(binary_predictions, y_test))
    print('Recall on testing set:', recall_score(binary_predictions, y_test))

    #Confusion matrix

    matrix = confusion_matrix(binary_predictions, y_test, normalize='all')
    plt.figure(figsize=(16, 10))
    ax= plt.subplot()
    sns.heatmap(matrix, annot=True, ax = ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted Labels', size=20)
    ax.set_ylabel('True Labels', size=20)
    ax.set_title('Confusion Matrix', size=20)
    ax.xaxis.set_ticklabels([0,1], size=15)
    ax.yaxis.set_ticklabels([0,1], size=15)

    #Save the weights for visualization

    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape) # shape: (vocab_size, embedding_dim)

    word_index = list(tokenizer.word_index.keys())
    word_index = word_index[:max_vocab-1]

    # Write to file so we can use tensorflow's embedding projector to visualize what our network learned. This is only based on the fake news dataset.

    import io
    out_v = io.open('fakenews_vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('fakenews_meta.tsv', 'w', encoding='utf-8')

    for num, word in enumerate(word_index):
      vec = weights[num+1] # skip 0, it's padding.
      out_m.write(word + "\n")
      out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()
    model.save("FakeNews.model")


article = [str(input("Copy paste an article"))]

def predict_article(article):

    try:
        try: # use a loaded model if it exists
            prediction = loaded_model.predict(preprocessing(article))
        except: # use the newly trained model
            prediction = model.predict(preprocessing(article))
    except ValueError:
        print("Error: Unexpected result of 'predict_function'")
    print(prediction)
    if prediction >= 1:
        return "Most likely Real News"

    elif 0.5 < prediction > 1:
        return "Possibly fake news, check sources"
    elif prediction < 0.5:

        return "Most likely fake news"


print(predict_article(article))








