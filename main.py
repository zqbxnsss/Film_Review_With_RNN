import tensorflow as tf

class RNN(tf.keras.Model):
    def __init__(self, units):
        super(RNN, self).__init__()
        self.state0 = [tf.zeros([batch_size, units])]
        self.state1 = [tf.zeros([batch_size, units])]
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.rnn_cell0 = tf.keras.layers.SimpleRNNCell(units=units, dropout=0.2)
        self.rnn_cell1 = tf.keras.layers.SimpleRNNCell(units=units, dropout=0.2)
        self.out_layer = tf.keras.layers.Dense(1)
    def call(self, inputs, training=None):
        """
        :param inputs: [b,80]
        :param training:
        :return:
        """
        state0 = self.state0
        state1 = self.state1
        x = self.embedding(inputs)
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training=training)
            out1, state1 = self.rnn_cell1(out0, state1, training=training)
        x = self.out_layer(out1)
        prob = tf.sigmoid(x)
        return prob


def get_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_len)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_len)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # train data
    train_db = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_db = train_db.shuffle(10000).batch(batch_size, drop_remainder=True)
    # test data
    test_db = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_db = test_db.batch(batch_size, drop_remainder=True)
    return train_db, test_db


if __name__ == '__main__':
    # parameters
    total_words = 10000
    max_review_len = 80
    embedding_len = 100
    batch_size = 1024
    learning_rate = 0.0001
    iteration_num = 20
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.losses.BinaryCrossentropy(from_logits=True)
    model = RNN(64)
    model.build(input_shape=[None, 64])
    print(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # get data
    train_db, test_db = get_data()
    model.fit(train_db, epochs=iteration_num, validation_data=test_db, validation_freq=1)
