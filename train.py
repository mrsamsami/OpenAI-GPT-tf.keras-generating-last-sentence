import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from Transformer import Transformer
from Utils import iter_data, get_validation
import time

def train_plot(loss_results, accuracy_results):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].plot(accuracy_results)
    axes[1].set_xlabel("Epoch", fontsize=14);

def save(model, train_loss_results, validation_loss_results):
    with open('train_loss_results.pkl', 'wb') as pkl:
        pickle.dump(train_loss_results, pkl)

    with open('validation_loss_results.pkl', 'wb') as pkl:
        pickle.dump(validation_loss_results, pkl)

    model.save_weights("model/model.h5")

def train(model, learning_rate = 0.00025, n_epochs = 100, n_batch = 64, n_ctx = 512,
          train_steps = 50, validation_steps = 1000):

    X = tf.placeholder(tf.int32, [None, n_ctx, 2])
    M = tf.placeholder(tf.int32, [None, n_ctx])
    logits, losses = model([X, M])
    model.load_weights('model/model.h5')

    train_losses = []
    train_loss_results = []
    validation_loss_results = []

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    opt = optimizer.minimize(losses)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    step = 0
    train_generator = iter_data(n_batch, n_epochs)
    start = time.time()
    for tokens, masks in train_generator:
        step += 1
        _, train_logits, train_loss = sess.run([opt, logits, losses], {X: tokens, M: masks})
        train_losses.append(train_loss)

        if step % train_steps == 0:
            train_loss_results.append(sum(train_losses) / len(train_losses))
            train_losses = []
            print(f"step {step} : time= {time.time() - start}, train loss = {train_loss_results[-1]}")

        if step % validation_steps == 0:
            validation_generator = iter_data(n_batch, train = False)
            validation_losses = []
            for validation_tokens, validation_masks in validation_generator:
                validation_losses.append(sess.run(losses, {X: validation_tokens, M: validation_masks}))

            validation_loss_results.append(sum(validation_losses) / len(validation_losses))
            print(f"step {step} : time= {time.time() - start}, validation loss = {validation_loss_results[-1]}")
            save(model, train_loss_results, validation_loss_results)

if __name__ == "__main__":
    model = Transformer("Model", 40478)
    train(model)








