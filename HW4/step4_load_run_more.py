import os
import shutil
import datetime
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt

reloaded_model = tf.saved_model.load('imdb_bert_128')



def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()


examples = [
    'this is such an amazing movie!',  # this is the same sentence tried earlier
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...',
    'This restaurant has the best pasta in town!',
    'I found the book to be quite boring and hard to finish.',
    'The customer service was okay, nothing special.',
    'My new laptop is absolutely phenomenal!',
    'My new laptop is so phenomenal!',
    'My new laptop is so phenomenal',
    'My new laptop is so extraordinary',
    'My new laptop is phenomenal!',
    'My new laptop is so cool!',
    'The concert was a total disappointment.',
    'The garden scent was overwhelming, not in a pleasant way.',
    'This new software update is a labyrinthine mess.',
    'The poetry reading was an unexpected delight, like discovering a hidden gem.',
    'The virtual reality experience was disorienting, felt too unreal.',
    'I fucking hate you',
    'I fucking love you',
    'I f***ing hate you',
    'I f***ing love you',
    'f**k you',
    'love you',
    'I hate how much I love you',
    'Its frustrating how right you always are.'

]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)


# serving_results = reloaded_model \
#             .signatures['serving_default'](tf.constant(examples))

# serving_results = tf.sigmoid(serving_results['classifier'])

# print_my_examples(examples, serving_results)



